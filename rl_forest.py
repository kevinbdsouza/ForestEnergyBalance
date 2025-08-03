"""
Boreal Forest Management RL Environment
=======================================

This script implements a reinforcement learning pipeline for adaptive boreal
forest management. The environment is designed based on the following principles:

1.  **Time-Scale Separation**: The RL agent makes a decision once per simulated
    year, while the underlying physical model (`energy_balance_rc.py`) runs at a
    15-minute resolution.

2.  **Discrete Action Space**: The agent has two primary levers:
    *   **Stand Density**: Thinning or planting, changing stems per hectare.
    *   **Species Mix**: Nudging the forest composition towards deciduous or
        coniferous species.
    This is implemented as a `gym.spaces.MultiDiscrete([5, 3])`.

3.  **Stochasticity**: Each episode represents a 30-year rollout under a unique,
    stochastically generated weather sequence. This forces the agent to learn
    policies that are robust to climate variability.

4.  **Multi-Objective Reward**: The reward function balances two objectives:
    *   Maximizing carbon sequestration.
    *   Minimizing soil thaw (measured in thaw-degree-days).
    The scalarized reward is `R_t = w_C * Δcarbon_t - w_T * thaw_degree_days_t`.

5.  **Algorithm**: The recommended training algorithm is PPO (Proximal Policy
    Optimization) with Generalized Advantage Estimation (GAE), suitable for
    long-horizon tasks with sparse rewards.

This setup allows the agent to learn management strategies that account for
long-term ecological feedbacks and are robust to an uncertain future climate.
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

# Import the refactored simulation model
# Note: We will need to modify energy_balance_rc.py for this to work
import energy_balance_rc as ebm


class ForestEnv(gym.Env):
    """
    A Gym environment for boreal forest management.
    """
    metadata = {'render_modes': []}

    # --- Constants ---
    EPISODE_LENGTH_YEARS = 50
    MIN_STEMS_HA = 100
    MAX_STEMS_HA = 2500

    # --- Action Mapping ---
    # Δdensity ∈ {−300, −150, 0, +150, +300} stems ha⁻¹
    DENSITY_ACTIONS = [-300, -150, 0, 150, 300]
    # Δmix ∈ {−0.1, 0, +0.1} (conifer↔deciduous fraction)
    MIX_ACTIONS = [-0.1, 0, 0.1]

    # --- Reward Weights ---
    W_CARBON = 1.0  # Weight for carbon sequestration
    W_THAW = 0.01   # Weight for thaw penalty

    def __init__(self, config: dict | None = None):
        super().__init__()
        print("--- ForestEnv.__init__ called ---")

        self.config = config if config is not None else {}

        # --- Action and Observation Spaces ---
        self.action_space = spaces.MultiDiscrete([
            len(self.DENSITY_ACTIONS),
            len(self.MIX_ACTIONS)
        ])

        # Observation space: [year, norm_density, mix_fraction, norm_carbon_stock]
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32), # Max carbon ~50kg/m2 (more realistic for current model)
            dtype=np.float32
        )

        # To be initialized on reset
        self.simulator = None
        self.year = 0
        self.stem_density = 0
        self.conifer_fraction = 0.0
        self.biomass_carbon_kg_m2 = 0.0
        self.soil_carbon_kg_m2 = 0.0
        self.cumulative_thaw_dd = 0.0

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        print("--- ForestEnv.reset called ---")
        super().reset(seed=seed)

        # --- Initialize Forest State ---
        self.year = 0
        self.stem_density = 800  # Starting stems per hectare
        self.conifer_fraction = 0.5  # Start with a 50/50 mix
        # Initialize separate carbon pools (kg C/m^2)
        self.biomass_carbon_kg_m2 = 10.0
        self.soil_carbon_kg_m2 = 5.0
        self.cumulative_thaw_dd = 0.0
        
        # Validate initial state
        assert self.stem_density >= self.MIN_STEMS_HA, f"Initial stem density {self.stem_density} below minimum {self.MIN_STEMS_HA}"
        assert 0.0 <= self.conifer_fraction <= 1.0, f"Invalid conifer fraction: {self.conifer_fraction}"
        assert self.biomass_carbon_kg_m2 >= 0.0, f"Negative biomass carbon: {self.biomass_carbon_kg_m2}"
        assert self.soil_carbon_kg_m2 >= 0.0, f"Negative soil carbon: {self.soil_carbon_kg_m2}"

        # --- Initialize Simulator for a New Monte-Carlo Episode ---
        # The simulator is instantiated with a new random seed for weather.
        self.simulator = ebm.ForestSimulator(
            coniferous_fraction=self.conifer_fraction,
            stem_density=self.stem_density,
            weather_seed=self.np_random.integers(0, 2**31 - 1)
        )

        print("--- ForestEnv.reset finished ---")
        return self._get_obs(), self._get_info()

    def step(self, action: np.ndarray):
        print(f"--- ForestEnv.step called (Year: {self.year}) ---")
        
        try:
            # 1. Decode action and update management state
            density_action_idx, mix_action_idx = action
            delta_density = self.DENSITY_ACTIONS[density_action_idx]
            delta_mix = self.MIX_ACTIONS[mix_action_idx]

            # Apply thinning/planting
            old_density = self.stem_density
            new_density_unclipped = self.stem_density + delta_density
            self.stem_density = np.clip(new_density_unclipped, self.MIN_STEMS_HA, self.MAX_STEMS_HA)

            # Track carbon loss from thinning, based on the *actual* change in density
            carbon_loss_thinning = 0
            if self.stem_density < old_density:
                # Thinning occurred, calculate carbon loss proportionally (above-ground biomass only)
                # Use old_density in denominator to avoid division by zero if forest is cleared
                carbon_loss_thinning = self.biomass_carbon_kg_m2 * (old_density - self.stem_density) / old_density

            # Ensure carbon pools never go negative
            self.biomass_carbon_kg_m2 = max(0.0, self.biomass_carbon_kg_m2 - carbon_loss_thinning)

            # Apply species mix change
            self.conifer_fraction = np.clip(self.conifer_fraction + delta_mix, 0.0, 1.0)

            # 2. Run the physical simulation for one year
            # This is the main call to the modified energy_balance_rc module.
            # The simulator is updated with the new density and mix from the action.
            # We also pass the current carbon stock to ensure the simulator is stateless.
            annual_results = self.simulator.run_annual_cycle(
                new_conifer_fraction=self.conifer_fraction,
                new_stem_density=self.stem_density,
                current_biomass_carbon_kg_m2=self.biomass_carbon_kg_m2,
                current_soil_carbon_kg_m2=self.soil_carbon_kg_m2,
            )
            ΔC_biomass = annual_results['delta_biomass_carbon_kg_m2']
            ΔC_soil = annual_results['delta_soil_carbon_kg_m2']
            thaw_dd_year = annual_results['thaw_degree_days']

            # Update carbon pools and ensure they never go negative
            self.biomass_carbon_kg_m2 = max(0.0, self.biomass_carbon_kg_m2 + ΔC_biomass)
            self.soil_carbon_kg_m2 = max(0.0, self.soil_carbon_kg_m2 + ΔC_soil)
            ΔC_year = ΔC_biomass + ΔC_soil
            self.cumulative_thaw_dd += thaw_dd_year

            # 3. Calculate reward
            reward_carbon = self.W_CARBON * ΔC_year
            penalty_thaw = self.W_THAW * thaw_dd_year
            reward = reward_carbon - penalty_thaw

            # 4. Update state and check for termination
            self.year += 1
            
            # Check for early termination due to ecological constraints
            total_carbon = self.biomass_carbon_kg_m2 + self.soil_carbon_kg_m2
            terminated = (
                self.year >= self.EPISODE_LENGTH_YEARS or  # Normal episode end
                total_carbon < 1.0 or  # Carbon stocks too low (ecological failure)
                self.stem_density < self.MIN_STEMS_HA  # Forest density too low
            )
            truncated = False # Not using time limits other than episode end
            
            # Add penalty for early termination due to ecological failure
            if terminated and self.year < self.EPISODE_LENGTH_YEARS:
                reward -= 10.0  # Significant penalty for ecological failure

            print(f"--- ForestEnv.step finished (Year: {self.year}) ---")
            return self._get_obs(), reward, terminated, truncated, self._get_info()
            
        except Exception as e:
            print(f"Error in ForestEnv.step: {e}")
            # Return a safe default state and large negative reward
            return self._get_obs(), -100.0, True, False, self._get_info()

    def _get_obs(self):
        norm_year = self.year / self.EPISODE_LENGTH_YEARS
        norm_density = (self.stem_density - self.MIN_STEMS_HA) / (self.MAX_STEMS_HA - self.MIN_STEMS_HA)
        # Normalize total carbon stock assuming a plausible max value (e.g., 50 kg C/m^2)
        total_carbon = self.biomass_carbon_kg_m2 + self.soil_carbon_kg_m2
        norm_carbon = total_carbon / 50.0

        # Ensure observations are within bounds
        obs = np.array([
            np.clip(norm_year, 0.0, 1.0),
            np.clip(norm_density, 0.0, 1.0),
            np.clip(self.conifer_fraction, 0.0, 1.0),
            np.clip(norm_carbon, 0.0, 1.0)
        ], dtype=np.float32)
        
        return obs

    def _get_info(self):
        total_carbon = self.biomass_carbon_kg_m2 + self.soil_carbon_kg_m2
        return {
            "year": self.year,
            "stem_density_ha": self.stem_density,
            "conifer_fraction": self.conifer_fraction,
            "biomass_carbon_kg_m2": self.biomass_carbon_kg_m2,
            "soil_carbon_kg_m2": self.soil_carbon_kg_m2,
            "carbon_stock_kg_m2": total_carbon,
            "cumulative_thaw_dd": self.cumulative_thaw_dd
        }

    def close(self):
        # Any necessary cleanup
        pass

# --- Training Script ---
def train(total_timesteps=200_000, n_envs=1):
    """
    A helper function to instantiate the environment and train a PPO agent.
    """
    from stable_baselines3.common.env_util import make_vec_env
    
    print("--- train function called ---")
    print("Checking environment...")
    # Check the environment to ensure it follows the Gym API.
    try:
        check_env(ForestEnv())
        print("Environment check passed.")
    except Exception as e:
        print(f"Environment validation failed: {e}")
        raise

    print("--- Creating vectorized environment ---")
    # Vectorized environments for parallel training
    venv = make_vec_env(ForestEnv, n_envs=n_envs, seed=42)
    print("--- Vectorized environment created ---")

    # PPO agent
    # The hyperparameters are chosen as a starting point and may need tuning.
    model = PPO(
        "MlpPolicy",
        venv,
        learning_rate=3e-4,
        n_steps=2048 // n_envs,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        vf_coef=0.5,
        verbose=1,
        tensorboard_log="./ppo_forest_tensorboard/"
    )

    print("--- Starting model.learn ---")
    model.learn(total_timesteps=total_timesteps, progress_bar=True)
    print("--- Finished model.learn ---")
    model.save("ppo_forest_manager")
    venv.close()
    print("Training complete. Model saved to 'ppo_forest_manager.zip'")


import argparse
from tqdm import tqdm


def evaluate_policy(model_path="ppo_forest_manager.zip", n_eval_episodes=1000):
    """
    Evaluate a trained PPO agent on the ForestEnv.

    :param model_path: Path to the saved model zip file.
    :param n_eval_episodes: The number of episodes to run for evaluation.
    """
    print(f"Loading model from {model_path}...")
    model = PPO.load(model_path)

    # Create a single environment for evaluation
    eval_env = ForestEnv()

    print(f"Running evaluation for {n_eval_episodes} episodes...")

    results = {
        "final_carbon_stock_tC_ha": [],
        "cumulative_thaw_dd": [],
    }

    for _ in tqdm(range(n_eval_episodes)):
        obs, info = eval_env.reset()
        terminated = False
        while not terminated:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env.step(action)

        # Episode is done, record final metrics
        # Convert kg/m^2 to tonnes/hectare (1 kg/m^2 = 10 t/ha)
        total_carbon = info['biomass_carbon_kg_m2'] + info['soil_carbon_kg_m2']
        final_carbon_tC_ha = total_carbon * 10
        results["final_carbon_stock_tC_ha"].append(final_carbon_tC_ha)
        results["cumulative_thaw_dd"].append(info['cumulative_thaw_dd'])

    print("\n--- Evaluation Results ---")

    # Calculate median and 95% confidence intervals
    for key, values in results.items():
        median = np.median(values)
        q5 = np.percentile(values, 5)
        q95 = np.percentile(values, 95)
        print(f"Metric: {key}")
        print(f"  Median: {median:.2f}")
        print(f"  95% Interval: [{q5:.2f}, {q95:.2f}]")

    eval_env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train or evaluate a PPO agent for forest management.")
    parser.add_argument("--train", action="store_true", help="Flag to train a new model.")
    parser.add_argument("--evaluate", action="store_true", help="Flag to evaluate a trained model.")
    parser.add_argument("--model_path", type=str, default="ppo_forest_manager.zip", help="Path to the PPO model file.")
    parser.add_argument("--timesteps", type=int, default=200000, help="Number of timesteps for training.")
    parser.add_argument("--eval_episodes", type=int, default=1000, help="Number of episodes for evaluation.")

    args = parser.parse_args()

    if args.train:
        print("--- Starting Training ---")
        train(total_timesteps=args.timesteps)
    elif args.evaluate:
        print("--- Starting Evaluation ---")
        evaluate_policy(model_path=args.model_path, n_eval_episodes=args.eval_episodes)
    else:
        print("Please specify an action: --train or --evaluate")
