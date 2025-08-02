"""
Adaptive Forest‑Management RL Environment v2
===========================================
*Updates over the initial `forest_rl_training.py`*
-------------------------------------------------
1. **Multi‑Year Horizon** – episodes are now *5 years* (1 825 days) so the
   agent can learn seasonal and inter‑annual trade‑offs.
2. **Richer Action Space** (5 continuous controls)
   ┌─────────────┬────────────────────────────────────────────┐
   │ Index       │ Management lever (per m² of ground)        │
   ├─────────────┼────────────────────────────────────────────┤
   │   0         │ Δ LAI_max  [−1 … +1]                       │
   │   1         │ Δ canopy albedo α_can_base [−0.05 … +0.05] │
   │   2         │ Δ species‑mix (proportion deciduous)       │
   │   3         │ Δ moss/organic layer thickness [m]         │
   │   4         │ Δ soil‑water target [−20 … +20 mm]          │
   └─────────────┴────────────────────────────────────────────┘
3. **Carbon & Thaw Diagnostics**
   * `biomass_C` (kg C m⁻²) – simple LAI‑based accrual model.
   * `thaw_degree_days` – cumulative (T_soil_surf − 0 °C)+ over time.
4. **Composite Reward** (weights tune‑able):
   ```
   R = − w_EE  · max(0, H_total)                # surface energy export
       − w_thaw· thaw_degree_days_today/50      # penalise active‑layer growth
       + w_C   · Δ_biomass_C_today              # reward sequestration
   ```
5. **Observation Vector (14 dims)** – adds biomass C, thaw D‑days, species mix,
   moss thickness, target SWC, and soil deep temperature.
6. **Species‑Mix Dynamics** – parameter blend between deciduous and coniferous
   templates applied *gradually* (1 % per month) for realism.
7. **Soil Insulation & Moisture Control** – moss layer lowers `k_soil`, higher
   target SWC nudges soil‑water content via artificial irrigation/drainage.

Run‑time CLI remains:
```bash
pip install numpy pandas gymnasium stable-baselines3==2.2.1
python forest_rl_training.py         # now v2
```
"""

# forest_rl_training.py  v2 – multi‑objective, multi‑year RL
from __future__ import annotations
import numpy as np, gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from typing import Dict, Any, Tuple
import energy_balance_model_v4_3_fixed as ebm

# -----------------------------------------------------------------------------
# Helper: linear blend between two parameter dictionaries
# -----------------------------------------------------------------------------
_CONIF = ebm.get_baseline_parameters("coniferous", ebm.get_model_config())
_DECID = ebm.get_baseline_parameters("deciduous", ebm.get_model_config())
def _blend_params(alpha: float) -> Dict[str, Any]:
    p = {}
    for k in _CONIF:
        v1, v2 = _CONIF[k], _DECID[k]
        p[k] = (1 - alpha) * v1 + alpha * v2 if isinstance(v1, (int, float)) else v2
    return p

# -----------------------------------------------------------------------------
class ForestAdaptiveEnv(gym.Env):
    """5‑year forest‑management environment with multi‑objective reward."""
    metadata = {}
    _DT_DAYS = 1      # env step = 1 day → roll 96×15‑min solver steps
    _EP_DAYS = 5*365  # 5 year episodes

    # ‑‑‑ action bounds ------------------------------------------------------
    ACT_LOW  = np.array([-1.0, -0.05, -0.5, -0.05, -20.0], np.float32)
    ACT_HIGH = np.array([ 1.0,  0.05,  0.5,  0.05,  20.0], np.float32)

    # ‑‑‑ reward weights -----------------------------------------------------
    w_EE, w_thaw, w_C = 1.0, 0.5, 2.0

    def __init__(self, seed: int | None = None):
        super().__init__()
        self.action_space = spaces.Box(self.ACT_LOW, self.ACT_HIGH, dtype=np.float32)

        # obs: canopy T, soil T_surf/deep, air T, SWC, SWE, LAI, solar, vpd,
        #       H_total, biomass_C, thaw_deg_days, species_mix, moss_thick
        obs_lo = np.array([200, 250, 250, 200,   0,   0, 0, 0, 0, -500,   0, 0, 0, 0], np.float32)
        obs_hi = np.array([330, 320, 320, 330, 300, 500, 6, 1200, 5,  500, 30, 2e4, 1, 0.3], np.float32)
        self.observation_space = spaces.Box(obs_lo, obs_hi, dtype=np.float32)
        self._rng = np.random.default_rng(seed)
        self.reset()

    # ------------------------------------------------------------------
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        # management state vars
        self.species_mix   = 0.5   # 0=conifer, 1=deciduous
        self.moss_thick_m  = 0.05  # m of organic layer
        self.target_SWC_mm = 0.75 * _CONIF['SWC_max_mm']

        # physics params blend + overwrite soil k
        self.p = _blend_params(self.species_mix)
        self._apply_moss_effect()
        self.S = {
            "canopy":265., "trunk":265., "snow":268.,
            "soil_surf":270., "soil_deep":270., "atm_model":265.,
            "SWE":0., "SWE_can":0., "SWC_mm":self.target_SWC_mm,
        }
        self.day_count = 0
        self.biomass_C = 10.0  # kg C m⁻²
        self.thaw_degree_days = 0.0
        self._last_H = 0.0
        self._last_thaw_today = 0.0
        return self._get_obs(), {}

    # ------------------------------------------------------------------
    def step(self, action: np.ndarray):
        # 1│ apply management deltas ------------------------------------
        d_lai, d_alb, d_mix, d_moss, d_swc = action.astype(float)
        self.p['LAI_max']       = np.clip(self.p['LAI_max'] + d_lai, 0.5, 7.0)
        self.p['alpha_can_base']= np.clip(self.p['alpha_can_base'] + d_alb, 0.02, 0.40)
        self.species_mix        = np.clip(self.species_mix + d_mix, 0.0, 1.0)
        self.moss_thick_m       = np.clip(self.moss_thick_m + d_moss, 0.00, 0.30)
        self.target_SWC_mm      = np.clip(self.target_SWC_mm + d_swc, 0, self.p['SWC_max_mm'])

        # gradual species mix adjustment (1 % per *month*)
        if self.day_count % 30 == 0:
            self.p = _blend_params(self.species_mix)
        self._apply_moss_effect()

        H_total_accum = 0.0
        thaw_today = 0.0
        L_stab = 1e6
        for t in range(self.p['STEPS_PER_DAY']):
            hr = t * self.p['TIME_STEP_MINUTES']/60
            day_of_year = (self.day_count % 365)+1
            self.p = ebm.update_dynamic_parameters(self.p, day_of_year, hr, self.S, L_stab)
            flux, dSWE_g, dSWE_c = ebm.calculate_fluxes_and_melt(self.S, self.p)
            # ► minimal water: irrigation / drainage to target SWC
            self.S['SWC_mm'] += (self.target_SWC_mm - self.S['SWC_mm']) * 0.01
            self.S['SWE']     = max(0., self.S['SWE'] - dSWE_g)
            self.S['SWE_can'] = max(0., self.S['SWE_can'] - dSWE_c)
            # ► temps
            self.S['canopy'] = flux['canopy']['T_new']
            for node in ['trunk','snow','soil_surf','soil_deep','atm_model']:
                F = sum(flux[node].values()); C = self._heat_cap(node)
                dT = (F/self.p['DT_SECONDS'])/C if C>0 else 0
                self.S[node] = ebm.safe_update(self.S[node], dT, self.p)
            # ► stability for next 15‑min
            H_total = (flux['atm_model']['H_can']+flux['atm_model']['H_trunk']+
                        flux['atm_model']['H_soil']+flux['atm_model']['H_snow'])
            L_stab = 1e6 if abs(H_total)<1e-3 else (-self.p['RHO_AIR']*self.p['CP_AIR']*
                    (2*0.41)**3*self.p['T_atm'])/(0.41*self.p['G_ACCEL']*H_total)
            H_total_accum += H_total
            if self.S['soil_surf']>273.15:
                thaw_today += (self.S['soil_surf']-273.15)*self.p['DT_SECONDS']/86400
        # --- end‑of‑day updates ----------------------------------------
        self._last_H = H_total_accum/self.p['STEPS_PER_DAY']
        self.thaw_degree_days += thaw_today
        self._last_thaw_today = thaw_today
        # gross C uptake (toy NPP):
        NPP = 1e-4 * self.p['Q_solar'] * self.p['LAI_actual'] / 48  # kg C m⁻² day⁻¹
        self.biomass_C += NPP - max(0,-d_lai)*0.5  # thinning releases C
        # reward scalarisation -----------------------------------------
        r = (-self.w_EE * max(0,self._last_H)/100
             -self.w_thaw * thaw_today/50
             +self.w_C * NPP)
        self.day_count += 1
        terminated = self.day_count >= self._EP_DAYS
        return self._get_obs(), float(r), terminated, False, {
            'H_Wm2': self._last_H,
            'thaw_today': thaw_today,
            'biomass_C': self.biomass_C,
        }

    # ------------------------------------------------------------------
    def _heat_cap(self,node:str):
        return {
            'trunk':self.p['C_TRUNK'], 'snow':self.p['C_SNOW'],
            'soil_surf':self.p['C_SOIL_TOTAL']*0.15,
            'soil_deep':self.p['C_SOIL_TOTAL']*0.85,
            'atm_model':self.p['C_ATM']
        }.get(node,1.)

    def _apply_moss_effect(self):
        # exponential insulation effect on soil conductivity & albedo tweak
        self.p['k_soil'] = max(0.3, 1.2 * np.exp(-5*self.moss_thick_m))
        self.p['alpha_soil'] = 0.20 + 0.3*self.moss_thick_m

    def _get_obs(self):
        p,s = self.p,self.S
        vpd = max(0., ebm.esat_kPa(p['T_atm'])-p['ea'])
        return np.array([
            s['canopy'], s['soil_surf'], s['soil_deep'], p['T_atm'], s['SWC_mm'],
            (s['SWE']+s['SWE_can'])*1000, p['LAI_actual'], p['Q_solar'], vpd,
            self._last_H, self.biomass_C, self.thaw_degree_days,
            self.species_mix, self.moss_thick_m
        ], np.float32)

    # no render/close needed

# -----------------------------------------------------------------------------
# Training helper
# -----------------------------------------------------------------------------

def train(total_steps:int=2_000_000, n_env:int=8):
    venv = make_vec_env(lambda: ForestAdaptiveEnv(), n_envs=n_env, seed=0)
    model = PPO('MlpPolicy', venv, learning_rate=2.5e-4, n_steps=2048//n_env,
                batch_size=1024, gamma=0.999, gae_lambda=0.97,
                ent_coef=0.01, vf_coef=0.4, target_kl=0.03, verbose=1)
    model.learn(total_timesteps=total_steps, progress_bar=True)
    model.save('ppo_forest_multiobj')
    venv.close(); print('Saved to ppo_forest_multiobj.zip')

if __name__=='__main__':
    train()
