import numpy as np
from energy_balance_rc import ForestSimulator

def test_reproducible_episode():
    seed = 12345
    params = dict(
        coniferous_fraction=0.6,
        stem_density=1200,
        carbon_stock_kg_m2=20.0,
        weather_seed=seed,
    )
    sim1 = ForestSimulator(**params)
    sim2 = ForestSimulator(**params)

    res1 = sim1.run_annual_cycle(
        new_conifer_fraction=params["coniferous_fraction"],
        new_stem_density=params["stem_density"],
        current_biomass_carbon_kg_m2=params["carbon_stock_kg_m2"] * 0.7,  # Assume 70% biomass, 30% soil
        current_soil_carbon_kg_m2=params["carbon_stock_kg_m2"] * 0.3,
    )
    res2 = sim2.run_annual_cycle(
        new_conifer_fraction=params["coniferous_fraction"],
        new_stem_density=params["stem_density"],
        current_biomass_carbon_kg_m2=params["carbon_stock_kg_m2"] * 0.7,  # Assume 70% biomass, 30% soil
        current_soil_carbon_kg_m2=params["carbon_stock_kg_m2"] * 0.3,
    )

    for key in res1:
        assert np.isclose(res1[key], res2[key])
