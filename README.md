# Forest Energy Balance Model

This repository contains `energy_balance_rc.py`, a sophisticated single-point energy and water balance model for a forest ecosystem. The model simulates the interactions between the atmosphere, a two-layer canopy (coniferous and deciduous), trunk space, a multi-layer soil column, and a snowpack.

## Model Description

The model is designed to simulate the diurnal and seasonal cycles of temperature and water fluxes within a forest environment. It is built with a 15-minute time step and can be configured to represent different forest types by adjusting the fraction of coniferous vs. deciduous trees.

### Key Components

The model is structured around several key physical components:

*   **Atmosphere:** A single atmospheric layer that is forced by large-scale temperature and humidity, and which interacts with the surface via radiation, sensible heat, and latent heat fluxes.
*   **Canopy:** A representation of the forest canopy, which can be a mix of coniferous and deciduous species. It intercepts radiation, precipitation, and exchanges heat and moisture with the atmosphere.
*   **Trunk Space:** The layer between the canopy and the ground, containing the tree trunks.
*   **Snowpack:** A dynamic snow layer on the ground that accumulates snowfall, melts, and affects the surface energy balance.
*   **Soil:** A two-layer soil model that simulates temperature and soil moisture dynamics.

### Simulated Processes

The model simulates a range of physical and ecological processes:

*   **Energy Balance:** The model solves the energy balance for each component, including radiative (shortwave and longwave), sensible, latent, and conductive heat fluxes. The canopy energy balance is solved implicitly for numerical stability.
*   **Water Balance:** The model tracks the flow of water through the system, including precipitation (rain and snow), canopy interception, evaporation, transpiration, infiltration, and runoff.
*   **Phenology:** A simple phenology model controls the seasonal growth and senescence of leaves for deciduous trees, dynamically updating the Leaf Area Index (LAI).
*   **Photosynthesis:** The model now includes a calculation for the energy sink due to photosynthesis, based on a light-use efficiency (LUE) approach. This flux is coupled with environmental stress factors (VPD, soil moisture) similarly to transpiration.

### Key Features

This model version (V4.3) includes several important features that enhance its realism and stability:

1.  **Implicit Canopy Energy Balance:** The canopy temperature is solved implicitly, which prevents numerical instability and allows for larger, more stable time steps.
2.  **Realistic Heat Capacities:** The model uses realistic, leaf-area dependent heat capacities for the canopy.
3.  **Aerodynamic Conductance:** Bulk aerodynamic conductances are calculated with a Monin-Obukhov stability correction.
4.  **Dynamic Properties:** Many parameters, such as albedo, emissivity, and interception fractions, are updated dynamically based on the state of the system (e.g., LAI, snow cover).

## How to Run

The script can be run directly from the command line:

```bash
python energy_balance_rc.py
```

This will execute a series of simulations for different forest types (deciduous, coniferous, and mixed) and generate a set of diagnostic plots in the `plots/` directory.
