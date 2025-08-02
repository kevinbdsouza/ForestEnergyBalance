#!/usr/bin/env python3
"""
ENERGY‑BALANCE MODEL  –  VERSION 4.3_FIXED   (June 2025)
-------------------------------------------------------
Key corrective changes
---------------------
1. **Canopy energy‑balance solved implicitly** (radiation + sensible + latent +
   conduction) → eliminates ±100 K spikes.
2. **Realistic canopy heat capacity**
       - Leaf‑off: 2 × 10⁴ J m⁻² K⁻¹  (woody biomass)
       - Leaf‑on : 8 × 10⁴ J m⁻² K⁻¹  (woody + foliage water)
3. **Aerodynamic conductance floor** (`h_can ≥ 5 W m⁻² K⁻¹`) even when LAI →0.
4. **Snow/soil convection never vanishes** – prevents isolation artefacts.
5. Minor: cleaned constants, docstrings, guard‑rails on Newton solver.

The numerical structure is still *explicit* for all other state variables
(15‑min Δt), but canopy temperature is now obtained implicitly each step, so
larger timesteps are stable.  The rest of the code is kept as close to the
user's V4.2 layout as possible to simplify diff‑checking.
"""

# ----------------------------------------------------------------------------------
# STANDARD LIBS
# ----------------------------------------------------------------------------------
import os
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# ----------------------------------------------------------------------------------
# MODEL CONFIGURATION
# ----------------------------------------------------------------------------------

def get_model_config() -> Dict[str, Any]:
    """Returns a dictionary with all model constants and configurable parameters."""
    return dict(
        # --- Physical constants -------------------------------------------
        SIGMA=5.670374419e-8,  # Stefan–Boltzmann (W m⁻² K⁻⁴)
        RHO_AIR=1.225,         # density of air (kg m⁻³)
        CP_AIR=1005,           # specific heat capacity of air (J kg⁻¹ K⁻¹)
        RHO_WATER=1_000.0,     # density of water (kg m⁻³)
        RHO_SNOW=250.0,        # assumed snow density (kg m⁻³)
        KAPPA=0.41,            # von Kármán constant
        G_ACCEL=9.81,          # gravitational acceleration (m s⁻²)
        PSYCHROMETRIC_GAMMA=0.066, # kPa K⁻¹
        Lv=2.5e6,              # Latent heat of vaporization (J kg⁻¹)
        Lf=3.34e5,             # Latent heat of fusion (J kg⁻¹)
        EPS=np.finfo(float).eps,

        # --- Time discretisation --------------------------------------------
        TIME_STEP_MINUTES=15,

        # --- Heat capacities ( J m⁻² K⁻¹ ) ------------------------------------
        C_CANOPY_LEAF_OFF=2.0e4,
        C_CANOPY_LEAF_ON=8.0e4,
        C_TRUNK=5.0e5,
        C_SNOW=1.0e5,
        C_ATM=1.2e5,
        C_SOIL_TOTAL=6.0e6,

        # --- Misc. stability guards & model params --------------------------
        T_MIN=180.0, T_MAX=330.0,
        DT_CLIP=15.0,
        SWE_SMOOTHING=0.01,
        CANOPY_MIN_H=10.0,
        H_atm=100.0,
        tau_adv=3600.0,

        # --- Aerodynamic parameters -----------------------------------------
        z_ref_h=15.0,     # reference height for h_can (m)
        z0_can=1.5,       # canopy roughness length (m)
        z_ref_soil=2.0,   # reference height for h_soil (m)
        z0_soil=0.01,     # soil roughness length (m)
        h_trunk_const=5.0,    # constant term for h_trunk
        h_trunk_wind_coeff=4.0, # wind coefficient for h_trunk

        # --- Soil & Water parameters ---------------------------------------
        d_soil_surf=0.3, d_soil_deep=1.7,
        k_soil=1.2,              # soil thermal conductivity (W m-1 K-1)
        SWC_max_mm=150.0,
        soil_stress_threshold=0.4,
        T_deep_boundary=270.0,

        # --- Radiation parameters ------------------------------------------
        k_ext_factor=0.5 * 1.2, # Beer's law extinction coefficient factor
        k_snow_factor=0.80,    # Snow interception reduction factor
        eps_can=0.98, eps_snow=0.98, eps_soil=0.95, eps_trunk=0.95,
        eps_atm_max=0.9, eps_atm_min_T=265.0, eps_atm_sensitivity=15.0,
        eps_atm_coeff_a=0.80, eps_atm_coeff_b=0.15,
        alpha_snow=0.80,
        alpha_soil=0.20,
        alpha_trunk=0.25,

        # --- Forcing parameters --------------------------------------------
        latitude_deg=62.0,
        # Temperature forcing
        T_annual_mean_offset=-8.0, # °C
        T_seasonal_amplitude=22.0, # °C
        T_diurnal_amplitude=6.0,   # °C
        T_hour_peak_diurnal=4.0,   # hour of min temp
        # RH forcing
        mean_relative_humidity=0.70,
        # Precip forcing
        rain_summer_mm_day=4.0,
        rain_shoulder_mm_day=2.0,
        winter_snow_mm_day=1.5,
        # Precip seasons (day of year)
        summer_day_start=150, summer_day_end=250,
        shoulder_1_start=90, shoulder_1_end=150,
        shoulder_2_start=250, shoulder_2_end=300,
        snow_season_end=120, snow_season_start=280,
        
        # Phenology (deciduous)
        growth_day=140, fall_day=270,
        growth_rate=0.1, fall_rate=0.1,
        woody_area_index=0.35,

        # --- Trunk parameters ---------------------------------------------
        A_trunk_plan=0.03, A_trunk_vert=0.08,
        k_ct_base_con=0.18, k_ct_base_dec=0.12, # trunk thermal conductivity
        d_ct=0.1, A_c2t=0.08,
        k_tsn=0.05, d_tsn=0.1,
        k_tso=0.8, d_tso=0.1,
        k_snow_pack=0.3,

        # --- Priestley-Taylor coefficient ---------------------------------
        PT_ALPHA=1.26
    )

# ----------------------------------------------------------------------------------
# SIMPLE UTILS
# ----------------------------------------------------------------------------------

def safe_update(T_old: float, dT: float, p: Dict) -> float:
    """Clip ΔT and keep temperature within broad physical bounds."""
    dT = np.clip(dT, -p['DT_CLIP'], p['DT_CLIP'])
    return np.clip(T_old + dT, p['T_MIN'], p['T_MAX'])


def esat_kPa(T: float) -> float:
    """Saturation vapour pressure (kPa) – Tetens eqn."""
    return 0.6108 * np.exp(17.27 * (T - 273.15) / ((T - 273.15) + 237.3))


def delta_svp_kPa_per_K(T: float) -> float:
    es = esat_kPa(T)
    return 4098.0 * es / ((T - 273.15) + 237.3) ** 2


def get_stability_correction(z_ref: float, L: float) -> Tuple[float, float]:
    """
    Calculate Monin-Obukhov stability correction factors, psi_m and psi_h.
    Uses standard formulations from Dyer (1974) and Paulson (1970).
    """
    if abs(L) < 1e-6: # Avoid division by zero
        L = -1e-6 if L < 0 else 1e-6

    zeta = z_ref / L

    if zeta >= 0:  # Stable conditions
        psi_m = -5.0 * zeta
        psi_h = -5.0 * zeta
    else:  # Unstable conditions
        x = (1 - 16 * zeta) ** 0.25
        y = (1 - 16 * zeta) ** 0.5
        psi_m = 2 * np.log((1 + x) / 2) + np.log((1 + x**2) / 2) - 2 * np.arctan(x) + np.pi / 2
        psi_h = 2 * np.log((1 + y) / 2)
        
    return psi_m, psi_h


def h_aero(u: float, z_ref: float, z0: float, L: float, p: Dict) -> float:
    """Bulk sensible-heat conductance (W m⁻² K⁻¹) with dynamic stability correction."""
    u = max(u, 0.1)
    
    _, psi_h = get_stability_correction(z_ref, L)
    
    # The stability correction term psi_h is subtracted
    ra_log_term = np.log(z_ref / z0) - psi_h
    if ra_log_term <= 0: # Guard against non-physical values
        ra_log_term = np.log(z_ref / z0)

    ra = ra_log_term**2 / (p['KAPPA']**2 * u)
    return p['RHO_AIR'] * p['CP_AIR'] / max(ra, 1.0)


# ----------------------------------------------------------------------------------
# PARAMETER INITIALISATION
# ----------------------------------------------------------------------------------

def get_baseline_parameters(config: Dict, coniferous_fraction: float = 0.0) -> Dict[str, Any]:
    """
    Initialise a parameter dictionary `p` for a given species mix,
    drawing from a base model configuration.
    """
    p = config.copy()
    p['coniferous_fraction'] = coniferous_fraction
    deciduous_fraction = 1.0 - coniferous_fraction

    u = 2.0  # reference wind (m s⁻¹)

    # --- Species-specific parameters ---
    con_params = dict(
        alpha_can_base=0.08, k_ct=p['k_ct_base_con'], LAI_max=4.0,
        can_int_frac_rain=0.25, can_int_frac_snow=0.40, A_can_max=0.90
    )
    dec_params = dict(
        alpha_can_base=0.18, k_ct=p['k_ct_base_dec'], LAI_max=5.0,
        can_int_frac_rain=0.20, can_int_frac_snow=0.25, A_can_max=0.90
    )

    # --- Weighted average of parameters ---
    mixed_params = {}
    for key in con_params:
        mixed_params[key] = (con_params[key] * coniferous_fraction) + (dec_params[key] * deciduous_fraction)
    p.update(mixed_params)

    # Store individual LAI_max values for later use in phenology calculation.
    p['LAI_max_coniferous'] = con_params['LAI_max']
    p['LAI_max_deciduous'] = dec_params['LAI_max']

    # Add derived parameters that don't depend on forest type
    p['h_trunk'] = p['h_trunk_const'] + p['h_trunk_wind_coeff'] * u
    p['k_ext'] = p['k_ext_factor']
    p['k_snow'] = p['k_ext'] * p['k_snow_factor']

    # Aerodynamic conductances are now calculated dynamically in the time loop
    # based on stability (L), so they are no longer fixed in `p`.

    p['DT_SECONDS'] = p['TIME_STEP_MINUTES'] * 60
    p['STEPS_PER_DAY'] = int(24 * 60 / p['TIME_STEP_MINUTES'])

    return p


# ----------------------------------------------------------------------------------
# DYNAMIC FORCING & PROP. UPDATES
# ----------------------------------------------------------------------------------

def update_dynamic_parameters(p: Dict, day: int, hour: float, S: dict, L: float):
    """Update meteorological forcing and derived parameters for the current step."""
    # --- Air temperature: annual + diurnal harmonic ------------------------------
    day_angle = 2 * np.pi * (day - 1) / 365.0
    p["T_large_scale"] = 273.15 + p['T_annual_mean_offset'] - p['T_seasonal_amplitude'] * np.cos(day_angle)
    p["T_atm"] = p["T_large_scale"] - p['T_diurnal_amplitude'] * np.cos(2 * np.pi * (hour - p['T_hour_peak_diurnal']) / 24.0)

    # --- Ambient vapour pressure (for VPD calculations) -------------------------
    p['ea'] = p['mean_relative_humidity'] * esat_kPa(p['T_atm'])

    # --- Soil Moisture Stress (depends on a state variable) ---------------------
    swc_frac = S['SWC_mm'] / p['SWC_max_mm']
    stress_range = 1.0 - p['soil_stress_threshold']
    p['soil_stress'] = np.clip((swc_frac - p['soil_stress_threshold']) / stress_range, 0.0, 1.0)

    # --- Precipitation & Snowfall (simple seasonal model) -----------------------
    rain_mm_day = 0.0
    if p['T_atm'] > 274.15: # Only applies if air is warm enough for rain
        if p['summer_day_start'] < day < p['summer_day_end']: # Summer
            rain_mm_day = p['rain_summer_mm_day']
        elif (p['shoulder_1_start'] < day < p['shoulder_1_end']) or (p['shoulder_2_start'] < day < p['shoulder_2_end']): # Shoulder seasons
            rain_mm_day = p['rain_shoulder_mm_day']
    p['rain_m_step'] = rain_mm_day / 1000.0 / p['STEPS_PER_DAY']

    snowfall_m_step = 0.0
    if p['T_atm'] < 272.15 and (day > p['snow_season_start'] or day < p['snow_season_end']):
        snowfall_m_step = p['winter_snow_mm_day'] / 1000 / p['STEPS_PER_DAY']
    p['snowfall_m_step'] = snowfall_m_step

    # --- Short‑wave radiation ----------------------------------------------------
    decl = -23.45 * np.cos(2 * np.pi * (day + 10) / 365.0)  # °
    cos_tz = (
            np.sin(np.deg2rad(p['latitude_deg'])) * np.sin(np.deg2rad(decl)) +
            np.cos(np.deg2rad(p['latitude_deg'])) * np.cos(np.deg2rad(decl)) * np.cos(np.deg2rad(15 * (hour - 12)))
    )
    p["Q_solar"] = max(0.0, 1000.0 * cos_tz)

    # --- Leaf phenology (mixed species) ---------------------------------------
    p["LAI_actual"] = 0.0
    deciduous_fraction = 1.0 - p['coniferous_fraction']

    # Deciduous contribution to LAI
    growth_day, fall_day = p['growth_day'], p['fall_day']
    leaf_on = 1 / (1 + np.exp(-p['growth_rate'] * (day - growth_day)))
    leaf_off = 1 / (1 + np.exp(p['fall_rate'] * (day - fall_day)))
    LAI_deciduous_actual = p["LAI_max_deciduous"] * leaf_on * leaf_off
    
    # Coniferous contribution is constant
    LAI_coniferous_actual = p['LAI_max_coniferous']
    
    # Total LAI is the weighted sum
    p["LAI_actual"] = (LAI_coniferous_actual * p['coniferous_fraction']) + \
                      (LAI_deciduous_actual * deciduous_fraction)
    
    # Total LAI including woody area index (weighted)
    woody_area_coniferous = p['woody_area_index'] # Assuming same for both for now
    woody_area_deciduous = p['woody_area_index']
    total_woody_area = (woody_area_coniferous * p['coniferous_fraction']) + \
                       (woody_area_deciduous * deciduous_fraction)
    p["LAI"] = p["LAI_actual"] + total_woody_area

    # --- Canopy Interception & Associated Evaporation Flux ----------------------
    scaling_num = 1 - np.exp(-p['k_snow'] * p['LAI'])
    scaling_den = 1 - np.exp(-p['k_snow'] * p['LAI_max'])
    interception_scaling = scaling_num / scaling_den if scaling_den > 0 else 0.0

    total_precip_m = p['snowfall_m_step'] + p['rain_m_step']
    # Intercepted rain and snow are now calculated separately
    p['rain_intercepted_m'] = p['rain_m_step'] * p['can_int_frac_rain'] * interception_scaling
    p['snow_intercepted_m'] = p['snowfall_m_step'] * p['can_int_frac_snow'] * interception_scaling

    # Assume intercepted rain evaporates instantly; this becomes an energy sink for the canopy
    p['evap_intercepted_rain_flux'] = (p['rain_intercepted_m'] * p['Lv'] * p['RHO_WATER']) / p['DT_SECONDS']

    # --- Geometric factors (UNIFIED) ---------------------------------------------
    p["K_can"] = np.exp(-p["k_ext"] * p["LAI"])
    p["alpha_can"] = p["alpha_can_base"] if p["LAI_actual"] > 0.1 else p["alpha_trunk"]
    # A_can is the dynamic fractional area used for LW radiation view factors
    p["A_can"] = p["A_can_max"] * (1 - np.exp(-p["k_ext"] * p["LAI"]))

    snow_frac = S['SWE'] / (S['SWE'] + p['SWE_SMOOTHING'])
    p["A_snow"] = (1.0 - p["A_trunk_plan"]) * snow_frac
    p["A_soil"] = 1.0 - p["A_trunk_plan"] - p["A_snow"]

    # --- Derived geometric parameters ---
    # Derive trunk density from max canopy cover for model self-consistency
    p['trunk_density_per_m2'] = p['A_can_max']
    p['trunk_radius_m'] = np.sqrt(p['A_trunk_plan'] / (p['trunk_density_per_m2'] * np.pi))

    # --- Aerodynamic conductances (NOW DYNAMIC) --------------------------------
    # Reference wind speed `u` is still fixed for simplicity
    u = 2.0 
    h_can_raw = h_aero(u, p['z_ref_h'], p['z0_can'], L, p)
    h_soil_raw = h_aero(u, p['z_ref_soil'], p['z0_soil'], L, p)
    p['h_can'] = max(h_can_raw, p['CANOPY_MIN_H'])  # never < floor
    p['h_soil'] = max(h_soil_raw, 3.0) * (1 - snow_frac)
    p['h_snow'] = max(h_soil_raw, 3.0) * snow_frac * 0.5

    # --- Atmospheric emissivity -------------------------------------------------
    p["eps_atm"] = min(
        p['eps_atm_max'],
        p['eps_atm_coeff_a'] + p['eps_atm_coeff_b'] * np.tanh((p["T_atm"] - p['eps_atm_min_T']) / p['eps_atm_sensitivity'])
    )

    return p


# ----------------------------------------------------------------------------------
# CANOPY ENERGY‑BALANCE SOLVER (implicit)
# ----------------------------------------------------------------------------------

def solve_canopy_energy_balance(
        T_guess: float,
        p: Dict[str, Any],
        forcings: Dict[str, float],
        conduct: Dict[str, float],
        SWE_can: float
) -> Tuple[float, Dict[str, float]]:
    """Newton–Raphson root of canopy EB: Rnet – H – LE – Conduction = 0."""

    # Unpack to local vars (faster / clearer inside loop) -----------------------
    eps_can = p['eps_can']
    A_can = p['A_can']
    h_can = conduct['h_can']
    k_ct = p['k_ct']
    A_c2t = p['A_c2t']
    d_ct = p['d_ct']
    PT_ALPHA = p['PT_ALPHA']
    soil_stress = forcings['soil_stress']
    evap_intercepted_rain_flux = forcings['evap_intercepted_rain_flux']

    # constants from forcing struct
    Q_abs_can = forcings['Q_abs_can']
    L_down_atm = forcings['L_down_atm']
    L_up_grnd = forcings['L_up_ground']
    T_trunk = forcings['T_trunk']
    T_air = forcings['T_air']
    ea = forcings['ea']

    # Helper lambdas ------------------------------------------------------------
    def rnet(T):
        lw_balance = eps_can * (L_down_atm + L_up_grnd) - 2 * eps_can * p['SIGMA'] * T ** 4
        return Q_abs_can + A_can * lw_balance

    def latent(T, Rn):
        if T <= 273.15 or Rn <= 0:
            return 0.0
        es = esat_kPa(T)
        vpd = max(0.0, es - ea)
        vpd_stress = np.exp(-0.15 * vpd)
        total_stress = vpd_stress * soil_stress
        Delta = delta_svp_kPa_per_K(T)
        fr_PT = Delta / (Delta + p['PSYCHROMETRIC_GAMMA'])
        return PT_ALPHA * fr_PT * Rn * total_stress

    # --- NEW V2: Check for melt conditions before solving for temperature ---
    melt_energy_sink = 0.0
    if SWE_can > 0:
        T_freeze = 273.15
        Rn_at_freeze = rnet(T_freeze)
        LE_at_freeze = latent(T_freeze, Rn_at_freeze)  # Will be 0
        H_at_freeze = h_can * (T_freeze - T_air)
        Cnd_at_freeze = k_ct * A_c2t / d_ct * (T_freeze - T_trunk)
        F_at_freeze = Rn_at_freeze - H_at_freeze - LE_at_freeze - Cnd_at_freeze - evap_intercepted_rain_flux

        if F_at_freeze > 0:
            # Energy is available for melt. Check if it's enough to melt all snow.
            energy_to_melt_all = (SWE_can * p['Lf'] * p['RHO_WATER']) / p['DT_SECONDS']

            if F_at_freeze < energy_to_melt_all:
                # Partial melt: T_can is clamped at 0°C. All energy goes to melt.
                flux_dict = dict(
                    Rnet_can=Rn_at_freeze, H_can=-H_at_freeze, LE_can=-LE_at_freeze,
                    Cnd_can=-Cnd_at_freeze, Net_can=0.0,
                    Melt_flux_can=F_at_freeze, LE_int_rain=evap_intercepted_rain_flux
                )
                return T_freeze, flux_dict
            else:
                # Full melt: All snow is removed. This becomes a fixed energy sink,
                # and the solver will find the final T_can > 0°C.
                melt_energy_sink = energy_to_melt_all

    # Newton iterations ---------------------------------------------------------
    T = np.clip(T_guess, p['T_MIN'] + 1.0, p['T_MAX'] - 1.0)
    for _ in range(6):
        Rn = rnet(T)
        LE = latent(T, Rn)
        H = h_can * (T - T_air)
        Cnd = k_ct * A_c2t / d_ct * (T - T_trunk)
        F = Rn - H - LE - Cnd - melt_energy_sink - evap_intercepted_rain_flux

        if abs(F) < 1e-3:
            break

        # Numerical derivative dF/dT -------------------------------------------
        dT = 0.1  # K
        Rn_p = rnet(T + dT)
        LE_p = latent(T + dT, Rn_p)
        H_p = h_can * (T + dT - T_air)
        Cnd_p = k_ct * A_c2t / d_ct * (T + dT - T_trunk)
        F_p = Rn_p - H_p - LE_p - Cnd_p - melt_energy_sink - evap_intercepted_rain_flux
        dF = (F_p - F) / dT
        if dF == 0:
            dF = 1e-4
        T -= F / dF
        T = np.clip(T, p['T_MIN'] + 1.0, p['T_MAX'] - 1.0)

    # Re‑compute flux components to return --------------------------------------
    Rn = rnet(T)
    LE = latent(T, Rn)
    H = h_can * (T - T_air)
    Cnd = k_ct * A_c2t / d_ct * (T - T_trunk)

    flux_dict = dict(Rnet_can=Rn, H_can=-H, LE_can=-LE, Cnd_can=-Cnd, Melt_flux_can=melt_energy_sink)
    net_flux = Rn - H - LE - Cnd - melt_energy_sink - evap_intercepted_rain_flux # (≈0 by construction)
    flux_dict['Net_can'] = net_flux
    flux_dict['LE_int_rain'] = evap_intercepted_rain_flux # Diagnostic

    return T, flux_dict


# ----------------------------------------------------------------------------------
# FLUX CALCULATOR (now calls implicit canopy solver)
# ----------------------------------------------------------------------------------

def calculate_fluxes_and_melt(S: Dict, p: Dict) -> Tuple[Dict[str, Dict[str, float]], float, float]:
    """
    Compute all energy flux components for every model node.
    Returns a detailed report of flux components and melt rates.
    """
    # Unpack state and parameters for clarity
    T_can_guess, T_trunk, T_snow = S['canopy'], S['trunk'], S['snow']
    T_soil_surf, T_soil_deep, T_air_model = S['soil_surf'], S['soil_deep'], S['atm_model']
    A_can, A_snow, A_soil = p['A_can'], p['A_snow'], p['A_soil']
    eps_can, eps_soil, eps_snow = p['eps_can'], p['eps_soil'], p['eps_snow']

    # Initialize the detailed flux report dictionary
    flux_report: Dict[str, Dict[str, float]] = {
        node: {} for node in S if 'SWE' not in node and 'SWC' not in node
    }

    # --- 1. Long‑wave radiation terms (independent of T_can) ----------------
    lw = lambda T: p['SIGMA'] * T ** 4
    L_down_atm = p['eps_atm'] * lw(p['T_atm'])
    L_emit_soil = eps_soil * lw(T_soil_surf)
    L_emit_snow = eps_snow * lw(T_snow)
    L_up_ground = (A_soil * L_emit_soil) + (A_snow * L_emit_snow)

    # --- 2. Short-wave radiation partitioning -------------------------------
    Q_solar_on_canopy = p['Q_solar'] * p['A_can_max']
    Q_solar_on_gap = p['Q_solar'] * (1 - p['A_can_max'])
    Q_abs_can = Q_solar_on_canopy * (1 - p['K_can']) * (1 - p['alpha_can'])

    # --- 3. Solve Canopy Energy Balance Implicitly --------------------------
    forcings_for_canopy = dict(
        Q_abs_can=Q_abs_can, L_down_atm=L_down_atm, L_up_ground=L_up_ground,
        T_trunk=T_trunk, T_air=T_air_model, ea=p['ea'], soil_stress=p['soil_stress'],
        evap_intercepted_rain_flux=p['evap_intercepted_rain_flux']
    )
    T_can_step, can_flux = solve_canopy_energy_balance(
        T_guess=T_can_guess, p=p, forcings=forcings_for_canopy,
        conduct=dict(h_can=p['h_can']), SWE_can=S['SWE_can']
    )
    # Note: can_flux components (H_can, LE_can, Cnd_can) are positive INTO canopy
    flux_report['canopy'] = {
        'Rnet': can_flux['Rnet_can'],
        'H': can_flux['H_can'],
        'LE_trans': can_flux['LE_can'],
        'Cnd_trunk': can_flux['Cnd_can'],
        'Melt': -can_flux.get('Melt_flux_can', 0.0),      # as energy sink
        'LE_int_rain': -can_flux.get('LE_int_rain', 0.0) # as energy sink
    }

    # --- 4. Calculate all other flux components using T_can_step ------------
    # Short‑wave on ground
    Q_transmitted = Q_solar_on_canopy * p['K_can']
    Q_ground = Q_solar_on_gap + Q_transmitted
    flux_report['soil_surf']['SW_in'] = A_soil * Q_ground * (1 - p['alpha_soil'])
    flux_report['snow']['SW_in'] = A_snow * Q_ground * (1 - p['alpha_snow'])

    # Initialize conditional fluxes to ensure they are always logged
    flux_report['snow']['Cnd_soil'] = 0.0
    flux_report['snow']['Cnd_trunk'] = 0.0
    flux_report['soil_surf']['Cnd_snow'] = 0.0
    flux_report['soil_surf']['Cnd_trunk'] = 0.0

    # Long‑wave on ground
    L_emit_can = eps_can * lw(T_can_step) if A_can > 0 else 0.0
    gap_fraction = 1.0 - A_can
    L_down_transmitted_atm = (1 - eps_can) * L_down_atm
    LW_in_soil = A_soil * eps_soil * (gap_fraction*L_down_atm + A_can*(L_emit_can + L_down_transmitted_atm))
    LW_in_snow = A_snow * eps_snow * (gap_fraction*L_down_atm + A_can*(L_emit_can + L_down_transmitted_atm))
    flux_report['soil_surf']['LW_net'] = LW_in_soil - A_soil * L_emit_soil
    flux_report['snow']['LW_net'] = LW_in_snow - A_snow * L_emit_snow

    # Soil evaporation
    Rn_soil = flux_report['soil_surf']['SW_in'] + flux_report['soil_surf']['LW_net']
    LE_soil = 0.0
    if Rn_soil > 0 and A_soil > 0:
        Delta_soil = delta_svp_kPa_per_K(T_soil_surf)
        fr_PT_soil = Delta_soil / (Delta_soil + p['PSYCHROMETRIC_GAMMA'])
        LE_soil = p['PT_ALPHA'] * fr_PT_soil * Rn_soil * p['soil_stress']
    flux_report['soil_surf']['LE_evap'] = -LE_soil  # Energy sink

    # Soil conduction (surf-deep-boundary)
    flux_surf_deep = (p['k_soil'] / (0.5*(p['d_soil_surf']+p['d_soil_deep']))) * (T_soil_surf - T_soil_deep)
    flux_deep_bound = (p['k_soil'] / (0.5*p['d_soil_deep'])) * (T_soil_deep - p['T_deep_boundary'])
    flux_report['soil_surf']['Cnd_deep'] = -flux_surf_deep
    flux_report['soil_deep']['Cnd_surf'] = flux_surf_deep
    flux_report['soil_deep']['Cnd_boundary'] = -flux_deep_bound

    # Snow-Soil Conduction
    if p['A_snow'] > 0:
        snow_depth = (S['SWE'] * p['RHO_WATER']) / p['RHO_SNOW']
        R_soil = (0.5 * p['d_soil_surf']) / p['k_soil']
        R_snow = (0.5 * snow_depth) / p['k_snow_pack']
        if (R_soil + R_snow) > 0:
            flux_soil_snow = (1/(R_soil+R_snow)) * p['A_snow'] * (T_soil_surf - T_snow)
            flux_report['soil_surf']['Cnd_snow'] = -flux_soil_snow
            flux_report['snow']['Cnd_soil'] = flux_soil_snow

    # Sensible Heat Fluxes (to/from Atmosphere)
    H_can = flux_report['canopy']['H']
    H_trunk = p['h_trunk'] * (T_trunk - T_air_model)
    H_soil = p['h_soil'] * (T_soil_surf - T_air_model)
    H_snow = p['h_snow'] * (T_snow - T_air_model)
    flux_report['trunk']['H'] = -H_trunk
    flux_report['soil_surf']['H'] = -H_soil
    flux_report['snow']['H'] = -H_snow
    flux_report['atm_model']['H_can'] = H_can
    flux_report['atm_model']['H_trunk'] = H_trunk
    flux_report['atm_model']['H_soil'] = H_soil
    flux_report['atm_model']['H_snow'] = H_snow

    # Upward LW to atmosphere
    LW_ground_to_atm = (gap_fraction*L_up_ground) + (A_can*(1-eps_can)*L_up_ground)
    flux_report['atm_model']['LW_up'] = LW_ground_to_atm
    
    # Atmosphere relaxation
    flux_relax = (p['C_ATM']/p['tau_adv']) * (p['T_large_scale'] - T_air_model)
    flux_report['atm_model']['Relax'] = flux_relax

    # Trunk Conduction (canopy-trunk and trunk-ground)
    Cnd_c_t = -flux_report['canopy']['Cnd_trunk'] # From canopy's perspective
    flux_report['trunk']['Cnd_canopy'] = Cnd_c_t
    if S['SWE'] > 0:
        snow_depth = (S['SWE'] * p['RHO_WATER']) / p['RHO_SNOW']
        lat_area = p['trunk_density_per_m2'] * 2 * np.pi * p['trunk_radius_m'] * snow_depth
        flux_t_snow = (p['k_tsn']/p['d_tsn']) * lat_area * (T_trunk - T_snow)
        flux_report['trunk']['Cnd_ground'] = -flux_t_snow
        flux_report['snow']['Cnd_trunk'] = flux_t_snow
    else:
        flux_t_soil = (p['k_tso']/p['d_tso']) * p['A_trunk_plan'] * (T_trunk - T_soil_surf)
        flux_report['trunk']['Cnd_ground'] = -flux_t_soil
        flux_report['soil_surf']['Cnd_trunk'] = flux_t_soil

    # --- 5. Snow Melt Calculations ------------------------------------------
    d_SWE_melt_grd, d_SWE_melt_can = 0.0, 0.0
    if S['SWE_can'] > 0 and 'Melt' in flux_report['canopy']:
        melt_flux = -flux_report['canopy']['Melt'] # Is positive for melting
        melt_rate = melt_flux / (p['Lf'] * p['RHO_WATER'])
        d_SWE_melt_can = min(melt_rate * p['DT_SECONDS'], S['SWE_can'])

    net_flux_snow = sum(flux_report['snow'].values())
    melt_energy_sink_grd = 0.0
    if S['SWE'] > 0 and T_snow >= 273.15 and net_flux_snow > 0:
        melt_rate = net_flux_snow / (p['Lf'] * p['RHO_WATER'])
        d_SWE_melt_grd = min(melt_rate * p['DT_SECONDS'], S['SWE'])
        melt_energy_sink_grd = net_flux_snow
    
    flux_report['snow']['Melt'] = -melt_energy_sink_grd # Always present now

    # Add T_can_new to the report for the implicit update step
    flux_report['canopy']['T_new'] = T_can_step

    return flux_report, d_SWE_melt_grd, d_SWE_melt_can


# ----------------------------------------------------------------------------------
# MAIN TIME‑INTEGRATION LOOP
# ----------------------------------------------------------------------------------

def run_dynamic_simulation(total_days: int = 1095, spin_up_days: int = 365, coniferous_fraction: float = 0.0, forest_name: str = "Mixed_Forest"):
    config = get_model_config()
    p = get_baseline_parameters(config, coniferous_fraction=coniferous_fraction)

    S = {
        "canopy": 265.0, "trunk": 265.0, "snow": 268.0, "soil_surf": 270.0,
        "soil_deep": 270.0, "atm_model": 265.0, "SWE": 0.0, "SWE_can": 0.0,
        "SWC_mm": p['SWC_max_mm'] * 0.75,
    }
    
    # Initialize Monin-Obukhov Length (L) for near-neutral conditions
    L_stability = 1e6 

    C = {
        "canopy": p['C_CANOPY_LEAF_OFF'], "trunk": p['C_TRUNK'], "snow": p['C_SNOW'],
        "atm_model": p['C_ATM'],
        "soil_surf": p['C_SOIL_TOTAL'] * (p['d_soil_surf'] / (p['d_soil_surf'] + p['d_soil_deep'])),
        "soil_deep": p['C_SOIL_TOTAL'] * (p['d_soil_deep'] / (p['d_soil_surf'] + p['d_soil_deep'])),
    }

    temp_nodes = [n for n in S if 'SWE' not in n and 'SWC' not in n]
    
    # Initialize history with state variables and primary forcings
    history_keys = list(S.keys()) + ['T_atm', 'Q_solar', 'LAI_actual', 'A_snow', 'ea', 'soil_stress', 'runoff_mm', 'L_stability']
    
    # Dynamically add keys for all flux components for all nodes
    p_init = update_dynamic_parameters(p, 1, 0, S, L_stability)
    # The function now returns 3 values; the call site must match.
    sample_flux_report, _, _ = calculate_fluxes_and_melt(S, p_init)
    for node, components in sample_flux_report.items():
        for comp in components:
            # These are helper values, not fluxes to be stored in history
            if comp not in ['T_new', 'LE_soil_demand', 'LE_evap_flux']:
                history_keys.append(f'F_{node}_{comp}')
    
    history = {k: [] for k in history_keys}

    sim_index = pd.date_range(
        start=pd.to_datetime('2023-01-01') - pd.to_timedelta(f'{spin_up_days}D'),
        periods=(total_days + spin_up_days) * p['STEPS_PER_DAY'],
        freq=f"{p['TIME_STEP_MINUTES']}min"
    )

    print(f"Running V4.3_FIXED simulation for {forest_name} …")

    for ts in sim_index:
        day_of_year, hour = ts.dayofyear, ts.hour + ts.minute / 60.0
        p = update_dynamic_parameters(p, day_of_year, hour, S, L_stability)
        C['canopy'] = p['C_CANOPY_LEAF_ON'] if p['LAI'] > 0.1 else p['C_CANOPY_LEAF_OFF']

        # --- Fluxes, Water Balance, and State Updates ---------------------
        flux_report, dSWE_g, dSWE_c = calculate_fluxes_and_melt(S, p)

        # Water balance
        S['SWE_can'] += p['snow_intercepted_m']
        S['SWE'] += p['snowfall_m_step'] - p['snow_intercepted_m']
        
        rain_throughfall = p['rain_m_step'] - p['rain_intercepted_m']
        # Melted canopy snow (dSWE_c) drips to the ground and becomes runoff or infiltrates
        water_in_mm = (rain_throughfall + dSWE_g + dSWE_c) * 1000.0
        S['SWC_mm'] += water_in_mm
        
        LE_transpiration = -flux_report['canopy'].get('LE_trans', 0.0)
        LE_soil_evap = -flux_report['soil_surf'].get('LE_evap', 0.0)
        total_evap_flux = LE_transpiration + LE_soil_evap
        water_out_mm = (total_evap_flux * p['DT_SECONDS']) / (p['Lv'] * p['RHO_WATER']) * 1000.0
        S['SWC_mm'] -= water_out_mm
        
        runoff_mm = max(0.0, S['SWC_mm'] - p['SWC_max_mm'])
        S['SWC_mm'] -= runoff_mm
        S['SWE'] = max(0.0, S['SWE'] - dSWE_g)
        S['SWE_can'] = max(0.0, S['SWE_can'] - dSWE_c)

        # Temperature updates
        S['canopy'] = flux_report['canopy']['T_new']
        for node in temp_nodes:
            if node == 'canopy': continue
            net_flux = sum(flux_report[node].values())
            dT = (net_flux / C[node]) * p['DT_SECONDS'] if C[node] > 0 else 0.0
            S[node] = safe_update(S[node], dT, p)
            
        # --- Update Monin-Obukhov Length for NEXT timestep ---
        # Sum only the sensible heat components for the stability calculation
        H_total = (flux_report['atm_model']['H_can'] +
                   flux_report['atm_model']['H_trunk'] +
                   flux_report['atm_model']['H_soil'] +
                   flux_report['atm_model']['H_snow'])
        
        if abs(H_total) > 1e-3: # Avoid near-zero flux
            # 1. Calculate friction velocity (u_star)
            u = 2.0 # Still using fixed reference wind
            psi_m, _ = get_stability_correction(p['z_ref_h'], L_stability)
            u_star_log_term = np.log(p['z_ref_h'] / p['z0_can']) - psi_m
            if u_star_log_term <= 0:
                 u_star_log_term = np.log(p['z_ref_h'] / p['z0_can'])
            u_star = u * p['KAPPA'] / u_star_log_term

            # 2. Calculate L
            T_air_kelvin = p['T_atm'] # Virtual temp approximation
            L_num = -p['RHO_AIR'] * p['CP_AIR'] * (u_star**3) * T_air_kelvin
            L_den = p['KAPPA'] * p['G_ACCEL'] * H_total
            if abs(L_den) > 1e-9:
                L_stability = L_num / L_den
            else:
                L_stability = 1e6 # Revert to neutral
        else:
            L_stability = 1e6 # Neutral conditions if no sensible heat flux

        # --- Bookkeeping ---------------------------------------------------
        for key in S: history[key].append(S[key])
        history['L_stability'].append(L_stability)
        
        # Store parameters from the `p` dictionary
        for key in ['T_atm', 'Q_solar', 'LAI_actual', 'A_snow', 'ea', 'soil_stress']:
             history[key].append(p[key])
        # Store the locally calculated runoff
        history['runoff_mm'].append(runoff_mm)

        # Store all detailed flux components
        for node, components in flux_report.items():
            for comp, value in components.items():
                key = f'F_{node}_{comp}'
                if key in history:
                    history[key].append(value)

    print("Simulation complete.")
    df = pd.DataFrame(history, index=sim_index)
    return df.loc[df.index.year >= 2023]


# ----------------------------------------------------------------------------------
# PLOTTING & SAVE HELPERS (unchanged except minor label tweak) ----------------------
# ----------------------------------------------------------------------------------

def plot_and_save_annual_results(df: pd.DataFrame, forest_title: str, tag="V4_3_FIXED"):
    """
    Generate a multi-panel plot of the key annual cycles for temperature,
    liquid water, and frozen water.
    """
    save_dir = Path("plots")
    save_dir.mkdir(parents=True, exist_ok=True)

    dfC = df.copy()
    temp_cols = [c for c in dfC.columns if 'T_' in c or c in ['canopy', 'soil_surf', 'soil_deep', 'trunk', 'snow']]
    for col in temp_cols:
        dfC[col] -= 273.15
    daily = dfC.resample('D').mean()

    fig, axs = plt.subplots(3, 1, figsize=(18, 12), sharex=True)
    fig.suptitle(f"Annual Cycle — {forest_title.replace('_', ' ')} ({tag})", fontsize=16)

    # --- Panel 1: Temperatures ----------------------------------------------
    axs[0].plot(daily.index, daily['canopy'], label='Canopy T', color='darkgreen', lw=2)
    axs[0].plot(daily.index, daily['soil_surf'], label='Surface Soil T', color='saddlebrown', lw=2)
    axs[0].plot(daily.index, daily['T_atm'], label='Air T', color='black', ls=':', alpha=0.8)
    axs[0].set_ylabel('Temperature [°C]')
    axs[0].grid(alpha=0.4, ls='--')
    axs[0].legend(loc='upper left')
    axs[0].set_title('Temperatures')

    # --- Panel 2: Liquid Water & LAI ----------------------------------------
    axs[1].fill_between(daily.index, daily['SWC_mm'], color='dodgerblue', alpha=0.3, label='Soil Water Content')
    axs[1].set_ylabel('Soil Water [mm]', color='dodgerblue')
    axs[1].tick_params(axis='y', labelcolor='dodgerblue')
    axs[1].legend(loc='upper left')
    
    ax2_twin = axs[1].twinx()
    ax2_twin.fill_between(daily.index, daily['LAI_actual'], color='green', alpha=0.2, label='LAI (actual)')
    ax2_twin.set_ylabel('LAI', color='green')
    ax2_twin.tick_params(axis='y', labelcolor='green')
    ax2_twin.legend(loc='upper right')
    axs[1].set_title('Soil Moisture and Phenology')
    
    # --- Panel 3: Snow Water Equivalent (SWE) -------------------------------
    axs[2].fill_between(daily.index, daily['SWE'] * 1000, color='deepskyblue', alpha=0.5, label='Ground SWE')
    axs[2].fill_between(daily.index, daily['SWE_can'] * 1000, color='lightseagreen', alpha=0.7, label='Canopy SWE')
    axs[2].set_ylabel('SWE [mm]')
    axs[2].grid(alpha=0.4, ls='--')
    axs[2].legend(loc='upper left')
    axs[2].set_title('Snowpack')

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = save_dir / f"annual_cycle_{forest_title}_{tag}.png"
    plt.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# EXTRA DIAGNOSTIC PLOTS  ▸  paste the whole block here
# ─────────────────────────────────────────────────────────────────────────────
def plot_weekly_ribbons(df, tag, save_dir=Path("plots")):
    """
    Weekly-mean temperature with a 5–95 % ribbon.
    Works for any DataFrame that has Kelvin temperatures and a DateTimeIndex.
    """
    wk_mean = df.resample('W').mean()
    wk_q05  = df.resample('W').quantile(0.05)
    wk_q95  = df.resample('W').quantile(0.95)

    fig, ax = plt.subplots(figsize=(14,6))
    for col, color in zip(['T_atm', 'canopy', 'soil_surf'],
                          ['black',   'green',  'sienna']):
        m  = wk_mean[col] - 273.15
        q0 = wk_q05 [col] - 273.15
        q9 = wk_q95 [col] - 273.15
        ax.plot(m.index,  m,  color=color, lw=1.8, label=col.replace('_',' '))
        ax.fill_between(m.index, q0, q9, color=color, alpha=.15)
    ax.set_ylabel('Temperature [°C]')
    ax.set_title(f'Weekly Mean & 90 % Range — {tag}')
    ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(save_dir / f'weekly_ribbon_{tag}.png', dpi=150)
    plt.close(fig)


def plot_seasonal_diurnal(df, tag, save_dir=Path("plots")):
    seasons = {'Winter':[12,1,2], 'Spring':[3,4,5],
               'Summer':[6,7,8],  'Autumn':[9,10,11]}
    fig, axs = plt.subplots(2,2, figsize=(12,8), sharex=True, sharey=True)

    for ax, (season, months) in zip(axs.flat, seasons.items()):
        sub = df[df.index.month.isin(months)]            # ① filter first
        g   = sub.groupby(sub.index.hour).mean()         # ② then group by hour
        ax.plot(g.index, g['canopy']-273.15, color='green', label='Canopy')
        ax.plot(g.index, g['T_atm'] -273.15, ls=':', color='black', label='Air')
        ax.set_title(season); ax.grid(alpha=.3); ax.set_xlabel('Hour')

    axs[0,0].set_ylabel('°C'); axs[1,0].set_ylabel('°C')
    axs[0,1].legend(loc='upper left')
    fig.suptitle(f'Diurnal Cycle by Season — {tag}'); fig.tight_layout()
    fig.savefig(save_dir / f'seasonal_diurnal_{tag}.png', dpi=150)
    plt.close(fig)


def plot_flux_budgets(df: pd.DataFrame, forest_title: str, tag="V4_3_FIXED"):
    """
    Plots the daily mean energy budget components for major model nodes.
    """
    save_dir = Path("plots")
    daily = df.resample('D').mean()
    # Use a color cycle to make plots clearer
    color_cycle = plt.cm.viridis(np.linspace(0, 0.9, 10)) # Increased for more fluxes

    nodes_to_plot = {
        'canopy': [k for k in df.columns if k.startswith('F_canopy_')],
        'snow': [k for k in df.columns if k.startswith('F_snow_')],
        'soil_surf': [k for k in df.columns if k.startswith('F_soil_surf_')],
        'trunk': [k for k in df.columns if k.startswith('F_trunk_')],
        'atm_model': [k for k in df.columns if k.startswith('F_atm_model_')],
        'soil_deep': [k for k in df.columns if k.startswith('F_soil_deep_')]
    }

    for node, f_cols in nodes_to_plot.items():
        if not f_cols: continue

        flux_df = daily[f_cols].copy()
        flux_df.columns = [c.replace(f'F_{node}_', '') for c in f_cols]
        flux_df['Net Flux'] = flux_df.sum(axis=1)

        fig, ax = plt.subplots(figsize=(18, 7))
        
        # Plot individual components with a color cycle
        sorted_cols = sorted([c for c in flux_df.columns if c != 'Net Flux'])
        for i, col in enumerate(sorted_cols):
            # Use modulo operator for color cycle to avoid index out of bounds
            ax.plot(flux_df.index, flux_df[col], label=col, lw=1.5, alpha=0.8, color=color_cycle[i % len(color_cycle)])
        
        # Plot the net flux
        ax.plot(flux_df.index, flux_df['Net Flux'], label='Net Flux', color='black', lw=2.5, ls='--')

        ax.grid(alpha=0.4, ls='--')
        ax.legend(loc='best', ncol=4) # Increased columns for legend
        ax.set_ylabel('Flux (W m⁻²)')
        ax.set_title(f"Energy Flux Budget for {node.upper()} — {forest_title.replace('_', ' ')} ({tag})")
        ax.axhline(0, color='black', lw=0.75, ls=':')

        fig.tight_layout()
        out = save_dir / f"flux_budget_{node}_{forest_title}_{tag}.png"
        plt.savefig(out, dpi=150)
        plt.close(fig)
        print(f"Saved: {out}")


def plot_flux_partition(df, tag, save_dir=Path("plots")):
    fig, ax = plt.subplots(figsize=(6,6))
    ax.scatter(df['F_canopy_Rnet'], -df['F_canopy_H'], s=4, alpha=.4, label='H')
    ax.scatter(df['F_canopy_Rnet'], -df['F_canopy_LE_trans'], s=4, alpha=.4, label='LE',
               c='orange')
    ax.plot([0,450],[0,450],'k--',lw=.8)
    ax.set_xlim(0,450); ax.set_ylim(0,450)
    ax.set_xlabel('Rnet (W m⁻²)'); ax.set_ylabel('Flux (W m⁻²)')
    ax.set_title(f'Canopy Flux Partition — {tag}'); ax.legend()
    fig.tight_layout(); fig.savefig(save_dir/f'flux_partition_{tag}.png', dpi=150)
    plt.close(fig)


def plot_flux_annual_cycle(df: pd.DataFrame, forest_title: str, tag="V4_3_FIXED"):
    """
    Plots the daily mean of major energy fluxes over the year, dynamically
    selecting the top 75% of contributors to the total flux magnitude.
    """
    save_dir = Path("plots")
    daily = df.resample('D').mean().copy()

    # --- Dynamic Flux Selection ---
    # 1. Identify all flux columns
    flux_cols = [c for c in daily.columns if c.startswith('F_')]
    
    # 2. Calculate the integrated absolute magnitude for each flux
    flux_magnitudes = daily[flux_cols].abs().sum().sort_values(ascending=False)
    
    # 3. Determine the top 75% contributors
    total_magnitude = flux_magnitudes.sum()
    cumulative_magnitude = 0.0
    flux_cols_to_plot = {}  # Use dict for label: col_name

    for col, magnitude in flux_magnitudes.items():
        if cumulative_magnitude < total_magnitude * 0.75 or len(flux_cols_to_plot) < 3: # Ensure at least 3 fluxes are plotted
            cumulative_magnitude += magnitude
            # Clean up the label for the legend
            label = col.replace('F_', '').replace('_', ' ').title()
            flux_cols_to_plot[label] = col
        else:
            break  # Stop once we've hit 75%

    fig, ax = plt.subplots(figsize=(18, 8))
    
    # Use a color cycle for better visualization
    color_cycle = plt.cm.turbo(np.linspace(0, 0.9, len(flux_cols_to_plot)))

    for i, (label, col) in enumerate(flux_cols_to_plot.items()):
        # Plotting fluxes with their natural sign is physically clearer.
        ax.plot(daily.index, daily[col], label=label, lw=1.5, alpha=0.8, color=color_cycle[i])

    ax.grid(alpha=0.4, ls='--')
    ax.legend(loc='best', ncol=max(2, len(flux_cols_to_plot) // 4))
    ax.set_ylabel('Flux (W m⁻²)')
    ax.set_title(f"Annual Cycle of Major Energy Fluxes (Top 75% Contributors) — {forest_title.replace('_', ' ')} ({tag})")
    ax.axhline(0, color='black', lw=0.5, ls='--')

    fig.tight_layout()
    out = save_dir / f"annual_flux_cycle_{forest_title}_{tag}.png"
    plt.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")


# ----------------------------------------------------------------------------------
# DRIVER ---------------------------------------------------------------------------
# ----------------------------------------------------------------------------------
if __name__ == "__main__":
    # --- Define Scenarios ---
    scenarios = {
        "Deciduous_Forest": 0.0,
        "Coniferous_Forest": 1.0,
        "Mixed_Forest_50_50": 0.5
    }

    for name, fraction in scenarios.items():
        print(f"--- Running scenario: {name} (Coniferous Fraction: {fraction}) ---")
        data = run_dynamic_simulation(
            coniferous_fraction=fraction,
            forest_name=name
        )
        plot_and_save_annual_results(data, name)

        # ✦ NEW diagnostic plots ✦
        plot_flux_budgets(data, name)
        plot_weekly_ribbons(data, name)
        plot_seasonal_diurnal(data, name)
        plot_flux_partition(data, name)
        plot_flux_annual_cycle(data, name)
