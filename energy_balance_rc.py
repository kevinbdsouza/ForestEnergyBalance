#!/usr/bin/env python3
"""
ENERGY-BALANCE MODEL  –  VERSION 5.0 (RL-Integrated)
-----------------------------------------------------
This version refactors the original energy balance model into a class-based
simulator (`ForestSimulator`) designed to be integrated into a reinforcement
learning loop.

Key changes:
1.  **Class-based Structure**: The entire simulation state and logic are
    encapsulated in the `ForestSimulator` class, allowing multiple independent
    instances to be created (e.g., for vectorized RL environments).
2.  **Stochastic Weather Generation**: The `update_dynamic_parameters` function
    now accepts a random number generator to introduce daily variability into
    temperature and precipitation, making the simulation non-deterministic.
    This is crucial for training robust RL policies.
3.  **RL-centric Annual Loop**: The `run_annual_cycle` method runs the
    simulation for a full year and returns key performance indicators needed for
    the RL reward function:
    *   `delta_carbon_kg_m2`: Net change in ecosystem carbon stock.
    *   `thaw_degree_days`: Cumulative soil thaw metric.
4.  **Carbon Accounting**: A simple carbon cycle model has been added.
    *   Gross Primary Production (GPP) is calculated from the model's
        photosynthesis flux (`G_photo`).
    *   Ecosystem Respiration (`R_eco`) is modeled using a Q10 temperature
        response function.
    *   Net Ecosystem Exchange (NEE) is `GPP - R_eco`.
5.  **Management Levers**: The simulator now accepts `stem_density` and
    `conifer_fraction` as inputs, which directly influence the physical
    parameters of the forest model (e.g., `A_can_max`, `LAI_max`).
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
# ALL FUNCTIONS (UNCHANGED)
# Most functions from the original script are kept here, outside the class.
# They are stateless and operate on the `p` and `S` dictionaries.
# ----------------------------------------------------------------------------------

def get_model_config() -> Dict[str, Any]:
    """
    Returns a dictionary with all model constants and configurable parameters.
    For use in Monte Carlo simulations, some key parameters are defined as
    ranges (min, max) rather than fixed values.
    NOTE: Ranges are estimates based on general ecological knowledge for
    Boreal Canada, as direct literature access was unavailable.
    """
    return dict(
        # --- Physical constants (Fixed) -----------------------------------
        SIGMA=5.670374419e-8, RHO_AIR=1.225, CP_AIR=1005, RHO_WATER=1_000.0,
        RHO_SNOW=250.0, KAPPA=0.41, G_ACCEL=9.81, PSYCHROMETRIC_GAMMA=0.066,
        Lv=2.5e6, Lf=3.34e5, EPS=np.finfo(float).eps,
        # --- Time discretisation (Fixed) ----------------------------------
        TIME_STEP_MINUTES=15,
        # --- Heat capacities (Fixed) --------------------------------------
        C_CANOPY_LEAF_OFF=2.0e4, C_CANOPY_LEAF_ON=8.0e4, C_TRUNK=5.0e5,
        C_SNOW=1.0e5, C_ATM=1.2e5, C_SOIL_TOTAL=6.0e6,
        # --- Misc. stability guards & model params (Fixed) ----------------
        T_MIN=180.0, T_MAX=330.0, DT_CLIP=15.0, SWE_SMOOTHING=0.01,
        CANOPY_MIN_H=10.0, H_atm=100.0, tau_adv=3600.0,
        # --- Aerodynamic parameters (some ranged) ------------------------
        z_ref_h=15.0, z0_can=1.5, z_ref_soil=2.0, z0_soil=0.01,
        h_trunk_const=5.0, h_trunk_wind_coeff=4.0,
        u_ref_range=(1.0, 5.0),
        # --- Soil & Water parameters ( 일부는 범위로 지정) --------------------
        d_soil_surf=0.3, d_soil_deep=1.7, k_soil=1.2, SWC_max_mm=150.0,
        soil_stress_threshold_range=(0.3, 0.6), T_deep_boundary=270.0,
        # --- Radiation parameters (Ranged) --------------------------------
        k_ext_factor_range=(0.5, 0.7), k_snow_factor_range=(0.75, 0.85), eps_can=0.98,
        eps_snow=0.98, eps_soil=0.95, eps_trunk=0.95, eps_atm_max=0.9,
        eps_atm_min_T=265.0, eps_atm_sensitivity=15.0, eps_atm_coeff_a=0.80,
        eps_atm_coeff_b=0.15, alpha_snow=0.80, alpha_soil=0.20, alpha_trunk=0.25,
        # --- Forcing parameters (Ranged) ----------------------------------
        latitude_deg_range=(56.0, 65.0), T_annual_mean_offset_range=(-10.0, -5.0),
        T_seasonal_amplitude_range=(20.0, 25.0), T_diurnal_amplitude_range=(4.0, 8.0),
        T_hour_peak_diurnal_range=(3.0, 5.0), mean_relative_humidity_range=(0.6, 0.8),
        rain_summer_prob_range=(0.10, 0.20), rain_summer_mm_day_range=(10.0, 20.0),
        rain_shoulder_prob_range=(0.05, 0.15), rain_shoulder_mm_day_range=(5.0, 15.0),
        snow_winter_prob_range=(0.15, 0.30), winter_snow_mm_day_range=(3.0, 8.0),
        summer_day_start=150, summer_day_end=250, shoulder_1_start=90,
        shoulder_1_end=150, shoulder_2_start=250, shoulder_2_end=300,
        snow_season_end=120, snow_season_start=280,
        T_daily_noise_std_range=(1.0, 2.0),  # Daily temperature noise std dev (K)
        # --- Phenology (Fixed) --------------------------------------------
        growth_day=140, fall_day=270, growth_rate=0.1, fall_rate=0.1,
        woody_area_index=0.35,
        # --- Trunk parameters (Fixed) -------------------------------------
        A_trunk_plan=0.03, A_trunk_vert=0.08, k_ct_base_con=0.18,
        k_ct_base_dec=0.12, d_ct=0.1, A_c2t=0.08, k_tsn=0.05, d_tsn=0.1,
        k_tso=0.8, d_tso=0.1, k_snow_pack=0.3,
        # --- Priestley-Taylor coefficient (Fixed) -------------------------
        PT_ALPHA=1.26,
        # --- Photosynthesis & Carbon Cycle (Ranged) -----------------------
        PAR_FRACTION=0.5, LUE_J_TO_G_C=2.5e-6, J_PER_GC=3.9e4,
        R_BASE_KG_M2_YR_range=(0.4, 0.7), Q10_range=(1.8, 2.3), T_REF_K=288.15,
        # --- RL Management Levers (Fixed) ---------------------------------
        MAX_DENSITY_FOR_FULL_CANOPY=1500.0,
        # --- Species-Specific Ranged Parameters ---------------------------
        LAI_max_conifer_range=(3.0, 5.0),
        LAI_max_deciduous_range=(4.0, 6.0),
        alpha_can_base_conifer_range=(0.07, 0.11),
        alpha_can_base_deciduous_range=(0.15, 0.20),
    )

def safe_update(T_old: float, dT: float, p: Dict) -> float:
    dT = np.clip(dT, -p['DT_CLIP'], p['DT_CLIP'])
    return np.clip(T_old + dT, p['T_MIN'], p['T_MAX'])

def esat_kPa(T: float) -> float:
    return 0.6108 * np.exp(17.27 * (T - 273.15) / ((T - 273.15) + 237.3))

def delta_svp_kPa_per_K(T: float) -> float:
    es = esat_kPa(T)
    return 4098.0 * es / ((T - 273.15) + 237.3) ** 2

def get_stability_correction(z_ref: float, L: float) -> Tuple[float, float]:
    if abs(L) < 1e-6: L = -1e-6 if L < 0 else 1e-6
    zeta = z_ref / L
    if zeta >= 0: # Stable
        psi_m = -5.0 * zeta
        psi_h = -5.0 * zeta
    else: # Unstable
        x = (1 - 16 * zeta) ** 0.25
        psi_m = 2 * np.log((1 + x) / 2) + np.log((1 + x**2) / 2) - 2 * np.arctan(x) + np.pi / 2
        psi_h = 2 * np.log((1 + (1 - 16 * zeta) ** 0.5) / 2)
    return psi_m, psi_h

def h_aero(u: float, z_ref: float, z0: float, L: float, p: Dict) -> float:
    u = max(u, 0.1)
    _, psi_h = get_stability_correction(z_ref, L)
    ra_log_term = np.log(z_ref / z0) - psi_h
    if ra_log_term <= 0: ra_log_term = np.log(z_ref / z0)
    ra = ra_log_term**2 / (p['KAPPA']**2 * u)
    return p['RHO_AIR'] * p['CP_AIR'] / max(ra, 1.0)

def get_baseline_parameters(p: Dict, coniferous_fraction: float, stem_density: float) -> Dict[str, Any]:
    """
    Initialise a parameter dictionary `p` for a given species mix,
    drawing from a base model configuration with sampled parameters.
    """
    p['coniferous_fraction'] = coniferous_fraction
    deciduous_fraction = 1.0 - coniferous_fraction
    u = p['u_ref']

    # --- RL MANAGEMENT LEVER MAPPING ---
    a_can_max_potential = 0.95
    p['A_can_max'] = np.clip(stem_density / p['MAX_DENSITY_FOR_FULL_CANOPY'], 0.05, 1.0) * a_can_max_potential

    # --- Mix species-specific parameters based on the coniferous fraction ---
    con_params = dict(
        alpha_can_base=p['alpha_can_base_conifer'], LAI_max=p['LAI_max_conifer'],
        k_ct=p['k_ct_base_con'], can_int_frac_rain=0.25, can_int_frac_snow=0.40
    )
    dec_params = dict(
        alpha_can_base=p['alpha_can_base_deciduous'], LAI_max=p['LAI_max_deciduous'],
        k_ct=p['k_ct_base_dec'], can_int_frac_rain=0.20, can_int_frac_snow=0.25
    )

    mixed_params = {}
    for key in con_params:
        mixed_params[key] = (con_params[key] * coniferous_fraction) + (dec_params[key] * deciduous_fraction)
    p.update(mixed_params)

    # Scale total LAI max by the density-driven A_can_max
    base_lai_max = p['LAI_max']
    p['LAI_max'] = base_lai_max * (p['A_can_max'] / a_can_max_potential)
    p['LAI_max_coniferous'] = p['LAI_max_conifer'] * (p['A_can_max'] / a_can_max_potential)
    p['LAI_max_deciduous'] = p['LAI_max_deciduous'] * (p['A_can_max'] / a_can_max_potential)

    # --- Set derived parameters ---
    p['h_trunk'] = p['h_trunk_const'] + p['h_trunk_wind_coeff'] * u
    p['k_ext'] = p['k_ext_factor']
    p['k_snow'] = p['k_ext'] * p['k_snow_factor']
    p['DT_SECONDS'] = p['TIME_STEP_MINUTES'] * 60
    p['STEPS_PER_DAY'] = int(24 * 60 / p['TIME_STEP_MINUTES'])
    return p

def update_dynamic_parameters(p: Dict, day: int, hour: float, S: dict, L: float, rng: np.random.Generator):
    # --- Air temperature: annual + diurnal + STOCHASTIC offset --------------
    day_angle = 2 * np.pi * (day - 1) / 365.0
    p["T_large_scale"] = 273.15 + p['T_annual_mean_offset'] - p['T_seasonal_amplitude'] * np.cos(day_angle) \
                         + rng.normal(0, p['T_daily_noise_std'])  # Daily stochastic temp
    p["T_atm"] = p["T_large_scale"] - p['T_diurnal_amplitude'] * np.cos(2 * np.pi * (hour - p['T_hour_peak_diurnal']) / 24.0)
    p['ea'] = p['mean_relative_humidity'] * esat_kPa(p['T_atm'])
    swc_frac = S['SWC_mm'] / p['SWC_max_mm']
    stress_range = 1.0 - p['soil_stress_threshold']
    p['soil_stress'] = np.clip((swc_frac - p['soil_stress_threshold']) / stress_range, 0.0, 1.0)

    # --- STOCHASTIC Precipitation & Snowfall --------------------------------
    rain_mm_day, snowfall_mm_day = 0.0, 0.0
    if p['T_atm'] > 274.15:
        if p['summer_day_start'] < day < p['summer_day_end']:
            if rng.random() < p['rain_summer_prob']: rain_mm_day = rng.exponential(p['rain_summer_mm_day'])
        elif (p['shoulder_1_start'] < day < p['shoulder_1_end']) or (p['shoulder_2_start'] < day < p['shoulder_2_end']):
            if rng.random() < p['rain_shoulder_prob']: rain_mm_day = rng.exponential(p['rain_shoulder_mm_day'])
    elif p['T_atm'] < 272.15 and (day > p['snow_season_start'] or day < p['snow_season_end']):
        if rng.random() < p['snow_winter_prob']: snowfall_mm_day = rng.exponential(p['winter_snow_mm_day'])

    p['rain_m_step'] = rain_mm_day / 1000.0 / p['STEPS_PER_DAY']
    p['snowfall_m_step'] = snowfall_mm_day / 1000.0 / p['STEPS_PER_DAY']

    # --- Short-wave radiation (deterministic part) ------------------------
    decl = -23.45 * np.cos(2 * np.pi * (day + 10) / 365.0)
    cos_tz = (np.sin(np.deg2rad(p['latitude_deg'])) * np.sin(np.deg2rad(decl)) +
              np.cos(np.deg2rad(p['latitude_deg'])) * np.cos(np.deg2rad(decl)) * np.cos(np.deg2rad(15 * (hour - 12))))
    p["Q_solar"] = max(0.0, 1000.0 * cos_tz)

    # --- Leaf phenology (mixed species) -----------------------------------
    leaf_on = 1 / (1 + np.exp(-p['growth_rate'] * (day - p['growth_day'])))
    leaf_off = 1 / (1 + np.exp(p['fall_rate'] * (day - p['fall_day'])))
    LAI_deciduous_actual = p["LAI_max_deciduous"] * leaf_on * leaf_off
    LAI_coniferous_actual = p['LAI_max_coniferous']
    p["LAI_actual"] = (LAI_coniferous_actual * p['coniferous_fraction']) + \
                      (LAI_deciduous_actual * (1.0 - p['coniferous_fraction']))
    total_woody_area = p['woody_area_index'] # Simplified from original
    p["LAI"] = p["LAI_actual"] + total_woody_area

    # --- Canopy Interception & Evaporation Flux ---------------------------
    scaling_num = 1 - np.exp(-p['k_snow'] * p['LAI'])
    scaling_den = 1 - np.exp(-p['k_snow'] * (p['LAI_max'] + p['woody_area_index']))
    interception_scaling = scaling_num / scaling_den if scaling_den > 0 else 0.0
    p['rain_intercepted_m'] = p['rain_m_step'] * p['can_int_frac_rain'] * interception_scaling
    p['snow_intercepted_m'] = p['snowfall_m_step'] * p['can_int_frac_snow'] * interception_scaling
    p['evap_intercepted_rain_flux'] = (p['rain_intercepted_m'] * p['Lv'] * p['RHO_WATER']) / p['DT_SECONDS']

    # --- Geometric factors ------------------------------------------------
    p["K_can"] = np.exp(-p["k_ext"] * p["LAI"])
    p["alpha_can"] = p["alpha_can_base"] if p["LAI_actual"] > 0.1 else p["alpha_trunk"]
    p["A_can"] = p["A_can_max"] * (1 - np.exp(-p["k_ext"] * p["LAI"]))
    snow_frac = S['SWE'] / (S['SWE'] + p['SWE_SMOOTHING'])
    p["A_snow"] = (1.0 - p["A_trunk_plan"]) * snow_frac
    p["A_soil"] = 1.0 - p["A_trunk_plan"] - p["A_snow"]
    p['trunk_density_per_m2'] = p['A_can_max']
    p['trunk_radius_m'] = np.sqrt(p['A_trunk_plan'] / (p['trunk_density_per_m2'] * np.pi + p['EPS']))

    # --- Dynamic Aerodynamic conductances ---------------------------------
    u = p['u_ref']
    p['h_can'] = max(h_aero(u, p['z_ref_h'], p['z0_can'], L, p), p['CANOPY_MIN_H'])
    h_soil_raw = h_aero(u, p['z_ref_soil'], p['z0_soil'], L, p)
    p['h_soil'] = max(h_soil_raw, 3.0) * (1 - snow_frac)
    p['h_snow'] = max(h_soil_raw, 3.0) * snow_frac * 0.5
    p["eps_atm"] = min(p['eps_atm_max'], p['eps_atm_coeff_a'] + p['eps_atm_coeff_b'] * np.tanh((p["T_atm"] - p['eps_atm_min_T']) / p['eps_atm_sensitivity']))
    return p

def solve_canopy_energy_balance(T_guess: float, p: Dict, forcings: Dict, conduct: Dict, SWE_can: float) -> Tuple[float, Dict, float]:
    """Newton–Raphson root of canopy EB: Rnet – H – LE – G – Conduction = 0."""
    eps_can, A_can, h_can, k_ct, A_c2t, d_ct = p['eps_can'], p['A_can'], conduct['h_can'], p['k_ct'], p['A_c2t'], p['d_ct']
    PT_ALPHA, soil_stress = p['PT_ALPHA'], forcings['soil_stress']
    evap_intercepted_rain_flux = forcings['evap_intercepted_rain_flux']
    Q_abs_can, L_down_atm, L_up_grnd = forcings['Q_abs_can'], forcings['L_down_atm'], forcings['L_up_ground']
    T_trunk, T_air, ea = forcings['T_trunk'], forcings['T_air'], forcings['ea']

    def rnet(T):
        return Q_abs_can + A_can * (eps_can * (L_down_atm + L_up_grnd) - 2 * eps_can * p['SIGMA'] * T ** 4)

    def latent(T, Rn):
        if T <= 273.15 or Rn <= 0 or p['LAI_actual'] <= 0.1: return 0.0
        vpd = max(0.0, esat_kPa(T) - ea)
        vpd_stress = np.exp(-0.15 * vpd)
        Delta = delta_svp_kPa_per_K(T)
        return PT_ALPHA * (Delta / (Delta + p['PSYCHROMETRIC_GAMMA'])) * Rn * vpd_stress * soil_stress

    def photosynthesis(T, Q_abs):
        if T <= 273.15 or Q_abs <= 0 or p['LAI_actual'] <= 0.1: return 0.0
        vpd = max(0.0, esat_kPa(T) - ea)
        vpd_stress = np.exp(-0.15 * vpd)
        total_stress = vpd_stress * soil_stress
        par_abs = Q_abs * p['PAR_FRACTION']
        # Returns carbon flux in gC/m2/s
        return par_abs * p['LUE_J_TO_G_C'] * total_stress

    melt_energy_sink = 0.0
    if SWE_can > 0:
        T_freeze = 273.15
        Rn_at_freeze = rnet(T_freeze)
        H_at_freeze = h_can * (T_freeze - T_air)
        Cnd_at_freeze = k_ct * A_c2t / d_ct * (T_freeze - T_trunk)
        F_at_freeze = Rn_at_freeze - H_at_freeze - Cnd_at_freeze - evap_intercepted_rain_flux # LE and G are 0 at freeze
        if F_at_freeze > 0:
            energy_to_melt_all = (SWE_can * p['Lf'] * p['RHO_WATER']) / p['DT_SECONDS']
            if F_at_freeze < energy_to_melt_all:
                flux_dict = {'Rnet_can': Rn_at_freeze, 'H_can': -H_at_freeze, 'LE_can': 0, 'G_photo_energy': 0, 'Cnd_can': -Cnd_at_freeze, 'Melt_flux_can': F_at_freeze, 'LE_int_rain': evap_intercepted_rain_flux}
                return T_freeze, flux_dict, 0.0
            else:
                melt_energy_sink = energy_to_melt_all

    T = np.clip(T_guess, p['T_MIN'] + 1.0, p['T_MAX'] - 1.0)
    gpp_g_m2_s = 0.0
    for _ in range(6):
        Rn = rnet(T)
        LE = latent(T, Rn)
        gpp_g_m2_s = photosynthesis(T, Q_abs_can)
        G_photo_energy = gpp_g_m2_s * p['J_PER_GC']
        H = h_can * (T - T_air)
        Cnd = k_ct * A_c2t / d_ct * (T - T_trunk)
        F = Rn - H - LE - G_photo_energy - Cnd - melt_energy_sink - evap_intercepted_rain_flux
        if abs(F) < 1e-3: break

        dT = 0.1
        Rn_p, LE_p = rnet(T + dT), latent(T + dT, rnet(T + dT))
        gpp_p = photosynthesis(T + dT, Q_abs_can)
        G_p = gpp_p * p['J_PER_GC']
        H_p, Cnd_p = h_can * (T + dT - T_air), k_ct * A_c2t / d_ct * (T + dT - T_trunk)
        F_p = Rn_p - H_p - LE_p - G_p - Cnd_p - melt_energy_sink - evap_intercepted_rain_flux
        dF = (F_p - F) / dT
        if abs(dF) < 1e-4: dF = 1e-4
        T -= F / dF
        T = np.clip(T, p['T_MIN'] + 1.0, p['T_MAX'] - 1.0)

    Rn, LE, H, Cnd = rnet(T), latent(T, rnet(T)), h_can * (T - T_air), k_ct * A_c2t / d_ct * (T - T_trunk)
    gpp_g_m2_s = photosynthesis(T, Q_abs_can)
    G_photo_energy = gpp_g_m2_s * p['J_PER_GC']
    flux_dict = {'Rnet_can': Rn, 'H_can': -H, 'LE_can': -LE, 'G_photo_energy': -G_photo_energy, 'Cnd_can': -Cnd, 'Melt_flux_can': melt_energy_sink, 'LE_int_rain': evap_intercepted_rain_flux}
    return T, flux_dict, gpp_g_m2_s

def calculate_fluxes_and_melt(S: Dict, p: Dict) -> Tuple[Dict, float, float, float]:
    """Compute all energy flux components for every model node."""
    T_can_guess, T_trunk, T_snow = S['canopy'], S['trunk'], S['snow']
    T_soil_surf, T_soil_deep, T_air_model = S['soil_surf'], S['soil_deep'], S['atm_model']
    A_can, A_snow, A_soil = p['A_can'], p['A_snow'], p['A_soil']
    eps_can, eps_soil, eps_snow = p['eps_can'], p['eps_soil'], p['eps_snow']
    flux_report = {node: {} for node in S if 'SWE' not in node and 'SWC' not in node}

    lw = lambda T: p['SIGMA'] * T ** 4
    L_down_atm = p['eps_atm'] * lw(p['T_atm'])
    L_emit_soil, L_emit_snow = eps_soil * lw(T_soil_surf), eps_snow * lw(T_snow)
    L_up_ground = (A_soil * L_emit_soil) + (A_snow * L_emit_snow)

    Q_solar_on_canopy = p['Q_solar'] * p['A_can_max']
    Q_abs_can = Q_solar_on_canopy * (1 - p['K_can']) * (1 - p['alpha_can'])

    forcings = {'Q_abs_can': Q_abs_can, 'L_down_atm': L_down_atm, 'L_up_ground': L_up_ground, 'T_trunk': T_trunk, 'T_air': T_air_model, 'ea': p['ea'], 'soil_stress': p['soil_stress'], 'evap_intercepted_rain_flux': p['evap_intercepted_rain_flux']}
    T_can_step, can_flux, gpp_g_m2_s = solve_canopy_energy_balance(T_can_guess, p, forcings, {'h_can': p['h_can']}, S['SWE_can'])

    flux_report['canopy'] = {'Rnet': can_flux['Rnet_can'], 'H': can_flux['H_can'], 'LE_trans': can_flux['LE_can'], 'G_photo': can_flux.get('G_photo_energy', 0.0), 'Cnd_trunk': can_flux['Cnd_can'], 'Melt': -can_flux.get('Melt_flux_can', 0.0), 'LE_int_rain': -can_flux.get('LE_int_rain', 0.0)}

    Q_transmitted = Q_solar_on_canopy * p['K_can']
    Q_ground = p['Q_solar'] * (1 - p['A_can_max']) + Q_transmitted
    flux_report['soil_surf']['SW_in'] = A_soil * Q_ground * (1 - p['alpha_soil'])
    flux_report['snow']['SW_in'] = A_snow * Q_ground * (1 - p['alpha_snow'])

    L_emit_can = eps_can * lw(T_can_step) if A_can > 0 else 0.0
    gap_fraction = 1.0 - A_can
    L_down_transmitted_atm = (1 - eps_can) * L_down_atm
    LW_in_soil = A_soil * eps_soil * (gap_fraction*L_down_atm + A_can*(L_emit_can + L_down_transmitted_atm))
    LW_in_snow = A_snow * eps_snow * (gap_fraction*L_down_atm + A_can*(L_emit_can + L_down_transmitted_atm))
    flux_report['soil_surf']['LW_net'] = LW_in_soil - A_soil * L_emit_soil
    flux_report['snow']['LW_net'] = LW_in_snow - A_snow * L_emit_snow

    # Ensure conduction terms are always present for downstream accounting
    flux_report['snow']['Cnd_soil'] = 0.0
    flux_report['snow']['Cnd_trunk'] = 0.0
    flux_report['soil_surf']['Cnd_snow'] = 0.0
    flux_report['soil_surf']['Cnd_trunk'] = 0.0

    Rn_soil = flux_report['soil_surf']['SW_in'] + flux_report['soil_surf']['LW_net']
    LE_soil = 0.0
    if Rn_soil > 0 and A_soil > 0:
        Delta_soil = delta_svp_kPa_per_K(T_soil_surf)
        LE_soil = p['PT_ALPHA'] * (Delta_soil / (Delta_soil + p['PSYCHROMETRIC_GAMMA'])) * Rn_soil * p['soil_stress']
    flux_report['soil_surf']['LE_evap'] = -LE_soil

    flux_surf_deep = (p['k_soil'] / (0.5*(p['d_soil_surf']+p['d_soil_deep']))) * (T_soil_surf - T_soil_deep)
    flux_deep_bound = (p['k_soil'] / (0.5*p['d_soil_deep'])) * (T_soil_deep - p['T_deep_boundary'])
    flux_report['soil_surf']['Cnd_deep'] = -flux_surf_deep
    flux_report['soil_deep']['Cnd_surf'] = flux_surf_deep
    flux_report['soil_deep']['Cnd_boundary'] = -flux_deep_bound

    if p['A_snow'] > 0:
        snow_depth = (S['SWE'] * p['RHO_WATER']) / p['RHO_SNOW']
        R_soil = (0.5 * p['d_soil_surf']) / p['k_soil']
        R_snow = (0.5 * snow_depth) / p['k_snow_pack']
        flux_soil_snow = (1/(R_soil+R_snow)) * p['A_snow'] * (T_soil_surf - T_snow) if (R_soil+R_snow) > 0 else 0.0
        flux_report['soil_surf']['Cnd_snow'] = -flux_soil_snow
        flux_report['snow']['Cnd_soil'] = flux_soil_snow

    H_trunk, H_soil, H_snow = p['h_trunk'] * (T_trunk - T_air_model), p['h_soil'] * (T_soil_surf - T_air_model), p['h_snow'] * (T_snow - T_air_model)
    flux_report['trunk']['H'], flux_report['soil_surf']['H'], flux_report['snow']['H'] = -H_trunk, -H_soil, -H_snow
    flux_report['atm_model']['H_can'], flux_report['atm_model']['H_trunk'], flux_report['atm_model']['H_soil'], flux_report['atm_model']['H_snow'] = flux_report['canopy']['H'], H_trunk, H_soil, H_snow

    LW_ground_to_atm = (gap_fraction * L_up_ground) + (A_can * (1 - eps_can) * L_up_ground)
    flux_report['atm_model']['LW_up'] = LW_ground_to_atm

    flux_report['atm_model']['Relax'] = (p['C_ATM']/p['tau_adv']) * (p['T_large_scale'] - T_air_model)
    flux_report['trunk']['Cnd_canopy'] = -flux_report['canopy']['Cnd_trunk']

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

    d_SWE_melt_grd, d_SWE_melt_can = 0.0, 0.0
    if S['SWE_can'] > 0 and 'Melt' in flux_report['canopy']:
        d_SWE_melt_can = min((-flux_report['canopy']['Melt'] / (p['Lf'] * p['RHO_WATER'])) * p['DT_SECONDS'], S['SWE_can'])

    net_flux_snow = sum(flux_report.get('snow', {}).values())
    melt_energy_sink_grd = 0.0
    if S['SWE'] > 0 and T_snow >= 273.15 and net_flux_snow > 0:
        d_SWE_melt_grd = min((net_flux_snow / (p['Lf'] * p['RHO_WATER'])) * p['DT_SECONDS'], S['SWE'])
        melt_energy_sink_grd = net_flux_snow
    flux_report['snow']['Melt'] = -melt_energy_sink_grd

    flux_report['canopy']['T_new'] = T_can_step
    return flux_report, d_SWE_melt_grd, d_SWE_melt_can, gpp_g_m2_s


# ----------------------------------------------------------------------------------
# THE SIMULATOR CLASS FOR RL
# ----------------------------------------------------------------------------------
class ForestSimulator:
    def __init__(self, coniferous_fraction: float, stem_density: float, carbon_stock_kg_m2: float, weather_seed: int):
        self.rng = np.random.default_rng(weather_seed)
        self.config = get_model_config()

        # Sample parameters for this specific simulation instance
        self.p = self._sample_parameters()

        # Initialize state and baseline parameters based on management levers
        self.p = get_baseline_parameters(self.p, coniferous_fraction, stem_density)

        self.S = {
            "canopy": 265.0, "trunk": 265.0, "snow": 268.0, "soil_surf": 270.0,
            "soil_deep": 270.0, "atm_model": 265.0, "SWE": 0.0, "SWE_can": 0.0,
            "SWC_mm": self.p['SWC_max_mm'] * 0.75,
        }
        self.carbon_stock_kg_m2 = carbon_stock_kg_m2
        self.L_stability = 1e6

    def _sample_parameters(self) -> Dict[str, Any]:
        """
        Samples values for ranged parameters from the config.
        This ensures each episode of the simulation runs with a slightly
        different but plausible set of physical parameters.
        """
        p = self.config.copy()
        for key, value in self.config.items():
            if key.endswith("_range"):
                sampled_value = self.rng.uniform(low=value[0], high=value[1])
                base_key = key.replace("_range", "")
                p[base_key] = sampled_value
                del p[key]

        return p

    def run_annual_cycle(self, new_conifer_fraction: float, new_stem_density: float) -> dict:
        total_gpp_kg_m2, total_reco_kg_m2, total_thaw_degree_days = 0.0, 0.0, 0.0
        self.p = get_baseline_parameters(self.p, new_conifer_fraction, new_stem_density)
        
        heat_caps = {
            "canopy": self.p['C_CANOPY_LEAF_OFF'], "trunk": self.p['C_TRUNK'], "snow": self.p['C_SNOW'],
            "atm_model": self.p['C_ATM'], "soil_surf": self.p['C_SOIL_TOTAL'] * 0.15,
            "soil_deep": self.p['C_SOIL_TOTAL'] * 0.85,
        }
        temp_nodes = [n for n in self.S if 'SWE' not in n and 'SWC' not in n]

        for day in range(1, 366):
            for t_step in range(self.p['STEPS_PER_DAY']):
                hour = t_step * self.p['TIME_STEP_MINUTES'] / 60.0
                self.p = update_dynamic_parameters(self.p, day, hour, self.S, self.L_stability, self.rng)
                heat_caps['canopy'] = self.p['C_CANOPY_LEAF_ON'] if self.p['LAI_actual'] > 0.1 else self.p['C_CANOPY_LEAF_OFF']

                flux, dSWE_g, dSWE_c, gpp_g_m2_s = calculate_fluxes_and_melt(self.S, self.p)

                gpp_kg_m2_step = gpp_g_m2_s * 1e-3 * self.p['DT_SECONDS']
                total_gpp_kg_m2 += gpp_kg_m2_step

                r_eco_kg_m2_yr = self.p['R_BASE_KG_M2_YR'] * (self.carbon_stock_kg_m2 / 15.0) * \
                                 self.p['Q10']**((self.S['soil_surf'] - self.p['T_REF_K']) / 10.0)
                reco_kg_m2_step = r_eco_kg_m2_yr / (365 * self.p['STEPS_PER_DAY'])
                total_reco_kg_m2 += reco_kg_m2_step

                self.S['SWE_can'] = max(0, self.S['SWE_can'] + self.p['snow_intercepted_m'] - dSWE_c)
                self.S['SWE'] = max(0, self.S['SWE'] + self.p['snowfall_m_step'] - self.p['snow_intercepted_m'] - dSWE_g)

                rain_throughfall = self.p['rain_m_step'] - self.p['rain_intercepted_m']
                water_in_mm = (rain_throughfall + dSWE_g + dSWE_c) * 1000.0

                LE_transpiration = -flux['canopy'].get('LE_trans', 0.0)
                LE_soil_evap = -flux['soil_surf'].get('LE_evap', 0.0)
                water_out_mm = ((LE_transpiration + LE_soil_evap) * self.p['DT_SECONDS']) / (self.p['Lv'] * self.p['RHO_WATER']) * 1000.0

                self.S['SWC_mm'] += water_in_mm - water_out_mm
                runoff_mm = max(0.0, self.S['SWC_mm'] - self.p['SWC_max_mm'])
                self.S['SWC_mm'] -= runoff_mm

                for node in temp_nodes:
                    net_flux = sum(flux.get(node, {}).values())
                    dT = (net_flux / heat_caps[node]) * self.p['DT_SECONDS'] if heat_caps.get(node, 0) > 0 else 0.0
                    self.S[node] = safe_update(self.S[node], dT, self.p) if node != 'canopy' else flux['canopy']['T_new']

                if self.S['soil_surf'] > 273.15:
                    total_thaw_degree_days += (self.S['soil_surf'] - 273.15) / self.p['STEPS_PER_DAY']

                H_total = sum(flux['atm_model'].get(k, 0.0) for k in ['H_can', 'H_trunk', 'H_soil', 'H_snow'])
                if abs(H_total) > 1e-3:
                    u = self.p['u_ref']
                    psi_m, _ = get_stability_correction(self.p['z_ref_h'], self.L_stability)
                    u_star_log_term = np.log(self.p['z_ref_h'] / self.p['z0_can']) - psi_m
                    u_star = u * self.p['KAPPA'] / u_star_log_term if u_star_log_term > 0 else 0.1
                    L_den = self.p['KAPPA'] * self.p['G_ACCEL'] * H_total
                    self.L_stability = -self.p['RHO_AIR'] * self.p['CP_AIR'] * (u_star**3) * self.p['T_atm'] / L_den if abs(L_den) > 1e-9 else 1e6
                else:
                    self.L_stability = 1e6

        net_carbon_change = total_gpp_kg_m2 - total_reco_kg_m2
        self.carbon_stock_kg_m2 += net_carbon_change

        return {"delta_carbon_kg_m2": net_carbon_change, "thaw_degree_days": total_thaw_degree_days}

# ----------------------------------------------------------------------------------
# ORIGINAL DRIVER & PLOTTING (for standalone testing)
# ----------------------------------------------------------------------------------
if __name__ == "__main__":
    print("Running energy_balance_rc.py as a standalone script for testing.")
    print("This demonstrates a single 1-year run of the ForestSimulator.")

    # --- Setup and run a single year simulation ---
    sim = ForestSimulator(
        coniferous_fraction=0.5,
        stem_density=800,
        carbon_stock_kg_m2=15.0,
        weather_seed=123
    )
    
    annual_results = sim.run_annual_cycle(
        new_conifer_fraction=0.5,
        new_stem_density=800
    )

    print("\n--- Annual Simulation Results ---")
    print(f"Net Carbon Change: {annual_results['delta_carbon_kg_m2']:.4f} kg C/m^2/yr")
    print(f"Thaw Degree Days: {annual_results['thaw_degree_days']:.2f} TDD")
    print(f"Final Carbon Stock: {sim.carbon_stock_kg_m2:.4f} kg C/m^2")

    # The original plotting functions could be adapted here to plot data from
    # a detailed history log, if one were added to the simulator class.
    # For now, we just confirm the simulator runs without error.
    print("\nStandalone run complete.")
