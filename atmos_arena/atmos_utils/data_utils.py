NAME_TO_CMIP = {
    "geopotential": "zg",
    "u_component_of_wind": "ua",
    "v_component_of_wind": "va",
    "temperature": "ta",
    "relative_humidity": "r",
    "specific_humidity": "hus",
}

NAME_TO_VAR = {
    "mean_surface_latent_heat_flux": "mslhf",
    "mean_surface_net_long_wave_radiation_flux": "msnlwrf",
    "mean_surface_net_short_wave_radiation_flux": "msnswrf",
    "mean_surface_sensible_heat_flux": "msshf",
    "mean_top_downward_short_wave_radiation_flux": "mtdnswrf",
    "mean_top_net_long_wave_radiation_flux": "mtnlwrf",
    "mean_top_net_short_wave_radiation_flux": "mtnswrf",
    "skin_temperature": "skt",
    "snow_depth": "snd",
    
    "2m_temperature": "t2m",
    "10m_u_component_of_wind": "u10",
    "10m_v_component_of_wind": "v10",
    "mean_sea_level_pressure": "msl",
    "10m_wind_speed": "w10",
    "surface_pressure": "sp",
    "toa_incident_solar_radiation": "tisr",
    "toa_incident_solar_radiation_6hr": "tisr_6hr",
    "toa_incident_solar_radiation_12hr": "tisr_12hr",
    "toa_incident_solar_radiation_24hr": "tisr_24hr",
    "total_precipitation": "tp",
    "total_precipitation_6hr": "tp_6hr",
    "total_precipitation_12hr": "tp_12hr",
    "total_precipitation_24hr": "tp_24hr",
    "land_sea_mask": "lsm",
    "orography": "orography",
    "slt": "slt",
    "lattitude": "lat2d",
    "longitude": "lon2d",
    "geopotential": "z",
    "u_component_of_wind": "u",
    "v_component_of_wind": "v",
    "vertical_velocity": "vel",
    "temperature": "t",
    "relative_humidity": "r",
    "specific_humidity": "q",
    "vorticity": "vo",
    "potential_vorticity": "pv",
    "total_cloud_cover": "tcc",
    
    "carbon_monoxide": "co",
    "nitrogen_dioxide": "no2",
    "nitrogen_monoxide": "no",
    "ozone": "go3",
    "particulate_matter_10um": "pm10",
    "particulate_matter_1um": "pm1",
    "particulate_matter_2.5um": "pm2p5",
    "sulphur_dioxide": "so2",
    "total_column_carbon_monoxide": "tcco",
    "total_column_nitrogen_dioxide": "tcno2",
    "total_column_nitrogen_monoxide": "tc_no",
    "total_column_ozone": "gtco3",
    "total_column_sulphur_dioxide": "tcso2",
}

VAR_TO_NAME = {v: k for k, v in NAME_TO_VAR.items()}

SINGLE_LEVEL_VARS = [
    "mean_surface_latent_heat_flux",
    "mean_surface_net_long_wave_radiation_flux",
    "mean_surface_net_short_wave_radiation_flux",
    "mean_surface_sensible_heat_flux",
    "mean_top_downward_short_wave_radiation_flux",
    "mean_top_net_long_wave_radiation_flux",
    "mean_top_net_short_wave_radiation_flux",
    "skin_temperature",
    "snow_depth",
    
    "2m_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "mean_sea_level_pressure",
    "10m_wind_speed",
    "surface_pressure",
    'sea_ice_cover',
    'sea_surface_temperature',
    "toa_incident_solar_radiation",
    "toa_incident_solar_radiation_6hr",
    "toa_incident_solar_radiation_12hr",
    "toa_incident_solar_radiation_24hr",
    'total_precipitation_6hr',
    'total_column_water_vapour',
    "total_precipitation_6hr",
    "total_precipitation_12hr",
    "total_precipitation_24hr",
    "total_cloud_cover",
    "land_sea_mask",
    "orography",
    "lattitude",
    
    "particulate_matter_10um",
    "particulate_matter_1um",
    "particulate_matter_2.5um",
    "total_column_carbon_monoxide",
    "total_column_nitrogen_dioxide",
    "total_column_nitrogen_monoxide",
    "total_column_ozone",
    "total_column_sulphur_dioxide",
]

PRESSURE_LEVEL_VARS = [
    "geopotential",
    "u_component_of_wind",
    "v_component_of_wind",
    "vertical_velocity",
    "wind_speed",
    "temperature",
    "relative_humidity",
    "specific_humidity",
    "vorticity",
    "potential_vorticity",
    
    "carbon_monoxide",
    "nitrogen_dioxide",
    "nitrogen_monoxide",
    "ozone",
    "sulphur_dioxide",
]

CHEMISTRY_VARS = [
    "particulate_matter_10um",
    "particulate_matter_1um",
    "particulate_matter_2.5um",
    "total_column_carbon_monoxide",
    "total_column_nitrogen_dioxide",
    "total_column_nitrogen_monoxide",
    "total_column_ozone",
    "total_column_sulphur_dioxide",
    "carbon_monoxide",
    "nitrogen_dioxide",
    "nitrogen_monoxide",
    "ozone",
    "sulphur_dioxide",
]

DEFAULT_PRESSURE_LEVELS = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]

single_level_weight_dict = {
    '2m_temperature': 1.0,
    '10m_u_component_of_wind': 0.1,
    '10m_v_component_of_wind': 0.1,
    'mean_sea_level_pressure': 0.1,
}

pressure_weights = [l / sum(DEFAULT_PRESSURE_LEVELS) for l in DEFAULT_PRESSURE_LEVELS]
pressure_level_weight_dict = {}
for var in PRESSURE_LEVEL_VARS:
    for l, w in zip(DEFAULT_PRESSURE_LEVELS, pressure_weights):
        pressure_level_weight_dict[var + "_" + str(l)] = w

WEIGHT_DICT = {**single_level_weight_dict, **pressure_level_weight_dict}

CONSTANTS = [
    "anisotropy_of_sub_gridscale_orography",
    "orography",
    "land_sea_mask",
    "slt",
    "lattitude",
    "longitude",
    "angle_of_sub_gridscale_orography",
    "geopotential_at_surface",
    "high_vegetation_cover",
    "lake_cover",
    "lake_depth",
    "low_vegetation_cover",
    "slope_of_sub_gridscale_orography",
    "soil_type",
    "standard_deviation_of_filtered_subgrid_orography",
    "standard_deviation_of_orography",
    "type_of_high_vegetation",
    "type_of_low_vegetation",
]

NAME_LEVEL_TO_VAR_LEVEL = {}

for var in SINGLE_LEVEL_VARS:
    if var in NAME_TO_VAR:
        NAME_LEVEL_TO_VAR_LEVEL[var] = NAME_TO_VAR[var]

for var in PRESSURE_LEVEL_VARS:
    if var in NAME_TO_VAR:
        for l in DEFAULT_PRESSURE_LEVELS:
            NAME_LEVEL_TO_VAR_LEVEL[var + "_" + str(l)] = NAME_TO_VAR[var] + "_" + str(l)

VAR_LEVEL_TO_NAME_LEVEL = {v: k for k, v in NAME_LEVEL_TO_VAR_LEVEL.items()}