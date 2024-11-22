import os
import numpy as np
import xarray as xr
from tqdm import tqdm
import argparse
from atmos_arena.atmos_utils.data_utils import (
    SINGLE_LEVEL_VARS,
    PRESSURE_LEVEL_VARS,
    DEFAULT_PRESSURE_LEVELS,
    NAME_TO_VAR,
    CHEMISTRY_VARS
)

VAR_LIST = [
    "2m_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "mean_sea_level_pressure",
    "particulate_matter_10um",
    "particulate_matter_1um",
    "particulate_matter_2.5um",
    "total_column_carbon_monoxide",
    "total_column_nitrogen_dioxide",
    "total_column_nitrogen_monoxide",
    "total_column_ozone",
    "total_column_sulphur_dioxide",
    "ozone",
    "specific_humidity",
    "sulphur_dioxide",
    "u_component_of_wind",
    "v_component_of_wind",
]

def main(args):
    list_single_vars = [v for v in VAR_LIST if v in SINGLE_LEVEL_VARS]
    list_pressure_vars = [v for v in VAR_LIST if v in PRESSURE_LEVEL_VARS]

    if not os.path.exists(os.path.join(args.save_dir, f"normalize_mean.npz")):
        normalize_mean = {}
        normalize_std = {}
        normalize_log_mean = {}
        normalize_log_std = {}

        for var in list_single_vars:
            normalize_mean[var] = []
            normalize_std[var] = []
            if var in CHEMISTRY_VARS:
                normalize_log_mean[var] = []
                normalize_log_std[var] = []
        for var in list_pressure_vars:
            for level in DEFAULT_PRESSURE_LEVELS:
                normalize_mean[f'{var}_{level}'] = []
                normalize_std[f'{var}_{level}'] = []
                if var in CHEMISTRY_VARS:
                    normalize_log_mean[f'{var}_{level}'] = []
                    normalize_log_std[f'{var}_{level}'] = []
    else:
        normalize_mean = np.load(os.path.join(args.save_dir, f"normalize_mean.npz"))
        normalize_std = np.load(os.path.join(args.save_dir, f"normalize_std.npz"))
        normalize_log_mean = np.load(os.path.join(args.save_dir, f"normalize_log_mean.npz"))
        normalize_log_std = np.load(os.path.join(args.save_dir, f"normalize_log_std.npz"))
        normalize_mean = {k: list(v) for k, v in normalize_mean.items()}
        normalize_std = {k: list(v) for k, v in normalize_std.items()}
        normalize_log_mean = {k: list(v) for k, v in normalize_log_mean.items()}
        normalize_log_std = {k: list(v) for k, v in normalize_log_std.items()}

    for var in tqdm(list_single_vars + list_pressure_vars, desc='variables', position=0):
        for year in tqdm(args.years, desc='years', position=1, leave=False):
            if var in list_pressure_vars:
                ds = xr.open_dataset(os.path.join(args.root_dir, var, f'{year}.nc'))
            else:
                ds_2017_to_2022 = xr.open_dataset(os.path.join(args.root_dir, var, '2017_to_2022.nc'))
                if year == 2017:
                    start_date = f"{year}-10-01"
                    end_date = f"{year}-12-31"
                elif year == 2022:
                    start_date = f"{year}-01-01"
                    end_date = f"{year}-11-30"
                else:
                    start_date = f"{year}-01-01"
                    end_date = f"{year}-12-31"
                ds = ds_2017_to_2022.sel(time=slice(start_date, end_date))
            
            # chunk to smaller sizes
            if args.chunk_size is not None:
                n_chunks = len(ds.time) // args.chunk_size + 1
            else:
                n_chunks = 1
                args.chunk_size = len(ds.time)
            
            for chunk_id in tqdm(range(n_chunks), desc='chunks', position=2, leave=False):
                ds_small = ds.isel(time=slice(chunk_id*args.chunk_size, (chunk_id+1)*args.chunk_size))
                if var in SINGLE_LEVEL_VARS:
                    ds_np = ds_small[NAME_TO_VAR[var]].values # N, H, W
                    normalize_mean[var].append(np.nanmean(ds_np))
                    normalize_std[var].append(np.nanstd(ds_np))
                    if var in CHEMISTRY_VARS:
                        ds_np_log = np.log(ds_np + 1e-32)
                        normalize_log_mean[var].append(np.nanmean(ds_np_log))
                        normalize_log_std[var].append(np.nanstd(ds_np_log))
                else:
                    ds_np = ds_small[NAME_TO_VAR[var]].values # N, Levels, H, W
                    try:
                        levels_in_ds = ds.level.values
                    except:
                        levels_in_ds = ds.isobaricInhPa.values
                    levels_in_ds = [int(l) for l in levels_in_ds]
                    assert np.sum(np.array(DEFAULT_PRESSURE_LEVELS) - levels_in_ds) == 0 # same order of pressure levels
                    for i, level in enumerate(levels_in_ds):
                        ds_np_lev = ds_np[:, i]
                        normalize_mean[f'{var}_{level}'].append(np.nanmean(ds_np_lev))
                        normalize_std[f'{var}_{level}'].append(np.nanstd(ds_np_lev))
                        if var in CHEMISTRY_VARS:
                            ds_np_lev_log = np.log(ds_np_lev + 1e-32)
                            normalize_log_mean[f'{var}_{level}'].append(np.nanmean(ds_np_lev_log))
                            normalize_log_std[f'{var}_{level}'].append(np.nanstd(ds_np_lev_log))
            
        if var in SINGLE_LEVEL_VARS:
            mean_over_files, std_over_files = np.array(normalize_mean[var]), np.array(normalize_std[var])
            # var(X) = E[var(X|Y)] + var(E[X|Y])
            variance = (std_over_files**2).mean() + (mean_over_files**2).mean() - mean_over_files.mean()**2
            std = np.sqrt(variance)
            # E[X] = E[E[X|Y]]
            mean = mean_over_files.mean()
            normalize_mean[var] = mean.reshape([1])
            normalize_std[var] = std.reshape([1])
            np.savez(os.path.join(args.save_dir, f"normalize_mean.npz"), **normalize_mean)
            np.savez(os.path.join(args.save_dir, f"normalize_std.npz"), **normalize_std)
            
            if var in CHEMISTRY_VARS:
                mean_log_over_files, std_log_over_files = np.array(normalize_log_mean[var]), np.array(normalize_log_std[var])
                variance_log = (std_log_over_files**2).mean() + (mean_log_over_files**2).mean() - mean_log_over_files.mean()**2
                std_log = np.sqrt(variance_log)
                mean_log = mean_log_over_files.mean()
                normalize_log_mean[var] = mean_log.reshape([1])
                normalize_log_std[var] = std_log.reshape([1])
                np.savez(os.path.join(args.save_dir, f"normalize_log_mean.npz"), **normalize_log_mean)
                np.savez(os.path.join(args.save_dir, f"normalize_log_std.npz"), **normalize_log_std)
        else:
            for l in DEFAULT_PRESSURE_LEVELS:
                var_lev = f'{var}_{l}'
                mean_over_files, std_over_files = np.array(normalize_mean[var_lev]), np.array(normalize_std[var_lev])
                # var(X) = E[var(X|Y)] + var(E[X|Y])
                variance = (std_over_files**2).mean() + (mean_over_files**2).mean() - mean_over_files.mean()**2
                std = np.sqrt(variance)
                # E[X] = E[E[X|Y]]
                mean = mean_over_files.mean()
                normalize_mean[var_lev] = mean.reshape([1])
                normalize_std[var_lev] = std.reshape([1])
                
                if var in CHEMISTRY_VARS:
                    mean_log_over_files, std_log_over_files = np.array(normalize_log_mean[var_lev]), np.array(normalize_log_std[var_lev])
                    variance_log = (std_log_over_files**2).mean() + (mean_log_over_files**2).mean() - mean_log_over_files.mean()**2
                    std_log = np.sqrt(variance_log)
                    mean_log = mean_log_over_files.mean()
                    normalize_log_mean[var_lev] = mean_log.reshape([1])
                    normalize_log_std[var_lev] = std_log.reshape([1])
        
            np.savez(os.path.join(args.save_dir, f"normalize_mean.npz"), **normalize_mean)
            np.savez(os.path.join(args.save_dir, f"normalize_std.npz"), **normalize_std)
            if var in CHEMISTRY_VARS:
                np.savez(os.path.join(args.save_dir, f"normalize_log_mean.npz"), **normalize_log_mean)
                np.savez(os.path.join(args.save_dir, f"normalize_log_std.npz"), **normalize_log_std)

    np.savez(os.path.join(args.save_dir, f"normalize_mean.npz"), **normalize_mean)
    np.savez(os.path.join(args.save_dir, f"normalize_std.npz"), **normalize_std)
    np.savez(os.path.join(args.save_dir, f"normalize_log_mean.npz"), **normalize_log_mean)
    np.savez(os.path.join(args.save_dir, f"normalize_log_std.npz"), **normalize_log_std)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compute statistics for CAMS data.')
    parser.add_argument('--root_dir', type=str, required=True, help='Root directory containing CAMS data')
    parser.add_argument('--save_dir', type=str, required=True, help='Directory to save statistics')
    parser.add_argument('--years', nargs='+', type=int, default=range(2017, 2021), help='Years to process')
    parser.add_argument('--chunk_size', type=int, default=100, help='Chunk size for processing')
    args = parser.parse_args()
    main(args)