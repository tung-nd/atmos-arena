import os
import argparse
import numpy as np
import xarray as xr
import h5py
from tqdm import tqdm
from atmos_arena.atmos_utils.data_utils import (
    SINGLE_LEVEL_VARS,
    PRESSURE_LEVEL_VARS,
    DEFAULT_PRESSURE_LEVELS,
    NAME_TO_VAR,
)

LIST_VARS = [
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

def create_one_step_dataset(root_dir, save_dir, split, years, chunk_size=None):
    save_dir_split = os.path.join(save_dir, split)
    os.makedirs(save_dir_split, exist_ok=True)
    
    list_single_vars = [v for v in LIST_VARS if v in SINGLE_LEVEL_VARS]
    list_pressure_vars = [v for v in LIST_VARS if v in PRESSURE_LEVEL_VARS]
    
    for year in tqdm(years, desc='years', position=0):
        ds_sample = xr.open_dataset(os.path.join(root_dir, list_pressure_vars[0], f'{year}.nc'))
        if chunk_size is not None:
            n_chunks = len(ds_sample.time) // chunk_size + 1
        else:
            n_chunks = 1
            chunk_size = len(ds_sample.time)
        
        idx_in_year = 0
        
        ds_dict = {}
        for var in list_pressure_vars:
            ds_dict[var] = xr.open_dataset(os.path.join(root_dir, var, f'{year}.nc'))
        for var in list_single_vars:
            ds_2017_to_2022 = xr.open_dataset(os.path.join(root_dir, var, '2017_to_2022.nc'))
            if year == 2017:
                start_date = f"{year}-10-01"
                end_date = f"{year}-12-31"
            elif year == 2022:
                start_date = f"{year}-01-01"
                end_date = f"{year}-11-30"
            else:
                start_date = f"{year}-01-01"
                end_date = f"{year}-12-31"
            ds_dict[var] = ds_2017_to_2022.sel(time=slice(start_date, end_date))

        for chunk_id in tqdm(range(n_chunks), desc='chunks', position=1, leave=False):
            dict_np = {}
            list_time_stamps = None
            ### convert ds to numpy
            for var in (list_single_vars + list_pressure_vars):
                ds = ds_dict[var].isel(time=slice(chunk_id*chunk_size, (chunk_id+1)*chunk_size))
                if list_time_stamps is None:
                    list_time_stamps = ds.time.values
                if var in list_single_vars:
                    dict_np[var] = ds[NAME_TO_VAR[var]].values
                else:
                    try:
                        available_levels = ds.isobaricInhPa.values
                    except:
                        available_levels = ds.level.values
                    available_levels = [int(l) for l in available_levels]
                    ds_np = ds[NAME_TO_VAR[var]].values
                    for i, level in enumerate(available_levels):
                        if level in DEFAULT_PRESSURE_LEVELS:
                            dict_np[f'{var}_{level}'] = ds_np[:, i]
                    
            for i in tqdm(range(len(list_time_stamps)), desc='time stamps', position=2, leave=False):
                data_dict = {
                    'input': {'time': str(list_time_stamps[i])}
                }
                for var in dict_np.keys():
                    data_dict['input'][var] = dict_np[var][i]
                    
                with h5py.File(os.path.join(save_dir_split, f'{year}_{idx_in_year:04}.h5'), 'w', libver='latest') as f:
                    for main_key, sub_dict in data_dict.items():
                        group = f.create_group(main_key)
                        for sub_key, array in sub_dict.items():
                            if sub_key != 'time':
                                group.create_dataset(sub_key, data=array, compression=None, dtype=np.float32)
                            else:
                                group.create_dataset(sub_key, data=array, compression=None)
                
                idx_in_year += 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create one-step dataset from CAMS data.')
    parser.add_argument("--root_dir", type=str, required=True, help='Input root directory')
    parser.add_argument("--save_dir", type=str, required=True, help='Output directory')
    parser.add_argument("--start", type=int, required=True, help='Start year')
    parser.add_argument("--end", type=int, required=True, help='End year')
    parser.add_argument("--split", type=str, required=True, help='Data split (train/val/test)')
    parser.add_argument("--chunk_size", type=int, default=100, help='Chunk size for processing')
    
    args = parser.parse_args()
    
    create_one_step_dataset(
        root_dir=args.root_dir,
        save_dir=args.save_dir,
        split=args.split,
        years=list(range(args.start, args.end)),
        chunk_size=args.chunk_size
    )