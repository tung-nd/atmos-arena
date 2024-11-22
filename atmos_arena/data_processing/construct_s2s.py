import os
import numpy as np
import torch
import h5py
from glob import glob
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import argparse


def get_out_path(root_dir, year, inp_file_idx, steps):
    out_file_idx = inp_file_idx + steps
    out_path = os.path.join(
        root_dir,
        f'{year}_{out_file_idx:04}.h5'
    )
    if not os.path.exists(out_path):
        for i in range(steps):
            out_file_idx = inp_file_idx + i
            out_path = os.path.join(
                root_dir,
                f'{year}_{out_file_idx:04}.h5'
            )
            if os.path.exists(out_path):
                max_step_forward = i
        remaining_steps = steps - max_step_forward
        next_year = year + 1
        out_path = os.path.join(
            root_dir,
            f'{next_year}_{remaining_steps-1:04}.h5'
        )
    return out_path

def get_data_given_path(path, variables):
    with h5py.File(path, 'r') as f:
        data = {
            main_key: {
                sub_key: np.array(value) for sub_key, value in group.items() if sub_key in variables
        } for main_key, group in f.items() if main_key in ['input']}
    return np.stack([data['input'][v] for v in variables], axis=0), data['input']

def read_and_process_data(root_dir_split, year, inp_file_idx, steps, variables):
    out_data = {}
    for step in steps:
        path = get_out_path(root_dir_split, year, inp_file_idx, step)
        data, _ = get_data_given_path(path, variables)
        out_data[step] = torch.from_numpy(data)
    return out_data

def process_file(path, root_dir_split, save_dir_split, variables, lead_time_list, data_freq, window_size_steps):
    year, inp_file_idx = os.path.basename(path).split('.')[0].split('_')
    year, inp_file_idx = int(year), int(inp_file_idx)
    inp_data, inp_dict = get_data_given_path(path, variables)
    inp_data = torch.from_numpy(inp_data)

    step_dict = {l: l // data_freq for l in lead_time_list}
    min_steps = min(step_dict.values())
    max_steps = max(step_dict.values())
    all_steps = range(min_steps, max_steps + window_size_steps)
    
    # Use ThreadPoolExecutor to parallelize data reading across steps
    num_cpus = multiprocessing.cpu_count()
    with ThreadPoolExecutor(max_workers=num_cpus) as executor:
        out_data_dict = executor.submit(read_and_process_data, root_dir_split, year, inp_file_idx, all_steps, variables).result()

    accumulated_data = {l: torch.zeros_like(inp_data) for l in lead_time_list}
    for step_i in all_steps:
        for l in lead_time_list:
            if step_i >= step_dict[l] and step_i < step_dict[l] + window_size_steps:
                accumulated_data[l] += out_data_dict[step_i]

    accumulated_data = {l: accumulated_data[l] / window_size_steps for l in lead_time_list}

    data_dict = {'input': inp_dict}
    for l in lead_time_list:
        data_dict[f'output_{l}'] = {
            v: accumulated_data[l][i].numpy() for i, v in enumerate(variables)
        }
    
    save_path = os.path.join(save_dir_split, os.path.basename(path))
    with h5py.File(save_path, 'w', libver='latest') as f:
        for main_key, sub_dict in data_dict.items():
            group = f.create_group(main_key)
            for sub_key, array in sub_dict.items():
                group.create_dataset(sub_key, data=array, compression=None, dtype=np.float32)


variables = [
    'angle_of_sub_gridscale_orography',
    'geopotential_at_surface',
    'high_vegetation_cover',
    'lake_cover',
    'lake_depth',
    'land_sea_mask',
    'low_vegetation_cover',
    'slope_of_sub_gridscale_orography',
    'soil_type',
    'standard_deviation_of_filtered_subgrid_orography',
    'standard_deviation_of_orography',
    'type_of_high_vegetation',
    'type_of_low_vegetation',
    '2m_temperature',
    '10m_u_component_of_wind',
    '10m_v_component_of_wind',
    '10m_wind_speed',
    'mean_sea_level_pressure',
    'geopotential_50', 'geopotential_100', 'geopotential_150', 'geopotential_200', 'geopotential_250', 'geopotential_300', 'geopotential_400', 'geopotential_500', 'geopotential_600', 'geopotential_700', 'geopotential_850', 'geopotential_925', 'geopotential_1000',
    'specific_humidity_50', 'specific_humidity_100', 'specific_humidity_150', 'specific_humidity_200', 'specific_humidity_250', 'specific_humidity_300', 'specific_humidity_400', 'specific_humidity_500', 'specific_humidity_600', 'specific_humidity_700', 'specific_humidity_850', 'specific_humidity_925', 'specific_humidity_1000', 'temperature_50',
    'temperature_100', 'temperature_150', 'temperature_200', 'temperature_250', 'temperature_300', 'temperature_400', 'temperature_500', 'temperature_600', 'temperature_700', 'temperature_850', 'temperature_925', 'temperature_1000',
    'u_component_of_wind_50', 'u_component_of_wind_100', 'u_component_of_wind_150', 'u_component_of_wind_200', 'u_component_of_wind_250', 'u_component_of_wind_300', 'u_component_of_wind_400', 'u_component_of_wind_500', 'u_component_of_wind_600', 'u_component_of_wind_700', 'u_component_of_wind_850', 'u_component_of_wind_925', 'u_component_of_wind_1000',
    'v_component_of_wind_50', 'v_component_of_wind_100', 'v_component_of_wind_150', 'v_component_of_wind_200', 'v_component_of_wind_250', 'v_component_of_wind_300', 'v_component_of_wind_400', 'v_component_of_wind_500', 'v_component_of_wind_600', 'v_component_of_wind_700', 'v_component_of_wind_850', 'v_component_of_wind_925', 'v_component_of_wind_1000',
    'vertical_velocity_50', 'vertical_velocity_100', 'vertical_velocity_150', 'vertical_velocity_200', 'vertical_velocity_250', 'vertical_velocity_300', 'vertical_velocity_400', 'vertical_velocity_500', 'vertical_velocity_600', 'vertical_velocity_700', 'vertical_velocity_850', 'vertical_velocity_925', 'vertical_velocity_1000',
    'wind_speed_50', 'wind_speed_100', 'wind_speed_150', 'wind_speed_200', 'wind_speed_250', 'wind_speed_300', 'wind_speed_400', 'wind_speed_500', 'wind_speed_600', 'wind_speed_700', 'wind_speed_850', 'wind_speed_925', 'wind_speed_1000'
]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process weather data for S2S prediction.')
    parser.add_argument('--root_dir', type=str, required=True, help='Root directory containing input data')
    parser.add_argument('--save_dir', type=str, required=True, help='Directory to save processed data')
    parser.add_argument('--data_freq', type=int, default=6, help='Data frequency in hours')
    parser.add_argument('--window_size_days', type=int, default=14, help='Window size in days')
    parser.add_argument('--lead_time_list', nargs='+', type=int, default=[336, 672, 1008, 1344],
                      help='List of lead times in hours')
    args = parser.parse_args()

    # Convert window size from days to steps
    window_size_steps = (args.window_size_days * 24) // args.data_freq
    max_lead_time = max(args.lead_time_list)

    # Process each split
    num_cpus = multiprocessing.cpu_count()
    for split in ['train', 'val', 'test']:
        save_dir_split = os.path.join(args.save_dir, split)
        os.makedirs(save_dir_split, exist_ok=True)
        root_dir_split = os.path.join(args.root_dir, split)
        file_paths = sorted(glob(os.path.join(root_dir_split, '*.h5')))
        
        num_files = len(file_paths) - max_lead_time // args.data_freq - window_size_steps
        
        with ProcessPoolExecutor(max_workers=num_cpus) as executor:
            list(tqdm(executor.map(process_file, 
                                 file_paths[:num_files],
                                 [root_dir_split]*num_files,
                                 [save_dir_split]*num_files,
                                 [variables]*num_files,
                                 [args.lead_time_list]*num_files,
                                 [args.data_freq]*num_files,
                                 [window_size_steps]*num_files),
                     total=num_files,
                     desc=f'Processing {split} data',
                     unit='files'))