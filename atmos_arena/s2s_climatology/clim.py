import os
os.sys.path.append('/localhome/prateiksinha/atmos-arena/atmos_arena/atmos_utils')
import numpy as np
from metrics import *
import torch
import csv
import pandas as pd
import xarray as xr

def load_clim(var, lvl=None):
    file_path = '/localhome/data/datasets/climate/climatology.nc'
    ds = xr.open_dataset(file_path)
    if lvl:
        return ds[var].sel(level=lvl).to_numpy()
    else:
        return ds[var].to_numpy()

def load_test(var, year, lvl=None):
    test_path = f'/localhome/data/datasets/climate/era5_corpenicus/{var}/{year}.nc'
    dataset = xr.open_dataset(test_path)
    var_abbrv = list(dataset.variables.keys())[-1]
    if lvl:
        return dataset[var_abbrv].sel(level=lvl).to_numpy()
    else:
        return dataset[var_abbrv].to_numpy()

device = 'cuda'
log_dir = '/localhome/prateiksinha/atmos-arena/atmos_arena/s2s_stormer/clim_data'
lat = np.arange(90,-90.25,-0.25)

variables = [
    '2m_temperature',
    'temperature',
    'specific_humidity',
    'geopotential'
]
levels = [
    None,
    850,
    700,
    500
]


years = ['2020', '2021', '2022', '2023']

for var, lvl in zip(variables, levels):
    
    print(f'loading {var} climatology...')
    clim = load_clim(var, lvl)
    clim = clim.swapaxes(0,1).reshape((1464, 721, 1440))
    clim = torch.tensor(clim).to(device)
    print('done')

    for year in years:
        print(f'loading {var} test data for {year}...')
        test = load_test(var, year, lvl)
        test = torch.tensor(test).to(device)
        print('done')

        window_size = 14 * 4

        csv_path = f'{log_dir}/{year}_{var}.csv'
        
        file = open(csv_path, mode='w', newline='')
        writer = csv.writer(file)
        writer.writerow(['year', 'variable', 'start', 'mse', 'weight_mse', 'weighted_rmse', 'bias']) 

        with torch.no_grad():
            for i in range(1464 - window_size):
                clim_window = torch.mean(clim[i:i+window_size,:,:], dim=0)[None,None,:,:]
                test_window = torch.mean(test[i:i+window_size,:,:], dim=0)[None,None,:,:]
                
                mse_ = mse(clim_window, test_window, [var])['loss'].item()
                l_mse = lat_weighted_mse(clim_window, test_window, [var], lat)['loss'].item()
                l_rmse = lat_weighted_rmse(clim_window, test_window, lambda x:x, [var], lat, None)['w_rmse']
                bias = lat_weighted_mean_bias(clim_window, test_window, lambda x:x, [var], lat, None)['mean_bias']
                
                writer.writerow([year, var, i, mse_, l_mse, l_rmse, bias])    
                print(i, end = '\r')
        
        file.close()
        print(f'saved csv for {var} & {year}')
        print()
        
csvs = []

with open(f'{log_dir}/final_metrics.txt', 'w') as f:
    for var in variables:
        f.write(f'{var}:\n')
        for year in years:
            csvs.append(f'{log_dir}/{year}_{var}.csv')
        all_years = pd.concat([pd.read_csv(x) for x in csvs])
        for metric in ['mse', 'weight_mse', 'weighted_rmse', 'bias']:
            f.write(f'{metric}: {all_years[metric].mean()}\n')
        f.write('\n')
        csvs = []    