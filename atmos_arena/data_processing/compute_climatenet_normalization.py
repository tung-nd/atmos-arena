import os
import xarray as xr
import numpy as np
from glob import glob
from tqdm import tqdm
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculate normalization statistics.')
    parser.add_argument('--root_dir', type=str, required=True,
                      help='Root directory containing the dataset')
    args = parser.parse_args()

    nc_files = sorted(glob(os.path.join(args.root_dir, 'train', '*.nc')))
    all_variables = [
        'TMQ',
        'U850',
        'V850',
        'UBOT',
        'VBOT',
        'QREFHT',
        'PS',
        'PSL',
        'T200',
        'T500',
        'PRECT',
        'TS',
        'TREFHT',
        'Z1000',
        'Z200',
        'ZBOT',
        'WS850',
        'WSBOT',
        'VRT850',
        'VRTBOT'
    ]
    normalize_mean = {v: [] for v in all_variables}
    normalize_std = {v: [] for v in all_variables}

    for nc_file in tqdm(nc_files):
        ds = xr.open_dataset(nc_file)
        for v in all_variables:
            normalize_mean[v].append(ds[v].mean().item())
            normalize_std[v].append(ds[v].std().item())

    normalize_mean = {v: np.array(normalize_mean[v]) for v in all_variables}
    normalize_std = {v: np.array(normalize_std[v]) for v in all_variables}
    for var in all_variables:
        # var(X) = E[var(X|Y)] + var(E[X|Y])
        variance = (normalize_std[var]**2).mean() + (normalize_mean[var]**2).mean() - normalize_mean[var].mean()**2
        normalize_std[var] = np.sqrt(variance).reshape([1])
        normalize_mean[var] = normalize_mean[var].mean().reshape([1])
        
    np.savez(os.path.join(args.root_dir, f"normalize_mean.npz"), **normalize_mean)
    np.savez(os.path.join(args.root_dir, f"normalize_std.npz"), **normalize_std)