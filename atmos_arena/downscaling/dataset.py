import os
import numpy as np
import torch
import h5py
import xarray as xr

from torch.utils.data import Dataset
from glob import glob
from atmos_arena.atmos_utils.data_utils import SINGLE_LEVEL_VARS

class ERA5DownscalingDataset(Dataset):
    def __init__(
        self,
        in_root_dir,
        out_root_dir,
        clim_path,
        in_variables,
        out_variables,
        in_transform,
        out_transform,
    ):
        super().__init__()
        
        self.in_root_dir = in_root_dir
        self.out_root_dir = out_root_dir
        self.in_variables = in_variables
        self.out_variables = out_variables
        self.in_transform = in_transform
        self.out_transform = out_transform
        
        self.in_file_paths = sorted(glob(os.path.join(in_root_dir, '*.h5')))
        self.out_file_paths = sorted(glob(os.path.join(out_root_dir, '*.h5')))
        # remove output file names that are not in input file names
        for out_file in self.out_file_paths:
            basename = os.path.basename(out_file)
            if os.path.join(in_root_dir, basename) not in self.in_file_paths:
                self.out_file_paths.remove(out_file)
        assert len(self.in_file_paths) == len(self.out_file_paths)
        
        # process climatology data
        clim_ds = xr.open_dataset(clim_path)
        clim_dict = {}
        for var in out_variables:
            if var in SINGLE_LEVEL_VARS:
                # reshape to hour, dayofyear, latitude, longitude
                clim_dict[var] = clim_ds[var].values.transpose(0, 1, 3, 2)
                clim_dict[var] = clim_dict[var].reshape(-1, *clim_dict[var].shape[2:])
            else:
                level = int(var.split('_')[-1])
                var_name = '_'.join(var.split('_')[:-1])
                clim_dict[var] = clim_ds[var_name].sel(level=level).values.transpose(0, 1, 3, 2)
                clim_dict[var] = clim_dict[var].reshape(-1, *clim_dict[var].shape[2:])
        self.clim_dict = clim_dict
        
    def __len__(self):
        return len(self.in_file_paths)
    
    def get_data_given_path(self, path, variables):
        with h5py.File(path, 'r') as f:
            data = {
                main_key: {
                    sub_key: np.array(value) for sub_key, value in group.items() if sub_key in variables
            } for main_key, group in f.items() if main_key in ['input']}
        
        data = [data['input'][v] for v in variables]
        return np.stack(data, axis=0)
    
    def __getitem__(self, index):
        in_data = self.get_data_given_path(self.in_file_paths[index], self.in_variables)
        in_data = torch.from_numpy(in_data)
        out_data = self.get_data_given_path(self.out_file_paths[index], self.out_variables)
        out_data = torch.from_numpy(out_data)
        lead_times = torch.Tensor([0.0]).to(dtype=in_data.dtype)
        
        # NOTE: this only works for 1-year test data
        clim = [self.clim_dict[v][index] for v in self.out_variables]
        clim = np.stack(clim, axis=0)
        clim = torch.from_numpy(clim)
        
        return (
            self.in_transform(in_data),
            self.out_transform(out_data),
            clim,
            lead_times,
            self.in_variables,
            self.out_variables,
        )
