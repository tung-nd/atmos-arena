import os
import numpy as np
import torch
import h5py
import xarray as xr

from torch.utils.data import Dataset
from glob import glob
from atmos_arena.atmos_utils.data_utils import SINGLE_LEVEL_VARS

class ERA5InfillingDataset(Dataset):
    def __init__(
        self,
        root_dir,
        clim_path,
        in_variables,
        out_variables,
        in_transform,
        out_transform,
        mask_ratio_range=None,
        predefined_mask_dict=None, # for validation and test only
    ):
        super().__init__()
        
        assert mask_ratio_range is not None or predefined_mask_dict is not None
        
        self.root_dir = root_dir
        self.in_variables = in_variables
        self.out_variables = out_variables
        self.in_transform = in_transform
        self.out_transform = out_transform
        self.mask_ratio_range = mask_ratio_range
        self.predefined_mask_dict = predefined_mask_dict
        
        self.file_paths = sorted(glob(os.path.join(root_dir, '*.h5')))
        if predefined_mask_dict is not None:
            assert len(self.file_paths) == list(predefined_mask_dict.values())[0].shape[0]
            
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
        return len(self.file_paths)
    
    def get_data_given_path(self, path, variables):
        with h5py.File(path, 'r') as f:
            data = {
                main_key: {
                    sub_key: np.array(value) for sub_key, value in group.items() if sub_key in variables
            } for main_key, group in f.items() if main_key in ['input']}
        
        data = [data['input'][v] for v in variables]
        return np.stack(data, axis=0)
    
    def __getitem__(self, index):
        in_data = self.get_data_given_path(self.file_paths[index], self.in_variables) # (V, H, W)
        in_data = torch.from_numpy(in_data) # (V, H, W)
        out_data = self.get_data_given_path(self.file_paths[index], self.out_variables) # (V', H, W)
        out_data = torch.from_numpy(out_data)
        lead_times = torch.Tensor([0.0]).to(dtype=in_data.dtype)
        
        # NOTE: this only works for 1-year test data
        clim = [self.clim_dict[v][index] for v in self.out_variables]
        clim = np.stack(clim, axis=0)
        clim = torch.from_numpy(clim)
        
        if self.predefined_mask_dict is None:
            mask_ratio = np.random.uniform(*self.mask_ratio_range)
            mask = np.random.choice([0, 1], size=in_data.shape[1:], p=[mask_ratio, 1 - mask_ratio]) # (H, W)
            mask = torch.from_numpy(mask).to(dtype=in_data.dtype)
            return (
                self.in_transform(in_data) * mask,
                self.out_transform(out_data),
                clim,
                lead_times,
                mask,
                self.in_variables,
                self.out_variables,
            )
        else:
            mask_dict = {
                ratio: torch.from_numpy(self.predefined_mask_dict[ratio][index]).to(dtype=in_data.dtype) for ratio in self.predefined_mask_dict.keys()
            }
            in_data_dict = {
                ratio: self.in_transform(in_data) * mask_dict[ratio] for ratio in mask_dict.keys()
            }
            return (
                in_data_dict,
                self.out_transform(out_data),
                clim,
                lead_times,
                mask_dict,
                self.in_variables,
                self.out_variables,
            )
