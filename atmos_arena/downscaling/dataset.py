import os
import numpy as np
import torch
import h5py

from torch.utils.data import Dataset
from glob import glob

class ERA5DownscalingDataset(Dataset):
    def __init__(
        self,
        in_root_dir,
        out_root_dir,
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
        
        return (
            self.in_transform(in_data),
            self.out_transform(out_data),
            lead_times,
            self.in_variables,
            self.out_variables,
        )