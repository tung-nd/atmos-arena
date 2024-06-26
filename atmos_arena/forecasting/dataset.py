import os
import numpy as np
import torch
import h5py

from torch.utils.data import Dataset
from glob import glob

class ERA5DirectDataset(Dataset):
    def __init__(
        self,
        root_dir,
        in_variables,
        out_variables,
        in_transform,
        out_transform,
        lead_time,
        data_freq=6, # 1-hourly or 3-hourly or 6-hourly data
    ):
        super().__init__()
        
        # assert out_variables is a subset of in_variables
        assert set(out_variables).issubset(set(in_variables))
        
        self.root_dir = root_dir
        self.in_variables = in_variables
        self.out_variables = out_variables
        self.in_transform = in_transform
        self.out_transform = out_transform
        self.lead_time = lead_time
        self.data_freq = data_freq
        
        file_paths = glob(os.path.join(root_dir, '*.h5'))
        self.file_paths = sorted(file_paths)
        
    def __len__(self):
        return len(self.file_paths) - self.lead_time // self.data_freq
    
    def get_out_path(self, year, inp_file_idx, steps):
        out_file_idx = inp_file_idx + steps
        out_path = os.path.join(
            self.root_dir,
            f'{year}_{out_file_idx:04}.h5'
        )
        if not os.path.exists(out_path):
            for i in range(steps):
                out_file_idx = inp_file_idx + i
                out_path = os.path.join(
                    self.root_dir,
                    f'{year}_{out_file_idx:04}.h5'
                )
                if os.path.exists(out_path):
                    max_step_forward = i
            remaining_steps = steps - max_step_forward
            next_year = year + 1
            out_path = os.path.join(
                self.root_dir,
                f'{next_year}_{remaining_steps-1:04}.h5'
            )
        return out_path
    
    def get_data_given_path(self, path, variables):
        with h5py.File(path, 'r') as f:
            data = {
                main_key: {
                    sub_key: np.array(value) for sub_key, value in group.items() if sub_key in variables
            } for main_key, group in f.items() if main_key in ['input']}
        return np.stack([data['input'][v] for v in variables], axis=0)
    
    def __getitem__(self, index):
        path = self.file_paths[index]
        
        steps = self.lead_time // self.data_freq
        year, inp_file_idx = os.path.basename(path).split('.')[0].split('_')
        year, inp_file_idx = int(year), int(inp_file_idx)
        out_path = self.get_out_path(year, inp_file_idx, steps)
        inp_data = self.get_data_given_path(path, self.in_variables)
        inp_data = torch.from_numpy(inp_data)
        out_data = self.get_data_given_path(out_path, self.out_variables)
        out_data = torch.from_numpy(out_data)
        
        lead_time_tensor = torch.Tensor([self.lead_time]).to(dtype=inp_data.dtype) / 100.0
        
        return (
            self.in_transform(inp_data),
            self.out_transform(out_data),
            lead_time_tensor,
            self.in_variables,
            self.out_variables,
        )