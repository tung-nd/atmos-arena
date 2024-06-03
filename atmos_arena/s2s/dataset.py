import os
import numpy as np
import torch
import h5py

from torch.utils.data import Dataset
from glob import glob

class ERA5WindowDataset(Dataset):
    def __init__(
        self,
        root_dir,
        in_variables,
        out_variables,
        in_transform,
        out_transform,
        lead_time,
        data_freq=6, # 1-hourly or 3-hourly or 6-hourly data
        lead_time_divider=100.0,
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
        self.lead_time_divider = lead_time_divider
        
        file_paths = glob(os.path.join(root_dir, '*.h5'))
        self.file_paths = sorted(file_paths)
        
    def __len__(self):
        return len(self.file_paths)
    
    def get_data_given_path(self, path):
        output_key = f'output_{self.lead_time}'
        variables = self.in_variables # out_variables is a subset of in_variables
        with h5py.File(path, 'r') as f:
            data = {
                main_key: {
                    sub_key: np.array(value) for sub_key, value in group.items() if sub_key in variables
            } for main_key, group in f.items() if main_key in ['input', output_key]}
        input = np.stack([data['input'][v] for v in variables], axis=0)
        output = np.stack([data[output_key][v] for v in self.out_variables], axis=0)
        return input, output
    
    # TODO: Confirm that this function works by writing auxiliary test/plot functions (.ipynb)
    
    def __getitem__(self, index):
        path = self.file_paths[index]
        inp_data, out_data = self.get_data_given_path(path)
        inp_data = torch.from_numpy(inp_data)
        out_data = torch.from_numpy(out_data)
        lead_time_tensor = torch.Tensor([self.lead_time]).to(dtype=inp_data.dtype) / self.lead_time_divider
        
        return (
            self.in_transform(inp_data),
            self.out_transform(out_data),
            lead_time_tensor,
            self.in_variables,
            self.out_variables,
        )