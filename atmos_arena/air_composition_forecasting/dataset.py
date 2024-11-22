import os
import numpy as np
import torch
import h5py

from torch.utils.data import Dataset
from glob import glob
from atmos_arena.atmos_utils.data_utils import CHEMISTRY_VARS

class CAMSDirectDataset(Dataset):
    def __init__(
        self,
        root_dir,
        in_variables,
        out_variables,
        mean_dict,
        std_dict,
        log_mean_dict,
        log_std_dict,
        lead_time,
        data_freq=12, # 12-hourly data by default
        lead_time_divisor=100.0,
    ):
        super().__init__()
        
        # assert out_variables is a subset of in_variables
        assert set(out_variables).issubset(set(in_variables))
        
        self.root_dir = root_dir
        self.in_variables = in_variables
        self.out_variables = out_variables
        self.mean_dict = {k: torch.Tensor(v) for k, v in mean_dict.items()}
        self.std_dict = {k: torch.Tensor(v) for k, v in std_dict.items()}
        self.log_mean_dict = {k: torch.Tensor(v) for k, v in log_mean_dict.items()}
        self.log_std_dict = {k: torch.Tensor(v) for k, v in log_std_dict.items()}
        self.lead_time = lead_time
        self.data_freq = data_freq
        self.lead_time_divisor = lead_time_divisor
        
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
    
    def normalize(self, x, variables):
        # x: 1, V, H, W
        assert len(x.shape) == 4
        assert x.shape[1] == len(variables)
        for i, v in enumerate(variables):
            # if any variable in CHEMISTRY_VARS appears in v
            if any([c in v for c in CHEMISTRY_VARS]):
                x[:, i] = 1 / self.log_std_dict[v] * (torch.log(x[:, i] + 1e-32) - self.log_mean_dict[v])
            else:
                x[:, i] = 1 / self.std_dict[v] * (x[:, i] - self.mean_dict[v])
        return x
    
    def denormalize(self, x, variables):
        # x: 1, V, H, W
        assert len(x.shape) == 4
        assert x.shape[1] == len(variables)
        for i, v in enumerate(variables):
            # if any variable in CHEMISTRY_VARS appears in v
            if any([c in v for c in CHEMISTRY_VARS]):
                x[:, i] = self.log_mean_dict[v].to(device=x.device) + self.log_std_dict[v].to(device=x.device) * x[:, i] # keep in log space
            else:
                x[:, i] = self.std_dict[v].to(device=x.device) * x[:, i] + self.mean_dict[v].to(device=x.device)
        return x
    
    def __getitem__(self, index):
        path = self.file_paths[index]
        
        steps = self.lead_time // self.data_freq
        year, inp_file_idx = os.path.basename(path).split('.')[0].split('_')
        year, inp_file_idx = int(year), int(inp_file_idx)
        out_path = self.get_out_path(year, inp_file_idx, steps)
        inp_data = self.get_data_given_path(path, self.in_variables)
        inp_data = torch.from_numpy(inp_data).unsqueeze(0)
        out_data = self.get_data_given_path(out_path, self.out_variables)
        out_data = torch.from_numpy(out_data).unsqueeze(0)
        
        lead_time_tensor = torch.Tensor([self.lead_time]).to(dtype=inp_data.dtype) / self.lead_time_divisor
        
        return (
            self.normalize(inp_data, self.in_variables),
            self.normalize(out_data, self.out_variables),
            lead_time_tensor,
            self.in_variables,
            self.out_variables,
        )
