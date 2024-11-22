import os
import xarray as xr
import torch
from glob import glob
from torch.utils.data import Dataset
from atmos_arena.chemistry_downscaling.normalization_constants import LOG_MEAN_DICT, LOG_STD_DICT, O3_SCALE_RATIO


class GEOSCFDownscalingDataset(Dataset):
    def __init__(self, root_dir, year_strs, variable, downscale_ratio, eps=1e-32):
        self.root_dir = root_dir
        self.year_strs = year_strs
        self.variable = variable
        self.downscale_ratio = downscale_ratio
        self.eps = eps
        
        # concat root_dir with each year_str and get file paths by globbing
        self.files = []
        for year_str in year_strs:
            self.files += glob(os.path.join(root_dir, year_str, '*.nc4'))
        self.files = sorted(self.files)
        
        if variable != 'O3':
            self.log_mean = LOG_MEAN_DICT[variable]
            self.log_std = LOG_STD_DICT[variable]
        
    def __len__(self):
        return len(self.files)
    
    def normalize(self, x):
        if self.variable != 'O3':
            x = 1 / self.log_std * (torch.log(x + self.eps) - self.log_mean)
        else:
            x = x * O3_SCALE_RATIO
        return x
    
    def denormalize(self, x):
        if self.variable != 'O3':
            x = torch.exp(self.log_mean + self.log_std * x)
        else:
            x = x / O3_SCALE_RATIO
        return x
    
    def downsample(self, x, downsample_ratio):
        if x.shape[0] % downsample_ratio[0] != 0 or x.shape[1] % downsample_ratio[1] != 0:
            raise ValueError("Tensor dimensions must be divisible by the downsampling ratio.")
        
        # Calculate the shape of the downsampled tensor
        new_shape = (x.shape[0] // downsample_ratio[0], downsample_ratio[0], 
                    x.shape[1] // downsample_ratio[1], downsample_ratio[1])
        
        # Reshape and downsample by averaging
        x_reshaped = x.view(new_shape)
        downsampled_x = x_reshaped.mean(dim=(1, 3))
        
        return downsampled_x

    def __getitem__(self, idx: int):
        ds = xr.open_dataset(self.files[idx]).transpose(..., 'lat', 'lon')
        x = torch.from_numpy(ds[self.variable].values.squeeze())
        x = self.normalize(x)
        lowres_x = self.downsample(x, (self.downscale_ratio, self.downscale_ratio))
        lead_times = torch.Tensor([0.0]).to(dtype=x.dtype)
        return lowres_x.unsqueeze(0), x.unsqueeze(0), lead_times, [self.variable]
