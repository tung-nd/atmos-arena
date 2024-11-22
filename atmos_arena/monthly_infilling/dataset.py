import numpy as np
import torch
import xarray as xr

from torch.utils.data import Dataset
from torchvision.transforms import Normalize

class ERA5MonthlyInfillingDataset(Dataset):
    def __init__(
        self,
        file_list,
        variable='2m_temperature',
    ):
        super().__init__()
        
        self.variable = variable
        
        ds = xr.open_mfdataset(file_list, combine='by_coords')
        # compute month average
        ds = ds.resample(time='M').mean()
        self.data = ds[variable].values # N, H, W
        
        self.mask_ratio_range = None
        self.predefined_mask_dict = None
        self.transform = None
        
    def __len__(self):
        return self.data.shape[0]
    
    def get_mean_std(self):
        return self.data.mean(), self.data.std()
    
    def set_transform(self, mean, std):
        self.transform = Normalize(mean, std)
    
    def set_mask_ratio_range(self, mask_ratio_range):
        self.mask_ratio_range = mask_ratio_range
        
    def set_predefined_mask_dict(self, predefined_mask_dict):
        self.predefined_mask_dict = predefined_mask_dict
    
    def __getitem__(self, index):
        in_data = self.data[index] # (H, W)
        in_data = torch.from_numpy(in_data).unsqueeze(0) # (1, H, W)
        lead_times = torch.Tensor([0.0]).to(dtype=in_data.dtype)
        if self.predefined_mask_dict is None:
            mask_ratio = np.random.uniform(*self.mask_ratio_range)
            mask = np.random.choice([0, 1], size=in_data.shape[1:], p=[mask_ratio, 1 - mask_ratio]) # (H, W)
            mask = torch.from_numpy(mask).to(dtype=in_data.dtype)
            return (
                self.transform(in_data) * mask,
                self.transform(in_data),
                lead_times,
                mask,
                [self.variable],
                [self.variable]
            )
        else:
            mask_dict = {
                ratio: torch.from_numpy(self.predefined_mask_dict[ratio][index]).to(dtype=in_data.dtype) for ratio in self.predefined_mask_dict.keys()
            }
            in_data_dict = {
                ratio: self.transform(in_data) * mask_dict[ratio] for ratio in mask_dict.keys()
            }
            return (
                in_data_dict,
                self.transform(in_data),
                lead_times,
                mask_dict,
                [self.variable],
                [self.variable]
            )
