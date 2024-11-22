import os
import xarray as xr
import torch
from glob import glob
from torch.utils.data import Dataset


class ClimateNetDataset(Dataset):
    def __init__(self, root_dir, in_variables, transform):
        self.root_dir = root_dir
        self.in_variables = in_variables
        self.transform = transform
        self.files = sorted(glob(os.path.join(root_dir, '*.nc')))
        
    def __len__(self):
        return len(self.files)

    def get_features(self, dataset: xr.Dataset):
        features = dataset[list(self.in_variables)].to_array().transpose('time', 'variable', 'lat', 'lon')
        features = features.to_numpy()
        features = self.transform(torch.from_numpy(features))
        return features.squeeze(0)

    def __getitem__(self, idx: int):
        dataset = xr.load_dataset(self.files[idx])
        inp = self.get_features(dataset)
        labels = torch.from_numpy(dataset['LABELS'].to_numpy())
        lead_times = torch.Tensor([0.0]).to(dtype=inp.dtype)
        return inp, labels, lead_times, self.in_variables
