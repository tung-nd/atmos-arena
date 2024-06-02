# Standard library
import os
from typing import Optional, Sequence, Tuple

# Third party
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import transforms
from lightning import LightningDataModule

# Local application
from s2s.dataset import ERA5WindowDataset


def collate_fn(
    batch,
) -> Tuple[torch.tensor, torch.tensor, Sequence[str], Sequence[str]]:
    inp = torch.stack([batch[i][0] for i in range(len(batch))]) # B, C, H, W
    out = torch.stack([batch[i][1] for i in range(len(batch))]) # B, C, H, W
    lead_times = torch.cat([batch[i][2] for i in range(len(batch))])
    in_variables = batch[0][3]
    out_variables = batch[0][4]
    return inp, out, lead_times, in_variables, out_variables


class WindowForecastingDataModule(LightningDataModule):
    def __init__(
        self,
        root_dir,
        in_variables,
        out_variables,
        lead_time,
        data_freq=6,
        batch_size=1,
        num_workers=0,
        pin_memory=False,
    ):
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.in_transforms = self.get_normalize(root_dir, in_variables)
        self.out_transforms = self.get_normalize(root_dir, out_variables)

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None
        
    def get_normalize(self, root_dir, variables):
        normalize_mean = dict(np.load(os.path.join(root_dir, "normalize_mean.npz")))
        normalize_mean = np.concatenate([normalize_mean[v] for v in variables], axis=0)
        normalize_std = dict(np.load(os.path.join(root_dir, "normalize_std.npz")))
        normalize_std = np.concatenate([normalize_std[v] for v in variables], axis=0)
        return transforms.Normalize(normalize_mean, normalize_std)

    def get_lat_lon(self):
        lat = np.load(os.path.join(self.hparams.root_dir, "lat.npy"))
        lon = np.load(os.path.join(self.hparams.root_dir, "lon.npy"))
        return lat, lon

    def setup(self, stage: Optional[str] = None):
        # load datasets only if they're not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            self.data_train = ERA5WindowDataset(
                root_dir=os.path.join(self.hparams.root_dir, 'train'),
                in_variables=self.hparams.in_variables,
                out_variables=self.hparams.out_variables,
                in_transform=self.in_transforms,
                out_transform=self.out_transforms,
                lead_time=self.hparams.lead_time,
                data_freq=self.hparams.data_freq,
            )
            
            if os.path.exists(os.path.join(self.hparams.root_dir, 'val')):
                self.data_val = ERA5WindowDataset(
                    root_dir=os.path.join(self.hparams.root_dir, 'val'),
                    in_variables=self.hparams.in_variables,
                    out_variables=self.hparams.out_variables,
                    in_transform=self.in_transforms,
                    out_transform=self.out_transforms,
                    lead_time=self.hparams.lead_time,
                    data_freq=self.hparams.data_freq, 
                )

            if os.path.exists(os.path.join(self.hparams.root_dir, 'test')):
                self.data_test = ERA5WindowDataset(
                    root_dir=os.path.join(self.hparams.root_dir, 'test'),
                    in_variables=self.hparams.in_variables,
                    out_variables=self.hparams.out_variables,
                    in_transform=self.in_transforms,
                    out_transform=self.out_transforms,
                    lead_time=self.hparams.lead_time,
                    data_freq=self.hparams.data_freq,
                )

    def train_dataloader(self):
        return DataLoader(
            self.data_train,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=collate_fn
        )

    def val_dataloader(self):
        if self.data_val is not None:
            return DataLoader(
                self.data_val,
                batch_size=self.hparams.batch_size,
                shuffle=False,
                drop_last=False,
                num_workers=self.hparams.num_workers,
                pin_memory=self.hparams.pin_memory,
                collate_fn=collate_fn
            )

    def test_dataloader(self):
        if self.data_test is not None:
            return DataLoader(
                self.data_test,
                batch_size=self.hparams.batch_size,
                shuffle=False,
                drop_last=False,
                num_workers=self.hparams.num_workers,
                pin_memory=self.hparams.pin_memory,
                collate_fn=collate_fn
            )

# datamodule = OneStepDataModule(
#     '/eagle/MDClimSim/tungnd/data/wb1/1.40625deg_1_step_6hr',
#     variables=[
#         "land_sea_mask",
#         "orography",
#         "lattitude",
#         "2m_temperature",
#         "10m_u_component_of_wind",
#         "10m_v_component_of_wind",
#         "toa_incident_solar_radiation",
#         "total_cloud_cover",
#         "geopotential_500",
#         "temperature_850"
#     ],
#     batch_size=128,
#     num_workers=1,
#     pin_memory=False
# )
# datamodule.setup()
# for batch in datamodule.train_dataloader():
#     inp, out, vars, out_vars = batch
#     print (inp.shape)
#     print (out.shape)
#     print (vars)
#     print (out_vars)
#     break