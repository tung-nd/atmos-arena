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
from atmos_arena.downscaling.dataset import ERA5DownscalingDataset


def collate_fn(
    batch,
) -> Tuple[torch.tensor, torch.tensor, Sequence[str], Sequence[str]]:
    inp = torch.stack([batch[i][0] for i in range(len(batch))]) # B, C, H, W
    out = torch.stack([batch[i][1] for i in range(len(batch))]) # B, C, H, W
    clim = torch.stack([batch[i][2] for i in range(len(batch))]) # B, C, H, W
    lead_times = torch.cat([batch[i][3] for i in range(len(batch))])
    in_variables = batch[0][4]
    out_variables = batch[0][5]
    return inp, out, clim, lead_times, in_variables, out_variables


class DownscalingDataModule(LightningDataModule):
    def __init__(
        self,
        in_root_dir,
        out_root_dir,
        in_variables,
        out_variables,
        clim_path,
        batch_size=1,
        num_workers=0,
        pin_memory=False,
    ):
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.in_transforms = self.get_normalize(in_root_dir, in_variables)
        self.out_transforms = self.get_normalize(out_root_dir, out_variables)

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
        lat = np.load(os.path.join(self.hparams.out_root_dir, "lat.npy"))
        lon = np.load(os.path.join(self.hparams.out_root_dir, "lon.npy"))
        return lat, lon

    def setup(self, stage: Optional[str] = None):
        # load datasets only if they're not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            self.data_train = ERA5DownscalingDataset(
                in_root_dir=os.path.join(self.hparams.in_root_dir, 'train'),
                out_root_dir=os.path.join(self.hparams.out_root_dir, 'train'),
                clim_path=self.hparams.clim_path,
                in_variables=self.hparams.in_variables,
                out_variables=self.hparams.out_variables,
                in_transform=self.in_transforms,
                out_transform=self.out_transforms,
            )
            
            if os.path.exists(os.path.join(self.hparams.in_root_dir, 'val')):
                self.data_val = ERA5DownscalingDataset(
                    in_root_dir=os.path.join(self.hparams.in_root_dir, 'val'),
                    out_root_dir=os.path.join(self.hparams.out_root_dir, 'val'),
                    clim_path=self.hparams.clim_path,
                    in_variables=self.hparams.in_variables,
                    out_variables=self.hparams.out_variables,
                    in_transform=self.in_transforms,
                    out_transform=self.out_transforms,
                )

            if os.path.exists(os.path.join(self.hparams.in_root_dir, 'test')):
                self.data_test = ERA5DownscalingDataset(
                    in_root_dir=os.path.join(self.hparams.in_root_dir, 'test'),
                    out_root_dir=os.path.join(self.hparams.out_root_dir, 'test'),
                    clim_path=self.hparams.clim_path,
                    in_variables=self.hparams.in_variables,
                    out_variables=self.hparams.out_variables,
                    in_transform=self.in_transforms,
                    out_transform=self.out_transforms,
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
