# Standard library
import os
from typing import Optional, Sequence, Tuple

# Third party
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from lightning import LightningDataModule

# Local application
from atmos_arena.chemistry_downscaling.dataset import GEOSCFDownscalingDataset


def collate_fn(
    batch,
) -> Tuple[torch.tensor, torch.tensor, Sequence[str], Sequence[str]]:
    inp = torch.stack([batch[i][0] for i in range(len(batch))]) # B, C, H, W
    out = torch.stack([batch[i][1] for i in range(len(batch))]) # B, C, H, W
    lead_times = torch.cat([batch[i][2] for i in range(len(batch))])
    variables = batch[0][3]
    return inp, out, lead_times, variables


class GEOSCFDownscalingDataModule(LightningDataModule):
    def __init__(
        self,
        root_dir,
        train_year_strs,
        val_year_strs,
        test_year_strs,
        variable,
        downscale_ratio,
        batch_size=1,
        num_workers=0,
        pin_memory=False,
    ):
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    def get_lat_lon(self):
        lat = np.load(os.path.join(self.hparams.root_dir, "lat.npy"))
        lon = np.load(os.path.join(self.hparams.root_dir, "lon.npy"))
        return lat, lon
    
    def denormalize(self, x):
        return self.data_train.denormalize(x)

    def setup(self, stage: Optional[str] = None):
        # load datasets only if they're not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            self.data_train = GEOSCFDownscalingDataset(
                root_dir=self.hparams.root_dir,
                year_strs=self.hparams.train_year_strs,
                variable=self.hparams.variable,
                downscale_ratio=self.hparams.downscale_ratio,
            )
            
            self.data_val = GEOSCFDownscalingDataset(
                root_dir=self.hparams.root_dir,
                year_strs=self.hparams.val_year_strs,
                variable=self.hparams.variable,
                downscale_ratio=self.hparams.downscale_ratio,
            )

            self.data_test = GEOSCFDownscalingDataset(
                root_dir=self.hparams.root_dir,
                year_strs=self.hparams.test_year_strs,
                variable=self.hparams.variable,
                downscale_ratio=self.hparams.downscale_ratio,
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