# Standard library
import os
from typing import Optional, Sequence, Tuple

# Third party
import torch
import numpy as np
from glob import glob
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import transforms
from lightning import LightningDataModule

# Local application
from monthly_infilling.dataset import ERA5MonthlyInfillingDataset


def collate_fn_train(
    batch,
) -> Tuple[torch.tensor, torch.tensor, Sequence[str], Sequence[str]]:
    inp = torch.stack([batch[i][0] for i in range(len(batch))]) # B, C, H, W
    out = torch.stack([batch[i][1] for i in range(len(batch))]) # B, C, H, W
    lead_times = torch.cat([batch[i][2] for i in range(len(batch))])
    mask = torch.stack([batch[i][3] for i in range(len(batch))]) # B, H, W
    in_variables = batch[0][4]
    out_variables = batch[0][5]
    return inp, out, lead_times, mask, in_variables, out_variables


def collate_fn_val(
    batch,
) -> Tuple[torch.tensor, torch.tensor, Sequence[str], Sequence[str]]:
    # each batch[i][0] is a dictionary of scalar keys and tensor values
    # for each key, we stack the tensors along the batch dimension
    inp_dict = {k: torch.stack([batch[i][0][k] for i in range(len(batch))]) for k in batch[0][0].keys()}
    out = torch.stack([batch[i][1] for i in range(len(batch))]) # B, C, H, W
    lead_times = torch.cat([batch[i][2] for i in range(len(batch))])
    mask_dict = {k: torch.stack([batch[i][3][k] for i in range(len(batch))]) for k in batch[0][3].keys()} # B, H, W
    in_variables = batch[0][4]
    out_variables = batch[0][5]
    return inp_dict, out, lead_times, mask_dict, in_variables, out_variables


class MonthlyInfillingDataModule(LightningDataModule):
    def __init__(
        self,
        root_dir,
        variable,
        training_mask_ratio_min,
        training_mask_ratio_max,
        eval_mask_ratios,
        train_years=range(1979, 2019),
        val_years=range(2019, 2020),
        test_years=range(2020, 2021),
        h5_data_dir='/eagle/MDClimSim/tungnd/data/wb2/1.40625deg_from_full_res_1_step_6hr_h5df',
        batch_size=1,
        num_workers=0,
        pin_memory=False,
    ):
        super().__init__()

        self.save_hyperparameters(logger=False)

        root_dir = os.path.join(root_dir, variable)
        all_files = sorted(glob(os.path.join(root_dir, "*.nc")))   
        
        # train dataset
        train_files = [f for f in all_files if any([str(y) in f for y in train_years])]
        self.data_train = ERA5MonthlyInfillingDataset(
            train_files,
            variable=variable,
        )
        mean_transform, std_transform = self.data_train.get_mean_std()
        self.data_train.set_transform(mean_transform, std_transform)
        self.data_train.set_mask_ratio_range((training_mask_ratio_min, training_mask_ratio_max))
        self.data_train.set_predefined_mask_dict(None)
        
        # val dataset
        val_files = [f for f in all_files if any([str(y) in f for y in val_years])]
        self.data_val = ERA5MonthlyInfillingDataset(
            val_files,
            variable=variable,
        )
        self.data_val.set_transform(mean_transform, std_transform)
        n_val = len(self.data_val)
        h, w = self.data_train[0][0].shape[1:]
        val_mask_dict = {}
        for ratio in eval_mask_ratios:
            val_mask_dict[ratio] = np.random.choice([0, 1], size=(n_val, h, w), p=[ratio, 1 - ratio])
        self.data_val.set_predefined_mask_dict(val_mask_dict)
        
        # test dataset
        test_files = [f for f in all_files if any([str(y) in f for y in test_years])]
        self.data_test = ERA5MonthlyInfillingDataset(
            test_files,
            variable=variable,
        )
        self.data_test.set_transform(mean_transform, std_transform)
        n_test = len(self.data_test)
        test_mask_dict = {}
        for ratio in eval_mask_ratios:
            test_mask_dict[ratio] = np.random.choice([0, 1], size=(n_test, h, w), p=[ratio, 1 - ratio])
        self.data_test.set_predefined_mask_dict(test_mask_dict)

    def get_lat_lon(self):
        lat = np.load(os.path.join(self.hparams.h5_data_dir, "lat.npy"))
        lon = np.load(os.path.join(self.hparams.h5_data_dir, "lon.npy"))
        return lat, lon

    def train_dataloader(self):
        return DataLoader(
            self.data_train,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=collate_fn_train
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
                collate_fn=collate_fn_val
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
                collate_fn=collate_fn_val
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