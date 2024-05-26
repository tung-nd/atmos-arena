# Standard library
import os
from typing import Optional, Sequence, Tuple

# Third party
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import transforms
from lightning import LightningDataModule
from torch.utils.data import Dataset
from datasets import load_dataset


class WIPDataset(Dataset):
    def __init__(
        self,
        hf_dataset,
        in_variables,
    ):
        super().__init__()
        
        self.hf_dataset = hf_dataset
        self.in_variables = in_variables
        
    def __len__(self):
        return len(self.hf_dataset)
    
    def __getitem__(self, index):
        input = np.array(self.hf_dataset[index]['input']).astype(np.float32)
        target = self.hf_dataset[index]['target']
        lead_times = torch.Tensor([0.0]).to(dtype=torch.float32)
        return torch.from_numpy(input), torch.Tensor([target]), lead_times, self.in_variables


def collate_fn(
    batch,
) -> Tuple[torch.tensor, torch.tensor, Sequence[str], Sequence[str]]:
    inp = torch.stack([batch[i][0] for i in range(len(batch))]) # B, C, H, W
    out = torch.stack([batch[i][1] for i in range(len(batch))]) # B, C, H, W
    lead_times = torch.cat([batch[i][2] for i in range(len(batch))])
    in_variables = batch[0][3]
    return inp, out, lead_times, in_variables


class WIPDataModule(LightningDataModule):
    def __init__(
        self,
        hf_path,
        in_variables,
        batch_size=1,
        num_workers=0,
        pin_memory=False,
    ):
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    def setup(self, stage: Optional[str] = None):
        # load datasets only if they're not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            dataset = load_dataset(self.hparams.hf_path)
            
            self.data_train = WIPDataset(
                hf_dataset=dataset['train'],
                in_variables=self.hparams.in_variables,
            )
            
            self.data_val = WIPDataset(
                hf_dataset=dataset['validation'],
                in_variables=self.hparams.in_variables,
            )

            self.data_test = WIPDataset(
                hf_dataset=dataset['test'],
                in_variables=self.hparams.in_variables,
            )

    def train_dataloader(self):
        return DataLoader(
            self.data_train,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.data_val,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=collate_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            self.data_test,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=collate_fn,
        )

# datamodule = WIPDataModule(
#     hf_path='sungduk/wip_cmip6_v2.2',
#     in_variables=['surface_air_temperature', 'surface_specific_humidity', 'precipitation'],
#     batch_size=4,
#     num_workers=1,
#     pin_memory=False,
# )
# datamodule.setup()
# train_loader = datamodule.train_dataloader()
# x, y, in_variables = next(iter(train_loader))
# print(x.shape, y.shape, in_variables)