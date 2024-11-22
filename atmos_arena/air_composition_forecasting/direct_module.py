from typing import Any, Union

import torch
from lightning import LightningModule
from atmos_arena.climax_arch import ClimaX
from atmos_arena.stormer_arch import Stormer
from atmos_arena.unet_arch import Unet
from atmos_arena.atmos_utils.lr_scheduler import LinearWarmupCosineAnnealingLR
from atmos_arena.atmos_utils.metrics import (
    lat_weighted_mse,
    lat_weighted_mse_val,
    lat_weighted_mae
)
from atmos_arena.atmos_utils.pos_embed import interpolate_pos_embed


class CAMSDirectForecastingModule(LightningModule):
    """Lightning module for climate projection.

    Args:
        net: ClimaX or Stormer model.
        pretrained_path (str, optional): Path to pre-trained checkpoint.
        lr (float, optional): Learning rate.
        beta_1 (float, optional): Beta 1 for AdamW.
        beta_2 (float, optional): Beta 2 for AdamW.
        weight_decay (float, optional): Weight decay for AdamW.
        warmup_epochs (int, optional): Number of warmup epochs.
        max_epochs (int, optional): Number of total epochs.
        warmup_start_lr (float, optional): Starting learning rate for warmup.
        eta_min (float, optional): Minimum learning rate.
    """
    def __init__(
        self,
        net: Union[ClimaX, Stormer, Unet],
        pretrained_path: str = "",
        lr: float = 5e-4,
        beta_1: float = 0.9,
        beta_2: float = 0.99,
        weight_decay: float = 1e-5,
        warmup_epochs: int = 5,
        max_epochs: int = 50,
        warmup_start_lr: float = 1e-8,
        eta_min: float = 1e-8,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False, ignore=["net"])
        self.net = net
        if len(pretrained_path) > 0:
            self.load_pretrained_weights(pretrained_path)

    def load_pretrained_weights(self, pretrained_path):
        if pretrained_path.startswith("http"):
            checkpoint = torch.hub.load_state_dict_from_url(pretrained_path, map_location=torch.device("cpu"))
        else:
            checkpoint = torch.load(pretrained_path, map_location=torch.device("cpu"))

        print("Loading pre-trained checkpoint from: %s" % pretrained_path)
        checkpoint_model = checkpoint["state_dict"]
        # interpolate positional embedding
        interpolate_pos_embed(self.net, checkpoint_model, new_size=self.net.in_img_size)

        state_dict = self.state_dict()
                
        for k in list(checkpoint_model.keys()):
            if 'token_embeds' in k or 'head' in k: # initialize embedding from scratch
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]
                continue
            
            if k not in state_dict.keys() or checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        # load pre-trained model
        msg = self.load_state_dict(checkpoint_model, strict=False)
        print(msg)

    def set_lat_lon(self, lat, lon):
        self.lat = lat
        self.lon = lon
        
    def set_lead_time(self, lead_time):
        self.lead_time = lead_time

    def training_step(self, batch: Any, batch_idx: int):
        x, y, lead_times, in_variables, out_variables = batch
        pred = self.net(x, lead_times, in_variables, out_variables)
        loss_dict = lat_weighted_mse(pred, y, out_variables, self.lat)
        for var in loss_dict.keys():
            self.log(
                "train/" + var,
                loss_dict[var],
                on_step=True,
                on_epoch=False,
                prog_bar=True,
                batch_size=x.size(0),
            )
        loss = loss_dict['loss']

        return loss

    def validation_step(self, batch: Any, batch_idx: int):
        x, y, lead_times, in_variables, out_variables = batch
        pred = self.net(x, lead_times, in_variables, out_variables)
        transform = lambda x: self.trainer.datamodule.denormalize(x, out_variables)
        metrics = [lat_weighted_mse_val, lat_weighted_mae]
        all_loss_dicts = [
            m(pred, y, transform, vars=out_variables, lat=self.lat, clim=None, log_postfix="") for m in metrics
        ]

        # combine loss dicts
        loss_dict = {}
        for d in all_loss_dicts:
            for k in d.keys():
                loss_dict[k] = d[k]

        for var in loss_dict.keys():
            self.log(
                "val/" + var,
                loss_dict[var],
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                sync_dist=True,
                batch_size=x.size(0),
            )
        return loss_dict

    def test_step(self, batch: Any, batch_idx: int):
        x, y, lead_times, variables, out_variables = batch
        pred = self.net(x, lead_times, variables, out_variables)
        transform = lambda x: self.trainer.datamodule.denormalize(x, out_variables)
        metrics=[lat_weighted_mse_val, lat_weighted_mae]
        all_loss_dicts = [
            m(pred, y, transform, vars=out_variables, lat=self.lat, clim=None, log_postfix="") for m in metrics
        ]

        # combine loss dicts
        loss_dict = {}
        for d in all_loss_dicts:
            for k in d.keys():
                loss_dict[k] = d[k]

        for var in loss_dict.keys():
            self.log(
                "test/" + var,
                loss_dict[var],
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                sync_dist=True,
            )
        return loss_dict

    def configure_optimizers(self):
        decay = []
        no_decay = []
        for name, m in self.named_parameters():
            if "var_embed" in name or "channel_emb" in name or "pos_embed" in name or "time_pos_embed" in name:
                no_decay.append(m)
            else:
                decay.append(m)

        optimizer = torch.optim.AdamW(
            [
                {
                    "params": decay,
                    "lr": self.hparams.lr,
                    "betas": (self.hparams.beta_1, self.hparams.beta_2),
                    "weight_decay": self.hparams.weight_decay,
                },
                {
                    "params": no_decay,
                    "lr": self.hparams.lr,
                    "betas": (self.hparams.beta_1, self.hparams.beta_2),
                    "weight_decay": 0
                },
            ]
        )
        
        n_steps_per_machine = len(self.trainer.datamodule.train_dataloader())
        n_steps = int(n_steps_per_machine / (self.trainer.num_devices * self.trainer.num_nodes))
        lr_scheduler = LinearWarmupCosineAnnealingLR(
            optimizer,
            self.hparams.warmup_epochs * n_steps,
            self.hparams.max_epochs * n_steps,
            self.hparams.warmup_start_lr,
            self.hparams.eta_min,
        )
        scheduler = {"scheduler": lr_scheduler, "interval": "step", "frequency": 1}
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
