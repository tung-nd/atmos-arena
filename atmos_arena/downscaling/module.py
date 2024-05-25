from typing import Any, Union

import numpy as np
import torch
from lightning import LightningModule
from climax_arch import ClimaX
from stormer_arch import Stormer
from atmos_utils.lr_scheduler import LinearWarmupCosineAnnealingLR
from atmos_utils.metrics import (
    mse,
    lat_weighted_mse_val,
    lat_weighted_rmse,
    lat_weighted_mean_bias,
    pearson
)
from atmos_utils.pos_embed import interpolate_pos_embed
from torchvision.transforms import transforms


class DownscalingModule(LightningModule):
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
        net: Union[ClimaX, Stormer],
        pretrained_path: str = "",
        lr: float = 5e-4,
        beta_1: float = 0.9,
        beta_2: float = 0.99,
        weight_decay: float = 1e-5,
        warmup_epochs: int = 60,
        max_epochs: int = 600,
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
        
        # for k in list(checkpoint_model.keys()):
            # if 'token_embeds' in k or 'head' in k: # initialize embedding from scratch
            #     print(f"Removing key {k} from pretrained checkpoint")
            #     del checkpoint_model[k]
            #     continue
                
        for k in list(checkpoint_model.keys()):
            if k not in state_dict.keys() or checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        # load pre-trained model
        msg = self.load_state_dict(checkpoint_model, strict=False)
        print(msg)

    def set_denormalization(self, mean, std):
        self.denormalization = transforms.Normalize(mean, std)

    def set_lat_lon(self, lat, lon):
        self.lat = lat
        self.lon = lon

    def training_step(self, batch: Any, batch_idx: int):
        x, y, lead_times, in_variables, out_variables = batch # lead_times is set to 0 for this task
        # interpolate x to match y shape
        x = torch.nn.functional.interpolate(x, size=y.shape[-2:], mode='bilinear')
        pred = self.net(x, lead_times, in_variables, out_variables)
        loss_dict = mse(pred, y, out_variables, self.lat)
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
        # interpolate x to match y shape
        x = torch.nn.functional.interpolate(x, size=y.shape[-2:], mode='bilinear')
        pred = self.net(x, lead_times, in_variables, out_variables)
        metrics = [lat_weighted_mse_val, lat_weighted_rmse]
        all_loss_dicts = [
            m(pred, y, self.denormalization, vars=out_variables, lat=self.lat, clim=None, log_postfix="") for m in metrics
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
        # interpolate x to match y shape
        x = torch.nn.functional.interpolate(x, size=y.shape[-2:], mode='bilinear')
        pred = self.net(x, lead_times, variables, out_variables)
        metrics=[lat_weighted_mse_val, lat_weighted_rmse, lat_weighted_mean_bias, pearson]
        all_loss_dicts = [
            m(pred, y, self.denormalization, vars=out_variables, lat=self.lat, clim=None, log_postfix="") for m in metrics
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
    
# model = StormerClimateBench(
#     in_img_size=[32, 64],
#     in_variables=['CO2', 'SO2', 'CH4', 'BC'],
#     out_variables=['tas'],
#     time_history=10,
#     patch_size=2,
#     embed_norm=True,
#     hidden_size=1024,
#     depth=24,
#     num_heads=16,
#     mlp_ratio=4.0,
#     freeze_encoder=True
# )
# module = ClimateBenchModule(net=model, pretrained_path='/eagle/MDClimSim/tungnd/stormer/models/6_12_24_climax_large_2_True_delta_8/checkpoints/epoch_015.ckpt')