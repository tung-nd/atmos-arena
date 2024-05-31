from typing import Any, Union

import numpy as np
import torch
from lightning import LightningModule
from extreme_detection.climax_climatenet_arch import ClimaXClimateNet
from stormer_arch import Stormer
from atmos_utils.lr_scheduler import LinearWarmupCosineAnnealingLR
from atmos_utils.pos_embed import interpolate_pos_embed
from extreme_detection.losses import loss_function
from extreme_detection.metrics import get_cm, get_confusion_metrics, get_dice_perClass, get_iou_perClass
from torchvision.transforms import transforms


class ClimateNetModule(LightningModule):
    """Lightning module for extreme detection.

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
        net: Union[ClimaXClimateNet],
        loss_type: str = 'jaccard',
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
        self.validation_step_cms = []

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
                
        for k in list(checkpoint_model.keys()):
            if k not in state_dict.keys() or checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        # load pre-trained model
        msg = self.load_state_dict(checkpoint_model, strict=False)
        print(msg)

    def training_step(self, batch: Any, batch_idx: int):
        x, y, lead_times, variables = batch # lead_times is set to 0 for this task
        pred = self.net(x, lead_times, variables)
        loss = loss_function(pred, y, self.hparams.loss_type)
        self.log(
            "train/loss",
            loss.item(),
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            batch_size=x.size(0),
        )

        return loss
    
    def validation_step(self, batch: Any, batch_idx: int):
        x, y, lead_times, variables = batch
        pred = self.net(x, lead_times, variables)
        loss = loss_function(pred, y, self.hparams.loss_type)
        self.log(
            "val/loss",
            loss.item(),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=x.size(0),
            sync_dist=True
        )
        
        pred = torch.softmax(pred, 1)
        pred = torch.max(pred, 1)[1]
        cm = get_cm(pred, y)
        self.validation_step_cms.append(cm)
        return cm
    
    def on_validation_epoch_end(self):
        cm = np.stack(self.validation_step_cms, axis=0).sum(0)
        precision, recall, specificity, sensitivity = get_confusion_metrics(cm)
        ious = get_iou_perClass(cm)
        dices = get_dice_perClass(cm)
        class_names = ['BG', 'TC', 'AR']
        metrics = {}
        for i in range(3):
            metrics[f"precision_{class_names[i]}"] = precision[i]
            metrics[f"recall_{class_names[i]}"] = recall[i]
            metrics[f"specificity_{class_names[i]}"] = specificity[i]
            metrics[f"sensitivity_{class_names[i]}"] = sensitivity[i]
            metrics[f"iou_{class_names[i]}"] = ious[i]
            metrics[f"dice_{class_names[i]}"] = dices[i]
        metrics = {f'val/{k}': v for k, v in metrics.items()}
        self.log_dict(
            metrics,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

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