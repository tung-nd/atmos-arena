import os
import torch
import numpy as np
from omegaconf import OmegaConf
from tqdm import tqdm
from atmos_arena.climax_arch import ClimaX
from atmos_arena.stormer_arch import Stormer
from atmos_arena.unet_arch import Unet
from atmos_arena.downscaling.datamodule import DownscalingDataModule


def get_best_checkpoint(dir):
    ckpt_paths = os.listdir(os.path.join(dir, 'checkpoints'))
    for ckpt_path in ckpt_paths:
        if 'epoch' in ckpt_path:
            return os.path.join(dir, 'checkpoints/', ckpt_path)


# climax_dir = '/eagle/MDClimSim/tungnd/atmost-arena/downscaling/downscaling_climax_finetune_backbone_32bs_5e-5_lr/'
# config_path = os.path.join(climax_dir, 'config.yaml')
# config = OmegaConf.load(config_path)
# model = ClimaX(**config['model']['net']['init_args'])
# ckpt_path = get_best_checkpoint(climax_dir)
# state_dict = torch.load(ckpt_path, map_location='cpu')['state_dict']
# state_dict = {k.replace('net.', ''): v for k, v in state_dict.items()}
# msg = model.load_state_dict(state_dict)
# print(f'Loaded checkpoint from {ckpt_path}', msg)
# model.eval()
# model = model.cuda()
# var_ids = [0, 3, 4]
# pred_path = '/eagle/MDClimSim/tungnd/atmost-arena/climax_preds.npy'
# target_path = '/eagle/MDClimSim/tungnd/atmost-arena/targets.npy'

# stormer_dir = '/eagle/MDClimSim/tungnd/atmost-arena/downscaling/downscaling_stormer_finetune_backbone_64bs_5e-5_lr/'
# config_path = os.path.join(stormer_dir, 'config.yaml')
# config = OmegaConf.load(config_path)
# model = Stormer(**config['model']['net']['init_args'])
# ckpt_path = get_best_checkpoint(stormer_dir)
# state_dict = torch.load(ckpt_path, map_location='cpu')['state_dict']
# state_dict = {k.replace('net.', ''): v for k, v in state_dict.items()}
# msg = model.load_state_dict(state_dict)
# print(f'Loaded checkpoint from {ckpt_path}', msg)
# model.eval()
# model = model.cuda()
# var_ids = [0, 4, 5]
# pred_path = '/eagle/MDClimSim/tungnd/atmost-arena/stormer_preds.npy'
# target_path = '/eagle/MDClimSim/tungnd/atmost-arena/targets.npy'

unet_dir = '/eagle/MDClimSim/tungnd/atmost-arena/downscaling/downscaling_unet_stormer_vars_128bs_5e-4_lr/'
config_path = os.path.join(unet_dir, 'config.yaml')
config = OmegaConf.load(config_path)
model = Unet(**config['model']['net']['init_args'])
ckpt_path = get_best_checkpoint(unet_dir)
state_dict = torch.load(ckpt_path, map_location='cpu')['state_dict']
state_dict = {k.replace('net.', ''): v for k, v in state_dict.items()}
msg = model.load_state_dict(state_dict)
print(f'Loaded checkpoint from {ckpt_path}', msg)
model.eval()
model = model.cuda()
var_ids = [0, 4, 5]
pred_path = '/eagle/MDClimSim/tungnd/atmost-arena/unet_preds.npy'
target_path = '/eagle/MDClimSim/tungnd/atmost-arena/targets.npy'

datamodule = DownscalingDataModule(**config['data'])
datamodule.setup()
test_loader = datamodule.test_dataloader()
num_batches = len(test_loader)
all_preds = []
all_targets = []
for i, batch in tqdm(enumerate(test_loader), total=num_batches):
    x, y, _, lead_times, in_variables, out_variables = batch
    x = x.cuda()
    y = y.cuda()
    lead_times = lead_times.cuda()
    x = torch.nn.functional.interpolate(x, size=y.shape[-2:], mode='bilinear')
    with torch.no_grad():
        pred = model(x, lead_times, in_variables, out_variables)
    y = y[:, var_ids]
    pred = pred[:, var_ids]
    all_preds.append(pred.cpu().numpy())
    all_targets.append(y.cpu().numpy())

all_preds = np.concatenate(all_preds, axis=0)
all_targets = np.concatenate(all_targets, axis=0)
np.save(pred_path, all_preds)
np.save(target_path, all_targets)