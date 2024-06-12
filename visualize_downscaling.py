import pickle
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import os
import yaml
import torch
from torchvision.transforms import transforms

from climax_arch import ClimaX
from stormer_arch import Stormer
from downscaling.dataset import ERA5DownscalingDataset

import argparse

VAR_MAP = {
    '2m_temperature': 'T2m',
    '10m_u_component_of_wind': 'U10m',
    '10m_v_component_of_wind': 'V10m',
    'mean_sea_level_pressure': 'MSLP',
    'geopotential_500': 'Z500',
    'temperature_850': 'T850',
    'temperature_500': 'T500',
    'specific_humidity_500': 'Q500',
    'specific_humidity_700': 'Q700',
    'u_component_of_wind_850': 'U850',
    'v_component_of_wind_850': 'V850',
}

def get_best_checkpoint(dir):
    ckpt_paths = os.listdir(os.path.join(dir, 'checkpoints'))
    for ckpt_path in ckpt_paths:
        if 'epoch' in ckpt_path:
            return os.path.join(dir, 'checkpoints/', ckpt_path)
        
def get_normalize(root_dir, variables):
    normalize_mean = dict(np.load(os.path.join(root_dir, "normalize_mean.npz")))
    normalize_mean = np.concatenate([normalize_mean[v] for v in variables], axis=0)
    normalize_std = dict(np.load(os.path.join(root_dir, "normalize_std.npz")))
    normalize_std = np.concatenate([normalize_std[v] for v in variables], axis=0)
    return transforms.Normalize(normalize_mean, normalize_std)

def get_denormalize(normalization):
    mean, std = normalization.mean, normalization.std
    std_denorm = 1 / std
    mean_denorm = -mean * std_denorm
    return transforms.Normalize(mean_denorm, std_denorm)

def plot(model, dataset, out_denormalize):
    # always use Jan 1st 2020
    x, y, _, lead_time_tensor, in_vars, out_vars = dataset[0]
    x, y = x.unsqueeze(0), y.unsqueeze(0)
    pred = model(torch.nn.functional.interpolate(x, size=y.shape[-2:], mode='bilinear').cuda(), lead_time_tensor.cuda(), in_vars, out_vars).cpu()
    
    truths = out_denormalize(y).squeeze().numpy()
    forecasts = out_denormalize(pred).squeeze().detach().numpy()
    ics = x.squeeze().numpy()
    
    out_vars = [v for v in out_vars if v != 'mean_sea_level_pressure']
    
    fig, axs = plt.subplots(len(out_vars), 4, figsize=(5*len(out_vars), 19))
    column_titles = ['Input (00:00:00 Jan 1st 2020)', f'Ground truth', f'Prediction', 'Bias']
    
    for i, var in enumerate(out_vars):
        in_var_id = in_vars.index(var)
        ic = ics[in_var_id]
        forecast_var = forecasts[i]
        truth_var = truths[i]
        bias = forecast_var - truth_var
        
        data_plot = [ic, truth_var, forecast_var, bias]
        data_plot = [np.flip(d, axis=0) for d in data_plot]
        
        if var == 'specific_humidity_700':
            data_plot = [ic*1000, truth_var*1000, forecast_var*1000, bias*1000]
        
        for j in range(4):
            ax = axs[i, j]
            im = ax.imshow(data_plot[j], cmap=plt.cm.RdBu)
            
            # Hide x, y ticks
            ax.set_xticks([])
            ax.set_yticks([])
            
            # Set column titles for the first row
            if i == 0:
                ax.set_title(column_titles[j], fontsize=18)
            
            # Set the y-axis label for the first column
            if j == 0:
                ax.set_ylabel(VAR_MAP[var], fontsize=18)
            
            # Add a colorbar for each subplot
            # fig.colorbar(im, ax=ax, fontsize=15)
            
            cbar = fig.colorbar(im, ax=ax)
            cbar.ax.tick_params(labelsize=12)
    
    if isinstance(model, ClimaX):
        model_name = 'ClimaX'
    elif isinstance(model, Stormer):
        model_name = 'Stormer'
        
    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.savefig(f'{model_name}_downscaling.pdf', bbox_inches='tight')
    plt.close()
    # Show plot
    # plt.show()

def main(model_path):
    ckpt_path = get_best_checkpoint(model_path)
    config_path = os.path.join(model_path, 'config.yaml')
    
    # Read config file
    with open(config_path, 'r') as stream:
        config = yaml.safe_load(stream)
    
    if 'climax' in model_path:
        model_cls = ClimaX
    elif 'stormer' in model_path:
        model_cls = Stormer
    model = model_cls(**config['model']['net']['init_args'])
    state_dict = torch.load(ckpt_path, map_location='cpu')['state_dict']
    state_dict = {k[4:]: v for k, v in state_dict.items()}
    msg = model.load_state_dict(state_dict)
    print(msg)
    model.cuda()
    
    in_root_dir = '/eagle/MDClimSim/tungnd/data/wb2/5.625deg_1_step_6hr_h5df/'
    out_root_dir = '/eagle/MDClimSim/tungnd/data/wb2/1.40625deg_from_full_res_1_step_6hr_h5df'
    clim_path = config['data']['clim_path']
    in_variables = config['data']['in_variables']
    out_variables = config['data']['out_variables']
    in_transform = get_normalize(in_root_dir, in_variables)
    out_transform = get_normalize(out_root_dir, out_variables)
    out_denormalize = get_denormalize(out_transform)
    
    dataset = ERA5DownscalingDataset(
        in_root_dir=os.path.join(in_root_dir, 'test'),
        out_root_dir=os.path.join(out_root_dir, 'test'),
        clim_path=clim_path,
        in_variables=in_variables,
        out_variables=out_variables,
        in_transform=in_transform,
        out_transform=out_transform,
    )
    
    plot(model, dataset, out_denormalize)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run the model with specified path')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model directory')
    args = parser.parse_args()
    main(args.model_path)
