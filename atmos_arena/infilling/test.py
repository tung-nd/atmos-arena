import xarray as xr
import numpy as np
import os 
import torch 
import h5py
import numpy as np 
import random 
from torchvision.transforms import transforms

from climax_arch import ClimaX
from stormer_arch import Stormer
from unet_arch import Unet

PATHS = {
    'temp_regridded' : '/localhome/data/datasets/climate/berkeley_land_and_ocean_monthly_temperature_128_256.nc', 
    'temp' : '/localhome/data/datasets/climate/berkeley_land_and_ocean_monthly_temperature.nc', 
    'model' : '/localhome/tungnd/atmos_arena/infilling/' # Loop through each folder, cd checkpoints, run epoch_x.ckpt
}


"""

Helper Functions 

"""


def inspect_temp_data(): 
    orig_ds = xr.open_dataset(PATHS['temp'])
    orig_np = orig_ds['temperature'].values
    regridded_ds = xr.open_dataset(PATHS['temp_regridded'])
    regridded_np = regridded_ds['temperature'].values
    times = orig_ds['time'].values

    times = times[:-3] # ignore 2024's months because they are incomplete
    orig_np = orig_np[:-3]
    regridded_np = regridded_np[:-3]
    print ('times', times[-48:])

    years = [int(t) for t in times]
    num_years = max(years) - min(years) + 1
    print ('min year:', min(years))
    print ('max year:', max(years))
    print ('number of years:', num_years)

    months = np.tile(np.arange(1, 13), num_years)
    print ('number of months:', len(months))
    print ('years:', years[-48:])
    print ('months:', months[-48:])

    chosen_years = [2020, 2021, 2022, 2023]
    chosen_ids = [i for i, t in enumerate(times) if t in chosen_years]
    orig_np = orig_np[chosen_ids]
    regridded_np = regridded_np[chosen_ids]

    print ('number of nan values in orig_np:', np.isnan(orig_np).sum())
    print ('number of nan values in regridded_np:', np.isnan(regridded_np).sum())


def get_models_paths(): 
    test_models_paths = {} 
    for folder_name in os.listdir(PATHS['model']):
        folder_path = os.path.join(PATHS['model'], folder_name)
        
        # Check if it is a directory
        if os.path.isdir(folder_path):
            checkpoints_dir = os.path.join(folder_path, 'checkpoints')
            
            # Check if the checkpoints directory exists
            if os.path.exists(checkpoints_dir) and os.path.isdir(checkpoints_dir):
                # List all files in the checkpoints directory
                for ckpt_file in os.listdir(checkpoints_dir):
                    if ckpt_file.endswith('.ckpt') and ckpt_file.startswith('epoch'):
                        ckpt_file_path = os.path.join(checkpoints_dir, ckpt_file)
                        #print(f"Found checkpoint: {ckpt_file_path}")
                        test_models_paths[folder_name] = ckpt_file_path
            else:
                print(f"No checkpoints directory in {folder_path}")
        else:
            print(f"{folder_name} is not a directory")

    # Confirm models loaded 
    #for m in test_models_paths: 
        #print(f" - {m} : {test_models_paths[m]}")
    
    return test_models_paths 


def get_normalize(variables):
    normalize_mean = dict(np.load(os.path.join(root_dir, "normalize_mean.npz")))
    normalize_mean = np.concatenate([normalize_mean[v] for v in variables], axis=0)
    normalize_std = dict(np.load(os.path.join(root_dir, "normalize_std.npz")))
    normalize_std = np.concatenate([normalize_std[v] for v in variables], axis=0)
    return transforms.Normalize(normalize_mean, normalize_std)


def get_data_given_path(path, variables):
    # with h5py.File(path, 'r') as f:
    #     data = {
    #         main_key: {
    #             sub_key: np.array(value) for sub_key, value in group.items() if sub_key in variables
    #     } for main_key, group in f.items() if main_key in ['input']}
    

    # for key, value in data.items():
    #     print(key)
    
    # print(data)
    # print(data.shape)

    # data = [data['input'][v] for v in variables]
    # return np.stack(data, axis=0)

    orig_ds = xr.open_dataset(PATHS['temp'])
    times = orig_ds['time'].values
    times = times[:-3] # ignore 2024's months because they are incomplete

    regridded_ds = xr.open_dataset(path)
    regridded_np = regridded_ds['temperature'].values
    regridded_np = regridded_np[:-3]

    return regridded_np


def get_item(data_path): 
    in_data = get_data_given_path(data_path, in_variables) # (V, H, W)
    in_data = torch.from_numpy(in_data) # (V, H, W)
    out_data = get_data_given_path(data_path, out_variables) # (V', H, W)
    out_data = torch.from_numpy(out_data)
    lead_times = torch.Tensor([0.0]).to(dtype=in_data.dtype)

    # CONVERT IN_DATA AND OUT_DATA FROM C TO K
    c_to_k = 273.15
    in_data = in_data - c_to_k
    out_data = out_data - c_to_k

    #if self.predefined_mask_dict is None:
    mask_ratio = random.choice(mask_ratio_range)
    mask = np.random.choice([0, 1], size=in_data.shape[1:], p=[mask_ratio, 1 - mask_ratio]) # (H, W)
    mask = torch.from_numpy(mask).to(dtype=in_data.dtype)
    
    return (
        in_transforms(in_data) * mask,
        out_transforms(out_data),
        lead_times,
        mask,
        in_variables,
        out_variables,
    )


def load_model(path): 
    model = None 

    if "unet" in path: 
        model = Unet(
                    in_channels = 1,
                    out_channels = 1,
                    hidden_channels = 128,
                    activation = "leaky",
                    norm = True,
                    dropout = 0.1,
                    ch_mults = [1, 2, 2, 4],
                    is_attn = [False, False, False, False],
                    mid_attn = False,
                    n_blocks = 2,
                )
                
    elif "climax" in path: 
        model = Climax(
                    default_vars = in_variables, 
                    img_size=[128, 256],
        )
    elif "stormer" in path: 
        model = Stormer(
            in_img_size = [128, 256], 
            in_variables = in_variables, 
            out_variables = out_variables,
        )
    else:
        return model # Error  

    
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    model.eval() 

    return model 


"""

Main Script 

"""

#inspect_temp_data() 

test_models_paths = get_models_paths()


in_variables = ["2m_temperature"]
out_variables = ["2m_temperature"]
mask_ratio_range = [0.1, 0.3, 0.5, 0.7, 0.9]

root_dir = '/localhome/data/datasets/climate/wb2/1.40625deg_6hr_h5df'

in_transforms = get_normalize(in_variables)
out_transforms = get_normalize(out_variables)

normalization = out_transforms
mean, std = normalization.mean, normalization.std
denormalization = transforms.Normalize(mean, std)

lat = np.load(os.path.join(root_dir, "lat.npy"))
lon = np.load(os.path.join(root_dir, "lon.npy"))


for m in test_models_paths: 

    x_dict, y, lead_times, mask_dict, in_variables, out_variables = get_item(PATHS['temp_regridded'])

    curr_model_path = test_models_paths[m]
    print(f"LOADING: {curr_model_path}")

    curr_model = load_model(curr_model_path)

    for mask_ratio in mask_dict.keys():
            x = x_dict[mask_ratio]
            mask = mask_dict[mask_ratio]
            pred = curr_model(x, lead_times, in_variables, out_variables)
            metrics = [lat_weighted_mse_val, lat_weighted_rmse, pearson]
            all_loss_dicts = [
                m(pred, y, denormalization, vars=out_variables, lat=lat, clim=None, log_postfix="", mask=(1-mask)) for m in metrics
            ]
            # combine loss dicts
            loss_dict = {}
            for d in all_loss_dicts:
                for k in d.keys():
                    loss_dict[k] = d[k]

            for var in loss_dict.keys():
                print(
                    f"{split}/{var}_{mask_ratio} | {loss_dict[var]}"
                )



    
    