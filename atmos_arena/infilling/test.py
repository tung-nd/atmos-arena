import xarray as xr
import numpy as np
import os 
import torch 
import h5py
import numpy as np 
import random 
import json
from torchvision.transforms import transforms
from pprint import pprint
from tqdm import tqdm

from climax_arch import ClimaX
from stormer_arch import Stormer
from unet_arch import Unet
from atmos_utils.metrics import lat_weighted_rmse


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
    
    return test_models_paths 


def get_normalize(variables):
    normalize_mean = dict(np.load(os.path.join(root_dir, "normalize_mean.npz")))
    normalize_mean = np.concatenate([normalize_mean[v] for v in variables], axis=0)
    normalize_std = dict(np.load(os.path.join(root_dir, "normalize_std.npz")))
    normalize_std = np.concatenate([normalize_std[v] for v in variables], axis=0)
    return transforms.Normalize(normalize_mean, normalize_std)


def get_data_given_path(path, variables):
    def remove_nan_observations(array):
        mask = ~np.isnan(array).any(axis=tuple(range(1, array.ndim)))
        num_removed = np.sum(~mask)
        print(f"Number of indices removed for NaNs: {num_removed}")
        return array[mask]

    regridded_ds = xr.open_dataset(path)
    regridded_np = regridded_ds['temperature'].values
    years = regridded_ds.time.values
    window = (years >= 2020) & (years < 2024) # 2020-2023
    regridded_np = regridded_np[window]
    regridded_np = remove_nan_observations(regridded_np)
        
    return regridded_np

class Dataset(torch.utils.data.DataLoader): 
    def __init__(self, data_path, mask_ratio_range):
        data_path = data_path
        in_data = get_data_given_path(data_path, in_variables) # (T, H, W)
        in_data = torch.from_numpy(in_data) # (T, H, W)
        out_data = get_data_given_path(data_path, out_variables) # (T, H, W)
        out_data = torch.from_numpy(out_data)

        # adjust units
        c_to_k = lambda x : x + 273.15
        self.in_data = c_to_k(in_data)
        self.out_data = c_to_k(out_data)
     
        self.length = in_data.shape[0]
        
        self.mask_ratio_range = mask_ratio_range
        
    def __getitem__(self, i): 
        mask = self.get_mask()
        return (
            in_transforms(self.in_data[i:i+1,:,:]) * mask,
            out_transforms(self.out_data[i:i+1,:,:]),
            mask
        )
    
    def __len__(self):
        return self.length

    def get_mask(self):
        mask_ratio = random.choice(self.mask_ratio_range)
        mask = np.random.choice([0, 1], size=self.in_data.shape[1:], p=[mask_ratio, 1 - mask_ratio]) # (H, W)
        mask = torch.from_numpy(mask).to(dtype=self.in_data.dtype)
        return mask

def load_model(path): 

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
        model = ClimaX(
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
        raise NotImplementedError()

    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    model.eval() 

    return model 

def log_metrics(path, dictionary):
    # Helper function to replace tensors with their scalar values
    def replace_tensors(d):
        for key, value in d.items():
            if isinstance(value, dict):
                d[key] = replace_tensors(value)
            # elif isinstance(value, type(torch.tensor([0]))):
            else:    
                d[key] = value.item()
            # else:
            #     d[key] = str(value)
        return d
    
    # Replace tensors in the dictionary
    updated_dict = replace_tensors(dictionary)
    
    # print(updated_dict)
    # Save the updated dictionary as a JSON file
    with open(path, 'w') as json_file:
        json.dump(updated_dict, json_file)

"""

Main Script 

"""

######################################################################
# SET PARAMETERS
######################################################################
PATHS = {
    'temp_regridded' : '/localhome/data/datasets/climate/berkeley_land_and_ocean_monthly_temperature_128_256.nc', 
    'temp' : '/localhome/data/datasets/climate/berkeley_land_and_ocean_monthly_temperature.nc', 
    'model' : '/localhome/tungnd/atmos_arena/infilling/' # Loop through each folder, cd checkpoints, run epoch_x.ckpt
}
batch_size = 1
root_dir = '/localhome/data/datasets/climate/wb2/1.40625deg_6hr_h5df'
in_variables = ["2m_temperature"]
out_variables = ["2m_temperature"]
mask_ratio_range = [0.1, 0.3, 0.5, 0.7, 0.9]
device = 'cuda'
log_dir = './infilling/results'
#######################################################################

# inspect_temp_data() 

# get models
print(f"\nSearching for model paths in {PATHS['model']}...")
test_models_paths = get_models_paths()
pprint(list(test_models_paths.keys()))
print()

# set mask ratios
print("Mask ratios used:")
print(mask_ratio_range)
print()

# set transforms
print('Input variables:')
pprint(in_variables)
print('Output variables:')
pprint(out_variables)
print()
in_transforms = get_normalize(in_variables)
out_transforms = get_normalize(out_variables)
mean, std = out_transforms.mean, out_transforms.std
std_denorm = 1 / std
mean_denorm = -mean * std_denorm
denormalization = transforms.Normalize(mean_denorm, std_denorm)

# set constants
lat = np.load(os.path.join(root_dir, "lat.npy"))
lon = np.load(os.path.join(root_dir, "lon.npy"))
lead_time = torch.Tensor([0.0]).to(device)

for ratio in mask_ratio_range:
    print(f'Evaluating for mask ratio = {ratio}')
    dataset = Dataset(PATHS['temp_regridded'], [ratio])
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size) 

    ratio_log_path = os.path.join(log_dir, f'{(ratio*10):02d}')
    if not os.path.exists(ratio_log_path):
        os.makedirs(ratio_log_path)
            
    for m in test_models_paths: 

        model_path = test_models_paths[m]
        print(f"Loading checkpoint: {model_path}")
        model = load_model(model_path)
        model = model.to(device)
        
        model_log_path = os.path.join(ratio_log_path, m)
        if not os.path.exists(model_log_path):
            os.makedirs(model_log_path)
        
        with torch.no_grad():
            for i, (x, y, mask) in enumerate(tqdm(dataloader)):
                x = x.to(device)
                y = y.to(device)
                mask = mask.to(device)
                pred = model(x, lead_time, in_variables, out_variables)
                loss_dict = lat_weighted_rmse(
                    pred, 
                    y, 
                    denormalization,
                    out_variables, 
                    lat, 
                    clim=None,
                    mask=(1-mask)
                )
                log_path_i = os.path.join(model_log_path,f'{i:04d}.json')
                log_metrics(log_path_i, loss_dict)
    
    print('Ratio complete!')

print('Done! See `atmos_arena/infilling/get_final_metrics.ipynb` for final metrics for each model')

   



    
    