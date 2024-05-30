import os
import h5py
import numpy as np
from glob import glob
from tqdm import tqdm
import argparse

def main(data_root_dir, split, save_dir, mask_ratios):
    os.makedirs(save_dir, exist_ok=True)
    
    # get all input file paths
    in_file_paths = sorted(glob(os.path.join(data_root_dir, split, '*.h5')))
    sample_file_path = in_file_paths[0]
    with h5py.File(sample_file_path, 'r') as f:
        sample_data = {
            main_key: {
                sub_key: np.array(value) for sub_key, value in group.items()
        } for main_key, group in f.items() if main_key in ['input']}    
    in_variables = list(sample_data['input'].keys())
    h, w = sample_data['input'][in_variables[0]].shape
    len_each_mask = len(in_file_paths)
    
    for ratio in tqdm(mask_ratios, desc='Generating masks', unit='mask'):
        # 0 mean masked, 1 mean unmasked
        mask = np.random.choice([0, 1], size=(len_each_mask, h, w), p=[ratio, 1 - ratio])
        # save mask as .npy file
        save_path = os.path.join(save_dir, f'{split}_{ratio}.npy')
        np.save(save_path, mask)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate infilling masks for datasets.')
    parser.add_argument('--data_root_dir', type=str, required=True, help='Root directory of the data')
    parser.add_argument('--split', type=str, default='test', help='Data split to use (default: test)')
    parser.add_argument('--save_dir', type=str, required=True, help='Directory to save the generated masks')
    parser.add_argument('--mask_ratios', type=float, nargs='+', default=[0.1, 0.3, 0.5, 0.7, 0.9], help='List of masking ratios to use')

    args = parser.parse_args()
    main(args.data_root_dir, args.split, args.save_dir, args.mask_ratios)