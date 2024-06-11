import os
import xarray as xr
import numpy as np
from glob import glob
from tqdm import tqdm
import argparse
import data_processing.regrid_helpers as regridding

def main(year_strs):
    root_dir = '/eagle/MDClimSim/tungnd/data/geoscf/'
    save_dir = '/eagle/MDClimSim/tungnd/data/geoscf_1.40625'
    ddeg_out = 1.40625

    os.makedirs(save_dir, exist_ok=True)
    h, w = int(180 / ddeg_out), int(360 / ddeg_out)

    lat_start = -90 + ddeg_out/2
    lat_stop = 90 - ddeg_out/2
    new_lat = np.linspace(lat_start, lat_stop, num=h, endpoint=True)
    new_lon = np.linspace(0, 360, num=w, endpoint=False)
    regridder = None

    year_dirs = [os.path.join(root_dir, year_str) for year_str in year_strs]
    for year, dir in zip(year_strs, year_dirs):
        all_nc_files = glob(os.path.join(dir, '*.nc4'))
        os.makedirs(os.path.join(save_dir, year), exist_ok=True)
        for nc_file in tqdm(all_nc_files):
            if os.path.exists(os.path.join(save_dir, year, os.path.basename(nc_file))):
                continue
            base_name = os.path.basename(nc_file)
            try:
                ds_in = xr.open_dataset(nc_file).transpose(..., 'lat', 'lon')
                ds_in = ds_in.sortby('lat') # -90 to 90
                ds_in = ds_in.assign_coords(lon=((ds_in.lon + 360) % 360))
                ds_in = ds_in.sortby('lon')
                ds_in = ds_in.assign_coords(lon=np.linspace(0, 359.75, ds_in.dims['lon']))
                if regridder is None:
                    old_lon = ds_in.coords['lon'].data
                    old_lat = ds_in.coords['lat'].data
                    source_grid = regridding.Grid.from_degrees(lon=old_lon, lat=np.sort(old_lat))
                    target_grid = regridding.Grid.from_degrees(lon=new_lon, lat=new_lat)
                    regridder = regridding.ConservativeRegridder(source_grid, target_grid)
                ds_out = regridder.regrid_dataset(ds_in)
                ds_out.to_netcdf(os.path.join(save_dir, year, base_name))
            except:
                continue

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--years', nargs='+', help='List of years to process', required=True)
    args = parser.parse_args()
    main(args.years)