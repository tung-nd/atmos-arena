import os
import xarray as xr
import numpy as np
from tqdm import tqdm
from glob import glob
import argparse
import atmos_arena.data_processing.regridding as regridding

def main(args):
    os.makedirs(args.save_dir, exist_ok=True)
    
    vars = [
        "2m_temperature",
        "10m_u_component_of_wind",
        "10m_v_component_of_wind",
        "mean_sea_level_pressure",
        "particulate_matter_10um",
        "particulate_matter_1um",
        "particulate_matter_2.5um",
        "total_column_carbon_monoxide",
        "total_column_nitrogen_dioxide",
        "total_column_nitrogen_monoxide",
        "total_column_ozone",
        "total_column_sulphur_dioxide",
        
        "carbon_monoxide",
        "geopotential",
        "nitrogen_dioxide",
        "nitrogen_monoxide",
        "ozone",
        "specific_humidity",
        "sulphur_dioxide",
        "temperature",
        "u_component_of_wind",
        "v_component_of_wind",
    ]

    lat_start = -90 + args.ddeg_out/2
    lat_stop = 90 - args.ddeg_out/2
    new_lat = np.linspace(lat_start, lat_stop, num=128, endpoint=True)
    new_lon = np.linspace(0, 360, num=256, endpoint=False)
    regridder = None

    var_dirs = [os.path.join(args.root_dir, v) for v in vars]
    for dir in tqdm(var_dirs, desc='vars', position=0):
        var_name = os.path.basename(dir)
        os.makedirs(os.path.join(args.save_dir, var_name), exist_ok=True)
        list_nc_files = glob(os.path.join(dir, '*.nc'))
        for nc_file in tqdm(list_nc_files, desc='files', position=1, leave=False):
            ds_in = xr.open_dataset(nc_file, chunks={'time': 100})
            if regridder is None:
                old_lon = ds_in.coords['longitude'].data
                old_lat = ds_in.coords['latitude'].data
                source_grid = regridding.Grid.from_degrees(lon=old_lon, lat=np.sort(old_lat))
                target_grid = regridding.Grid.from_degrees(lon=new_lon, lat=new_lat)
                regridder = regridding.ConservativeRegridder(source_grid, target_grid)
            
            ds_out = regridder.regrid_dataset(ds_in).transpose(..., 'latitude', 'longitude')
            ds_out.to_netcdf(os.path.join(args.save_dir, var_name, os.path.basename(nc_file)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Regrid CAMS data.')
    parser.add_argument('--root_dir', type=str, required=True, help='Root directory containing variable folders')
    parser.add_argument('--save_dir', type=str, required=True, help='Directory to save regridded files')
    parser.add_argument('--ddeg_out', type=float, default=1.40625, help='Output resolution in degrees')
    args = parser.parse_args()
    main(args)