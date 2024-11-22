import os
import numpy as np
import xarray as xr
import glob
from pathlib import Path
from tqdm import tqdm
import argparse


def velocity(u, v):
    wind_speed = np.sqrt(u**2 + v**2)
    return wind_speed


def vorticity(u, v, lat):
    dv = v.differentiate("lon")
    cosTheta = np.cos(lat*(np.pi/180))
    
    u_cosTheta = u * cosTheta
    du = (1/(1e-3+cosTheta)) * u_cosTheta.differentiate("lat")

    zeta = dv - du
    return zeta.astype(np.float32)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process NetCDF files.')
    parser.add_argument('--origin_folder', type=str, required=True,
                      help='Input directory containing NetCDF files')
    parser.add_argument('--dest_folder', type=str, required=True,
                      help='Output directory for processed files')
    args = parser.parse_args()

    for split in ['train', 'test']:
        os.makedirs(os.path.join(args.dest_folder, split), exist_ok=True)
        for idx, filename_in in tqdm(enumerate(glob.glob(os.path.join(args.origin_folder, split, '*.nc')))):
            ds = xr.open_dataset(filename_in)

            file = Path(filename_in)

            lat = ds["lat"]
            lon = ds["lon"]

            # Wind speed at 850 hPa
            ds["WS850"] = velocity(ds.U850, ds.V850)
            ds.WS850.attrs['description'] = 'wind speed at 850 mbar pressure surface'
            ds.WS850.attrs['units'] = 'm/s'

            # Wind speed at lower level
            ds["WSBOT"] = velocity(ds.UBOT, ds.VBOT)
            ds.WSBOT.attrs['description'] = 'lowest level wind speed'
            ds.WSBOT.attrs['units'] = 'm/s'

            # Wind vorticity at 850 hPa
            ds["VRT850"] = vorticity(ds.U850[0,:,:], ds.V850[0,:,:], lat).astype(np.float32)
            ds.VRT850.attrs['description'] = 'wind vorticity at 850 mbar pressure surface'
            ds.VRT850.attrs['units'] = 's^-1'

            # Wind vorticity at 850 hPa
            ds["VRTBOT"] = vorticity(ds.UBOT[0,:,:], ds.VBOT[0,:,:], lat).astype(np.float32)
            ds.VRTBOT.attrs['description'] = 'lowest level wind vorticity'
            ds.VRTBOT.attrs['units'] = 's^-1'
        
            # Save sample to new folder
            filename_out = os.path.join(args.dest_folder, split, file.stem + file.suffix)

            ds.to_netcdf(filename_out)