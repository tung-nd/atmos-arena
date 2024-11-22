import os
from glob import glob
import argparse
import xarray as xr
import numpy as np
import xesmf as xe

def regrid(
        ds_in,
        ddeg_out,
        method='bilinear',
        reuse_weights=True,
        cmip=False,
        rename=None
):
    """
    Regrid horizontally.
    :param ds_in: Input xarray dataset
    :param ddeg_out: Output resolution
    :param method: Regridding method
    :param reuse_weights: Reuse weights for regridding
    :return: ds_out: Regridded dataset
    """
    # Rename to ESMF compatible coordinates
    if 'latitude' in ds_in.coords:
        ds_in = ds_in.rename({'latitude': 'lat', 'longitude': 'lon'})
    if cmip:
        ds_in = ds_in.drop(('lat_bnds', 'lon_bnds'))
        if hasattr(ds_in, 'plev_bnds'):
            ds_in = ds_in.drop(('plev_bnds'))
        if hasattr(ds_in, 'time_bnds'):
            ds_in = ds_in.drop(('time_bnds'))
    if rename is not None:
        ds_in = ds_in.rename({rename[0]: rename[1]})

    # Create output grid
    grid_out = xr.Dataset(
        {
            'lat': (['lat'], np.arange(-90+ddeg_out/2, 90, ddeg_out)),
            'lon': (['lon'], np.arange(0, 360, ddeg_out)),
        }
    )

    # Create regridder
    regridder = xe.Regridder(
        ds_in, grid_out, method, periodic=True, reuse_weights=reuse_weights
    )

    # Hack to speed up regridding of large files
    ds_out = regridder(ds_in, keep_attrs=True).astype('float32')

    if rename is not None:
        if rename[0] == 'zg':
            ds_out['z'] *= 9.807
        if rename[0] == 'rsdt':
            ds_out['tisr'] *= 60*60
            ds_out = ds_out.isel(time=slice(1, None, 12))
            ds_out = ds_out.assign_coords({'time': ds_out.time + np.timedelta64(90, 'm')})

    return ds_out

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Regrid NetCDF files.')
    parser.add_argument('--path', type=str, help='Input path')
    parser.add_argument('--save_path', type=str, required=True, help='Output path')
    parser.add_argument('--ddeg_out', type=float, default=5.625, help='Output resolution in degrees')
    args = parser.parse_args()

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path, exist_ok=True)

    list_simu = ['hist-GHG.nc', 'hist-aer.nc', 'historical.nc', 'ssp126.nc', 'ssp370.nc', 'ssp585.nc', 'ssp245.nc']
    ps = glob(os.path.join(args.path, f"*.nc"))
    ps_ = []
    for p in ps:
        for simu in list_simu:
            if simu in p:
                ps_.append(p)
    ps = ps_

    constant_vars = ['CO2', 'CH4']
    for p in ps:
        x = xr.open_dataset(p)
        if 'input' in p:
            for v in constant_vars:
                x[v] = x[v].expand_dims(dim={'latitude': 96, 'longitude': 144}, axis=(1,2))
        x_regridded = regrid(x, args.ddeg_out, reuse_weights=False)
        x_regridded.to_netcdf(os.path.join(args.save_path, os.path.basename(p)))