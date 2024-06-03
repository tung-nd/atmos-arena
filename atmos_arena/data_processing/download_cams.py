import os
import argparse
import cdsapi
import yaml
import xarray as xr

PRESSURE_LEVELS = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
PRESSURE_LEVEL_VARS = [
    "carbon_monoxide",
    "nitrogen_dioxide",
    "nitrogen_monoxide",
    "ozone",
    "sulphur_dioxide",
    "u_component_of_wind",
    "v_component_of_wind",
    "specific_humidity",
    "temperature",
    "geopotential",
]
HOURS = ['00:00', '12:00']


def download_cams(save_dir, variable, year, api_key_path):
    os.makedirs(os.path.join(save_dir, variable), exist_ok=True)
    
    # check if file has already been downloaded
    year_name = '2017_to_2022' if year is None else f'{year}'
    if os.path.exists(os.path.join(save_dir, variable, f"{year_name}.nc")):
        # check if file can be opened with xarray
        try:
            xr.open_dataset(os.path.join(save_dir, variable, f"{year_name}.nc"))
            print (f"File {variable}/{year_name}.nc already exists. Skipping download.")
            return
        except:
            pass
    
    with open(api_key_path, 'r') as f:
        credentials = yaml.safe_load(f)
    client = cdsapi.Client(url=credentials['url'], key=credentials['key'])
    
    if year is None:
        date = "2017-10-01/2022-11-30"
    elif year == 2017:
        date = f"{year}-10-01/{year}-12-31"
    elif year == 2022:
        date = f"{year}-01-01/{year}-11-30"
    else:
        date = f"{year}-01-01/{year}-12-31"
    download_args = {
        'type': 'forecast',
        'date': date,
        "format": "netcdf_zip",
        "variable": variable,
        "time": HOURS,
        'leadtime_hour': '0',
    }
    if variable in PRESSURE_LEVEL_VARS:
        download_args["pressure_level"] = PRESSURE_LEVELS
    
    client.retrieve('cams-global-atmospheric-composition-forecasts', download_args, os.path.join(save_dir, variable, f"{year_name}.netcdf_zip"))

def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--variable", type=str, required=True)
    parser.add_argument("--year", type=int, default=None)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--api_key_path", type=str, default='/home/tungnd/.cdsapirc_ads')

    args = parser.parse_args()
    
    if args.variable in PRESSURE_LEVEL_VARS:
        assert args.year is not None, "Year must be specified for pressure level variables"
    else:
        assert args.year is None, "Year must not be specified for surface level variables"
    
    os.makedirs(args.save_dir, exist_ok=True)
    download_cams(args.save_dir, args.variable, args.year, args.api_key_path)

if __name__ == "__main__":
    main()