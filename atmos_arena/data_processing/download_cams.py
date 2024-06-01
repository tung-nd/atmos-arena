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
    if os.path.exists(os.path.join(save_dir, variable, f"{year}.nc")):
        # check if file can be opened with xarray
        try:
            xr.open_dataset(os.path.join(save_dir, variable, f"{year}.nc"))
            print (f"File {variable}/{year}.nc already exists. Skipping download.")
            return
        except:
            pass
    
    with open(api_key_path, 'r') as f:
        credentials = yaml.safe_load(f)
    client = cdsapi.Client(url=credentials['url'], key=credentials['key'])
    
    if year == 2017:
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
    
    client.retrieve('cams-global-atmospheric-composition-forecasts', download_args, os.path.join(save_dir, variable, f"{year}.netcdf_zip"))

def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--variable", type=str, required=True)
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--api_key_path", type=str, default='/home/tungnd/.cdsapirc_ads')

    args = parser.parse_args()
    
    os.makedirs(args.save_dir, exist_ok=True)
    download_cams(args.save_dir, args.variable, args.year, args.api_key_path)

if __name__ == "__main__":
    main()