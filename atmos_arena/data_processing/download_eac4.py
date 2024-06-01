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
HOURS = ['00:00', '03:00', '06:00', '09:00', '12:00', '15:00', '18:00', '21:00']


def download_eac4(save_dir, variable, year, api_key_path):
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
    
    download_args = {
        'date': f"{year}-01-01/{year}-12-31",
        "format": "netcdf",
        "variable": variable,
        "time": HOURS,
    }
    if variable in PRESSURE_LEVEL_VARS:
        download_args["pressure_level"] = PRESSURE_LEVELS
    
    client.retrieve('cams-global-reanalysis-eac4', download_args, os.path.join(save_dir, variable, f"{year}.nc"))

# with open('/home/tungnd/.cdsapirc_ads', 'r') as f:
#     credentials = yaml.safe_load(f)
# c = cdsapi.Client(url=credentials['url'], key=credentials['key'])

# # Data retrieval request
# c.retrieve(
#     'cams-global-reanalysis-eac4',
#     {
#         'date': '2003-01-01/2023-12-30',  # Date range for the data
#         'format': 'netcdf',  # Output format
#         'variable': [
#             '10m_u_component_of_wind', '10m_v_component_of_wind', '2m_temperature',
#             'carbon_monoxide', 'geopotential', 'mean_sea_level_pressure',
#             'nitrogen_dioxide', 'nitrogen_monoxide', 'ozone',
#             'particulate_matter_10um', 'particulate_matter_1um', 'particulate_matter_2.5um',
#             'specific_humidity', 'sulphur_dioxide', 'temperature',
#             'total_column_carbon_monoxide', 'total_column_nitrogen_dioxide', 'total_column_nitrogen_monoxide',
#             'total_column_ozone', 'total_column_sulphur_dioxide'
#         ],
#         'pressure_level': [
#             '50', '100', '150', '200', '250', '300', '400', '500', '600',
#             '700', '850', '925', '1000'
#         ],
#         'time': [
#             '00:00', '03:00', '06:00', '09:00', '12:00', '15:00',
#             '18:00', '21:00'
#         ],
#     },
#     'download.nc'  # Output file name
# )

def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--variable", type=str, required=True)
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--api_key_path", type=str, default='/home/tungnd/.cdsapirc_ads')

    args = parser.parse_args()
    
    os.makedirs(args.save_dir, exist_ok=True)
    download_eac4(args.save_dir, args.variable, args.year, args.api_key_path)

if __name__ == "__main__":
    main()