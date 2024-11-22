This README provides instructions on downloading and processing different datasets used in AtmosArena.

# ERA5

We use ERA5 data from WeatherBench 2 (WB2) for Medium-range Weather Forecasting, S2S Forecasting, Climate Downscaling, and Climate Data Infilling. To download WB2 data, run

```bash
python atmos_arena/data_processing/download_wb2.py --file [DATASET_NAME] --save_dir [SAVE_DIR]
```
in which [DATASET_NAME] refers to the specific version of ERA5 that WB2 offers, e.g., `1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr`. For more detail, see [here](https://weatherbench2.readthedocs.io/en/latest/data-guide.html#era5). Note that this will download all available variables in the dataset. After downloading, the data sructure should look like the following:

```bash
wb2_nc/
├── 2m_temperature/
│   ├── 1959.nc
│   ├── 1960.nc
│   ├── ...
│   └── 2023.nc
├── geopotential/
├── specific_humidity/
├── other variables...
├── sea_surface_temperature.nc
├── sea_ice_cover.nc
├── surface_pressure.nc
├── total_cloud_cover.nc
└── other constants...
```

(Optional) If you want to regrid the data to a different resolution, e.g., 1.40625&deg;, run

```bash
python atmos_arena/data_processing/regrid_wb2.py \
    --root_dir [ROOT_DIR] \
    --save_dir [SAVE_DIR] \
    --ddeg_out 1.40625 \
    --start_year [START_YEAR] \
    --end_year [END_YEAR] \
    --chunk_size [CHUNK_SIZE]
```

We then convert the netCDF file to H5DF format for easier data loading with Pytorch. To do this, run

```bash
python atmos_arena/data_processing/nc_to_h5df_era5.py \
    --root_dir [ROOT_DIR] \
    --save_dir [SAVE_DIR] \
    --start_year [START_YEAR] \
    --end_year [END_YEAR] \
    --split [SPLIT] \
    --chunk_size [CHUNK_SIZE]
```

The H5DF data should have the following structure

```bash
wb2_h5df/
├── train/
│   ├── 1979_0000.h5
│   ├── 1979_0001.h5
│   ├── ...
│   ├── 2018_1457.h5
│   └── 2018_1458.h5
├── val/
│   └── validation files...
├── test/
│   └── test files...
├── lat.npy
└── lon.npy
```

in which each h5 file of name `{year}_{idx}.h5` contains the data for all variables of a specific time of the year. The time interval between two consecutive indices depends on the data frequence, which is 6 hours by default in WB2.

Finally, we pre-compute the normalization constants needed for training. To do this, run

```bash
python atmos_arena/data_processing/compute_era5_normalization.py \
    --root_dir [ROOT_DIR] \
    --save_dir [SAVE_DIR] \
    --start_year [START_YEAR] \
    --end_year [END_YEAR] \
    --chunk_size [CHUNK_SIZE] \
    --lead_time [LEAD_TIME] \
    --data_frequency [FREQUENCY]
```

NOTE: start and end year must correspond to training data. Root dir should point to wb2_nc directory, and save_dir is your H5DF data directory. To compute normalization constants for the input, set LEAD_TIME to None, otherwise set it to the interval value you want to compute normalization constants for, e.g., 6.

## ERA5-S2S

To construct ERA5-S2S for the S2S forecasting task, run

```bash
python atmos_arena/data_processing/construct_s2s.py \
    --root_dir [ROOT_DIR] \
    --save_dir [OUT_DIR] \
    --lead_time_list 336 672 1008 1344
```

# ClimateNet

The data is publicly available at https://portal.nersc.gov/project/ClimateNet/. To download all data points, run

```bash
python atmos_arena/data_processing/download_climatenet.py --save_dir [SAVE_DIR]
```

(Optional) If you want to perform feature engineering to obtained derived variables similar to https://arxiv.org/abs/2304.00176, run

```bash
python atmos_arena/data_processing/feature_engineer_climatenet.py \
    --origin_folder [ORIGIN_FOLDER] \
    --dest_folder [DEST_FOLDER]
```

Finally, to compute normalization constants needed for training, run

```bash
python atmos_arena/data_processing/compute_climatenet_normalization.py --root_dir [ROOT_DIR]
```

# Berkeley Earth

The monthly land and ocean temperature data is publicly available at https://berkeley-earth-temperature.s3.us-west-1.amazonaws.com/Global/Gridded/Land_and_Ocean_LatLong1.nc. After downloading the data, you can optinally regrid the data to 1.40625 degree.

# ClimateBench

The data is publicly available at https://zenodo.org/records/7064308. After downloading the data, it should have the following structure:

```bash
├── train_val/
│   ├── inputs_1pctCO2.nc
│   ├── inputs_abrupt-4xCO2.nc
│   ├── inputs_hist-GHG.nc
│   ├── inputs_historical.nc
│   ├── inputs_ssp370-lowNTCF.nc 
│   ├── inputs_ssp370.nc
│   ├── inputs_ssp585.nc
│   ├── outputs_abrupt-4xCO2.nc
│   ├── outputs_hist-GHG.nc
│   ├── outputs_historical.nc
│   ├── outputs_piControl.nc
│   ├── outputs_ssp126.nc
│   ├── outputs_ssp370-lowNTCF.nc
│   ├── outputs_ssp370.nc
│   └── outputs_ssp585.nc
└── test/
   ├── inputs_historical.nc
   ├── inputs_ssp245.nc
   ├── outputs_historical.nc
   └── outputs_ssp245.nc
```

(Optional) To regrid the data to 5.625 degree, run

```bash
python atmos_arena/data_processing/regrid_climatebench.py --path [ROOT_DIR] --save_path [OUT_DIR] --ddeg_out 5.625
```

# GEOS-CF

The GEOS-CF data is publicly available at https://portal.nccs.nasa.gov/datashare/gmao/geos-cf/. To download the data, run:

```bash
python atmos_arena/data_processing/download_geoscf.py --dest_folder [DEST_FOLDER]
```

By default, the script downloads hourly chemistry data (`chm_tavg_1hr_g1440x721_v1`) using 60 threads.

(Optional) To regrid the data to 1.40625 degree for 2019, 2020, and 2021, run

```bash
python atmos_arena/data_processing/regrid_geoscf.py \
    --root_dir [ROOT_DIR] \
    --save_dir [SAVE_DIR] \
    --ddeg_out 1.40625 \
    --years 2019 2020 2021
```

# CAMS Analysis

CAMS Analysis is avaiable to download via the CDS API. To download the data, run

```bash
python atmos_arena/data_processing/download_cams.py \
    --variable [VAR_NAME] \
    --year [YEAR] \
    --save_dir [SAVE_DIR] \
    --api_key_path [API_KEY_PATH]
```

(Optional) To regrid the data to 1.40625 degree, run

```bash
python atmos_arena/data_processing/regrid_cams.py --root_dir [ROOT_DIR] --save_dir [OUT_DIR] --ddeg_out 1.40625
```

To convert .nc data into H5DF format, run

```bash
python atmos_arena/data_processing/nc_to_h5df_cams.py \
    --root_dir [NC_DIR] \
    --save_dir [OUT_DIR] \
    --start [START_YEAR] \
    --end [END_YEAR] \
    --split [SPLIT]
```

Finally, compute the normalization constants needed for training by running

```bash
python atmos_arena/data_processing/compute_cams_normalization.py --root_dir [NC_DIR] --save_dir [H5DF_DIR]
```