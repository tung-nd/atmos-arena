# AtmosArena: Benchmarking Foundation Models for Atmospheric Sciences

Official implementation of AtmosArena, a comprehensive multi-task benchmark for evaluating foundation models in atmospheric sciences. AtmosArena provides standardized tasks, datasets, and evaluation metrics to facilitate systematic comparison of deep learning models across various atmospheric science applications.

## Supported Tasks and Datasets

AtmosArena encompasses a diverse suite of tasks in atmospheric sciences, organized into two main categories: atmospheric physics and atmospheric chemistry. Each task is paired with carefully curated datasets to ensure comprehensive evaluation of model capabilities.

### Atmospheric Physics Tasks
- **Medium-range Weather Forecasting**: Global weather prediction (hours to 2 weeks) using ERA5 dataset
- **Sub-seasonal-to-seasonal (S2S) Forecasting**: Extended prediction (2 weeks to 2 months) using ERA5 dataset
- **Extreme Weather Events Detection**: Identification of tropical cyclones and atmospheric rivers using ClimateNet dataset
- **Climate Downscaling**: Spatial resolution enhancement using ERA5 dataset
- **Climate Data Infilling**: Missing data estimation using ERA5 and Berkeley Earth datasets
- **Climate Model Emulation**: Predicting climate responses using ClimateBench dataset

### Atmospheric Chemistry Tasks
- **Chemistry Downscaling**: High-resolution transformation of chemical compositions using GEOS-CF dataset
- **Composition Forecasting**: Prediction of air pollutant concentrations using CAMS Analysis dataset

## Leaderboard

We actively maintain a leaderboard for all tasks and datasets at [https://atmosarena.github.io/leaderboard/](https://atmosarena.github.io/leaderboard/). Current models in evaluation include ClimaX, Stormer, and UNet. We will continuously expand the leaderboard with additional baselines, tasks, datasets, and metrics.

## Install

First, install the dependencies as listed in `env.yml` and activate the environment:

```bash
conda env create -f env.yml
conda activate atmosarena
```

Then, install the AtmosArena package

```bash
pip install -e .
```

## Usage

Please refer to each task's directory for instructions on reproducing the results in the paper. Individual components (models, datasets, metrics) can be used independently as well.