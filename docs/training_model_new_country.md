# General Steps on How to Train a Model for a New Country

This guide provides step-by-step instructions for training a solar forecasting model for a new country using the Open Data PVNet framework. The process involves collecting data, configuring the system, and training the model.

## Prerequisites

Before starting, ensure you have:

- **Project Setup**: The Open Data PVNet project installed and configured (see [Getting Started Guide](getting_started.md))
- **Python Environment**: Python 3.9, 3.10, or 3.11 with all dependencies installed
- **Data Access**: Access to or ability to download:
  - Solar PV generation data for your target country
  - GFS (Global Forecast System) weather data
- **Storage**: Sufficient disk space for data storage (Zarr files can be large)
- **Optional**: AWS CLI configured (for S3 access), Hugging Face account (for model sharing)

## Choosing a Country

When selecting a country for training a new model, consider:

### Recommended Starting Countries

The following countries are good starting points due to data availability and solar capacity:

- **United Kingdom** - Already implemented (reference implementation)
- **United States** - Large solar capacity, good data availability
- **Netherlands** - High solar penetration, good data infrastructure
- **Belgium** - Strong renewable energy data systems
- **Germany** - One of the world's largest solar markets
- **France** - Significant solar capacity, good data access

### Finding Countries with Large Solar Installations

To identify other suitable countries:

1. **International Energy Agency (IEA)**: Check IEA's solar capacity statistics
2. **IRENA (International Renewable Energy Agency)**: Provides global renewable energy data
3. **National Grid Operators**: Many countries publish solar generation data
4. **Research Institutions**: Universities often maintain solar generation datasets

**Key Factors to Consider:**
- **Solar Capacity**: Countries with significant installed PV capacity (>1 GW)
- **Data Availability**: Public APIs or data portals for generation data
- **Data Quality**: Reliable, consistent, and regularly updated data
- **Geographic Coverage**: National or regional-level data (not just individual sites)

## Overview

Training a model for a new country requires:
1. **Generation Data**: Historical solar PV generation data for the country
2. **NWP Data**: Numerical Weather Prediction (weather forecast) data
3. **Configuration**: Setting up country-specific configuration files
4. **Training**: Running the model training pipeline
5. **Model Weights**: Saving and sharing the trained model

---

## Step 1: Get Generation Data

Generation data represents the actual solar PV power output for your target country. This serves as the "ground truth" for training the model.

### Options for Generation Data

#### Option A: Using Existing APIs (Recommended)

1. **PVlive API** (UK-specific, but similar APIs may exist for other countries)
   ```python
   from pvlive_api import PVLive
   
   pvl = PVLive()
   # Fetch generation data
   data = pvl.get_latest()
   ```

2. **Country-Specific APIs**
   - **Europe (Netherlands, Belgium, Germany, France)**: [ENTSO-E Transparency Platform](https://transparency.entsoe.eu/) - Provides generation data for European countries
   - **United States**: 
     - EIA (Energy Information Administration) - National level data
     - Regional ISOs: CAISO (California), ERCOT (Texas), PJM (Eastern US)
   - **United Kingdom**: PVlive API (already implemented in this project)
   - Check national grid operators and government energy data portals

3. **Manual Data Collection**
   - Download from national energy/grid operator websites
   - Use data from research institutions or universities
   - Format: CSV, JSON, or database exports

### Data Requirements

- **Time Resolution**: 30-minute intervals (recommended) or hourly
- **Coverage**: National or regional level
- **Time Period**: At least 1 year of historical data (more is better)
- **Format**: Should be convertible to Zarr format

### Converting to Zarr Format

Once you have generation data, convert it to Zarr format:

```python
import pandas as pd
import xarray as xr
import zarr

# Load your data (example with CSV)
df = pd.read_csv('generation_data.csv', parse_dates=['datetime'])

# Convert to xarray Dataset
ds = xr.Dataset(
    {
        'generation': (['time', 'gsp_id'], df.pivot_table(
            index='datetime', 
            columns='gsp_id', 
            values='generation'
        ).values)
    },
    coords={
        'time': df['datetime'].unique(),
        'gsp_id': df['gsp_id'].unique()
    }
)

# Save to Zarr
ds.to_zarr('generation_data.zarr', mode='w')
```

### Storage Location

Store your generation data in a location accessible to the training pipeline:
- **Local**: `./data/{country}/generation/{year}.zarr`
- **S3**: `s3://ocf-open-data-pvnet/data/{country}/generation/{year}.zarr`
- **Hugging Face**: Upload to a Hugging Face dataset

---

## Step 2: Get NWP Data (Start with GFS)

Numerical Weather Prediction (NWP) data provides weather forecasts that the model uses to predict solar generation. For new countries, **GFS (Global Forecast System)** is recommended as it provides global coverage.

### Why GFS?

- **Global Coverage**: Available for any country worldwide
- **Public Access**: Free and openly available
- **Good Resolution**: 0.25° (~25km) resolution
- **Multiple Variables**: Includes all necessary weather parameters

### Downloading GFS Data

#### Option A: Using the CLI (Recommended)

```bash
# Download GFS data for a specific date range
# Note: GFS data is available globally, so no region parameter needed
open-data-pvnet gfs archive --year 2023 --month 1 --day 1
```

#### Option B: Direct from AWS S3

GFS data is available on AWS S3:

```bash
# List available data
aws s3 ls s3://noaa-gfs-bdp-pds/gfs.20230101/00/atmos/ --no-sign-request

# Download specific files
aws s3 sync s3://noaa-gfs-bdp-pds/gfs.20230101/00/atmos/ ./gfs_data/ --no-sign-request
```

#### Option C: Using Python

```python
import xarray as xr
import s3fs

# Access GFS data from S3
s3 = s3fs.S3FileSystem(anon=True)
gfs_path = 's3://noaa-gfs-bdp-pds/gfs.20230101/00/atmos/'

# Open dataset
ds = xr.open_dataset(s3.open(gfs_path + 'gfs.t00z.pgrb2.0p25.f000'))
```

### Required GFS Variables

The model needs these weather variables (channels):

- `dswrf` - Downward shortwave radiation flux (solar radiation)
- `dlwrf` - Downward longwave radiation flux
- `tcc` - Total cloud cover
- `hcc` - High cloud cover
- `mcc` - Medium cloud cover
- `lcc` - Low cloud cover
- `t` - 2-metre temperature
- `r` - Relative humidity
- `u10`, `v10` - 10-metre wind components
- `u100`, `v100` - 100-metre wind components
- `prate` - Precipitation rate
- `vis` - Visibility

### Processing GFS Data for Your Country

1. **Crop to Country Region**: Extract only the geographic area covering your country
2. **Convert to Zarr**: Save in Zarr format for efficient access
3. **Time Alignment**: Ensure timestamps align with generation data

Example:

```python
import xarray as xr

# Load GFS data
gfs_ds = xr.open_zarr('gfs_global.zarr')

# Define bounding box for your country (example: Germany)
lat_min, lat_max = 47.0, 55.0
lon_min, lon_max = 5.0, 15.0

# Crop to country region
country_ds = gfs_ds.sel(
    latitude=slice(lat_max, lat_min),
    longitude=slice(lon_min, lon_max)
)

# Save country-specific GFS data
country_ds.to_zarr('gfs_germany.zarr', mode='w')
```

### Storage Location

Store processed GFS data:
- **Local**: `./data/{country}/gfs/{year}.zarr`
- **S3**: `s3://ocf-open-data-pvnet/data/{country}/gfs/{year}.zarr`

---

## Step 3: Make Configs for That Country

Create configuration files that tell the system where to find your data and how to process it.

### 3.1 Create Data Configuration File

Create a new configuration file: `src/open_data_pvnet/configs/PVNet_configs/datamodule/configuration/{country}_configuration.yaml`

Example for Germany (`germany_configuration.yaml`):

```yaml
general:
  description: Configuration for {Country} solar forecasting
  name: {country}_config

input_data:
  gsp:
    # Path to your generation data in zarr format
    zarr_path: "s3://ocf-open-data-pvnet/data/{country}/generation/2023.zarr"
    # Or local path: "./data/{country}/generation/2023.zarr"
    interval_start_minutes: -60  # 1 hour before forecast time
    interval_end_minutes: 480    # 8 hours after forecast time
    time_resolution_minutes: 30  # Match your data resolution
    dropout_timedeltas_minutes: []
    dropout_fraction: 0.0
    public: True  # Set to False if using private S3 bucket

  nwp:
    gfs:
      time_resolution_minutes: 180  # GFS resolution (3 hours)
      interval_start_minutes: -180  # 3 hours before
      interval_end_minutes: 540     # 9 hours after
      dropout_fraction: 0.0
      dropout_timedeltas_minutes: []
      # Path to your GFS data for the country
      zarr_path: "s3://ocf-open-data-pvnet/data/{country}/gfs/2023.zarr"
      provider: "gfs"
      # Adjust based on your cropped region size
      image_size_pixels_height: 32  # Adjust for your country
      image_size_pixels_width: 40   # Adjust for your country
      public: True
      channels:
        - dlwrf
        - dswrf
        - hcc
        - mcc
        - lcc
        - prate
        - r
        - t
        - tcc
        - u10
        - u100
        - v10
        - v100
        - vis
      # Normalization constants (calculate from your data)
      # IMPORTANT: You must calculate these from YOUR actual GFS data for your country
      # The values below are examples from UK data - replace with your country's values
      normalisation_constants:
        dlwrf:
          mean: 298.342
          std: 96.305916
        dswrf:
          mean: 168.12321
          std: 246.18533
        hcc:
          mean: 35.272
          std: 42.525383
        lcc:
          mean: 43.578342
          std: 44.3732
        mcc:
          mean: 33.738823
          std: 43.150745
        prate:
          mean: 2.8190969e-05
          std: 0.00010159573
        r:
          mean: 18.359747
          std: 25.440672
        t:
          mean: 278.5223
          std: 22.825893
        tcc:
          mean: 66.841606
          std: 41.030598
        u10:
          mean: -0.0022310058
          std: 5.470838
        u100:
          mean: 0.0823025
          std: 6.8899174
        v10:
          mean: 0.06219831
          std: 4.7401133
        v100:
          mean: 0.0797807
          std: 6.076132
        vis:
          mean: 19628.32
          std: 8294.022
      # Note: Calculate these from your GFS data using the script in section 3.2

  solar_position:
    interval_start_minutes: -60
    interval_end_minutes: 480
    time_resolution_minutes: 30
```

### 3.2 Calculate Normalization Constants

You need to calculate mean and std for each GFS channel from your data:

```python
import xarray as xr
import numpy as np

# Load your GFS data
gfs_ds = xr.open_zarr('gfs_{country}.zarr')

normalization = {}
# List all GFS channels you're using
channels = ['dlwrf', 'dswrf', 'hcc', 'mcc', 'lcc', 'prate', 'r', 't', 'tcc', 
            'u10', 'u100', 'v10', 'v100', 'vis']

for channel in channels:
    if channel in gfs_ds:
        data = gfs_ds[channel].values
        normalization[channel] = {
            'mean': float(np.mean(data)),
            'std': float(np.std(data))
        }
    else:
        print(f"Warning: Channel {channel} not found in dataset")

print(normalization)
# Copy these values into your config file
```

### 3.3 Update Training Configuration

Edit `src/open_data_pvnet/configs/PVNet_configs/datamodule/streamed_batches.yaml`:

```yaml
_target_: pvnet.data.DataModule
# Update this path to your country configuration
configuration: "/full/path/to/open-data-pvnet/src/open_data_pvnet/configs/PVNet_configs/datamodule/configuration/{country}_configuration.yaml"

num_workers: 8
prefetch_factor: 2
batch_size: 8

# Training period (adjust to your data availability)
train_period:
  - "2023-01-01"
  - "2023-06-30"
  
# Validation period
val_period:
  - "2023-07-01"
  - "2023-12-31"
```

### 3.4 Update Main Config

Edit `src/open_data_pvnet/configs/PVNet_configs/config.yaml`:

```yaml
# ... existing config ...

# Update renewable type if needed (currently "pv_uk")
renewable: "pv_uk"  # Keep as is for now, or create new type

# ... rest of config ...
```

---

## Step 4: Run and Train Model

### 4.1 Prepare Training Samples (Optional but Recommended)

For faster training, pre-generate samples:

```bash
# Update streamed_batches.yaml first (see Step 3.3)
python src/open_data_pvnet/scripts/save_samples.py
```

This creates training samples in `GFS_samples/` directory.

### 4.2 Configure Weights & Biases (Optional)

If using W&B for experiment tracking:

1. Create account at https://wandb.ai/
2. Edit `src/open_data_pvnet/configs/PVNet_configs/logger/wandb.yaml`:
   ```yaml
   project: "{country}_solar_forecast"
   save_dir: "{country}_runs"
   ```

### 4.3 Run Training

#### Option A: Using Pre-generated Samples (Faster)

1. Update `config.yaml`:
   ```yaml
   defaults:
     - datamodule: premade_batches.yaml
   ```

2. Update `premade_batches.yaml`:
   ```yaml
   configuration: "/full/path/to/your/{country}_configuration.yaml"
   sample_dir: "GFS_samples"  # Directory containing train/ and val/ subdirectories with batches
   ```

3. Run training:
   ```bash
   python run.py
   ```

#### Option B: Streaming Data (Slower but No Pre-processing)

1. Update `config.yaml`:
   ```yaml
   defaults:
     - datamodule: streamed_batches.yaml
   ```

2. Run training:
   ```bash
   python run.py
   ```

### 4.4 Monitor Training

- **Console Output**: Watch for loss values and metrics
- **Weights & Biases**: If configured, view at https://wandb.ai/
- **Logs**: Check `outputs/` directory for training logs

### 4.5 Training Tips

- **Start Small**: Use `num_train_samples: 5` and `num_val_samples: 5` for testing
- **Debug Mode**: Use `python run.py debug=true` for quick testing
- **GPU**: Training is faster on GPU (if available)
- **Data Quality**: Ensure data alignment and no missing values

---

## Step 5: Share / Save Model Weights

### 5.1 Locate Trained Model

After training, model checkpoints are saved in:
- `outputs/{timestamp}/checkpoints/` (Hydra default)
- Or location specified in your config

### 5.2 Save Model Weights

```python
import torch

# Load the best checkpoint
checkpoint = torch.load('outputs/.../checkpoints/best.ckpt')

# Extract model weights
model_state = checkpoint['state_dict']

# Save only weights
torch.save(model_state, '{country}_model_weights.pt')
```

### 5.3 Share Model Weights

#### Option A: Hugging Face Hub (Recommended)

```python
from huggingface_hub import HfApi

api = HfApi()
api.upload_file(
    path_or_fileobj='{country}_model_weights.pt',
    path_in_repo='models/{country}_weights.pt',
    repo_id='openclimatefix/{country}-solar-forecast',
    repo_type='model',
    token=os.getenv('HUGGINGFACE_TOKEN')
)
```

#### Option B: GitHub Releases

1. Create a new release on GitHub
2. Upload model weights as release assets
3. Document model version and training details

#### Option C: S3 Bucket

```bash
aws s3 cp {country}_model_weights.pt \
  s3://ocf-open-data-pvnet/models/{country}/weights.pt
```

### 5.4 Document Model

Create a README for your model:

```markdown
# {Country} Solar Forecasting Model

## Model Details
- **Country**: {Country Name}
- **Training Period**: {Start Date} to {End Date}
- **Data Sources**: GFS NWP, {Generation Data Source}
- **Performance**: MAE: {value}, RMSE: {value}

## Usage
[Include instructions on how to use the model]
```

---

## Troubleshooting

### Common Issues

1. **Data Not Found**
   - Check file paths in configuration
   - Verify S3 bucket permissions
   - Ensure data is in Zarr format

2. **Dimension Mismatches**
   - Verify `image_size_pixels_height/width` match your data
   - Check time alignment between generation and NWP data

3. **Out of Memory**
   - Reduce `batch_size` in config
   - Use smaller `num_workers`
   - Process data in smaller chunks

4. **Poor Model Performance**
   - Check data quality (missing values, outliers)
   - Verify normalization constants
   - Ensure sufficient training data (1+ years recommended)

---

## Next Steps

After training your first model:

1. **Evaluate Performance**: Compare predictions with actual generation
2. **Iterate**: Try different hyperparameters, data periods, or features
3. **Document**: Share your findings and model with the community
4. **Contribute**: Submit your model to the Open Climate Fix repository

---

## Resources

- [Getting Started Guide](getting_started.md)
- [Working with NWP Data](working_with_nwp.md)
- [GFS Data Understanding](notebooks/understanding_gfs_data.ipynb)
- [PVNet Documentation](https://github.com/openclimatefix/pvnet)

---

## Example: Complete Workflow for a New Country

This example uses **Germany** as a reference, but the same process applies to other countries like USA, Netherlands, Belgium, or France.

### Step-by-Step Example: Germany

```bash
# 1. Get generation data
# For Germany: Check ENTSO-E Transparency Platform or Bundesnetzagentur
# Save as: ./data/germany/generation/2023.zarr

# 2. Get GFS data for Germany region
open-data-pvnet gfs archive --year 2023 --month 1 --day 1
# Crop to Germany bounding box (lat: 47-55°N, lon: 5-15°E)
# Save as: ./data/germany/gfs/2023.zarr

# 3. Create configuration file
# Create: src/open_data_pvnet/configs/PVNet_configs/datamodule/configuration/germany_configuration.yaml
# Use the template in Step 3.1, replacing {country} with "germany"

# 4. Calculate normalization constants
# Run the script from Step 3.2 on your GFS data
# Update the normalisation_constants in your config file

# 5. Update training configuration
# Edit: streamed_batches.yaml with path to germany_configuration.yaml
# Set train_period and val_period based on your data availability

# 6. Generate training samples (optional but recommended)
python src/open_data_pvnet/scripts/save_samples.py

# 7. Train model
python run.py

# 8. Save and share model weights
# Model saved in: outputs/{timestamp}/checkpoints/best.ckpt
# Upload to Hugging Face or S3 (see Step 5)
```

### Country-Specific Data Sources

#### United States
- **Generation Data**: EIA (Energy Information Administration), CAISO, ERCOT
- **Data Format**: Often available via APIs or CSV downloads
- **Coverage**: Regional (ISO regions) or national level

#### Netherlands
- **Generation Data**: ENTSO-E Transparency Platform, TenneT (Dutch TSO)
- **Data Format**: API access via ENTSO-E, CSV exports
- **Coverage**: National level

#### Belgium
- **Generation Data**: ENTSO-E Transparency Platform, Elia (Belgian TSO)
- **Data Format**: API or CSV
- **Coverage**: National level

#### France
- **Generation Data**: ENTSO-E Transparency Platform, RTE (French TSO)
- **Data Format**: API access, data portal
- **Coverage**: National level

#### Germany
- **Generation Data**: ENTSO-E Transparency Platform, Bundesnetzagentur
- **Data Format**: API or CSV exports
- **Coverage**: National and regional level

**Note**: For all European countries, the [ENTSO-E Transparency Platform](https://transparency.entsoe.eu/) is a valuable resource for generation data.

---

**Need Help?** Open an issue on [GitHub](https://github.com/openclimatefix/open-data-pvnet/issues) or check the [Getting Started Guide](getting_started.md).

