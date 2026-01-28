# Running PVNet Model

## Overview
This guide provides step-by-step instructions for configuring and running the PVNet solar forecasting model using GFS (Global Forecast System) data.

## Prerequisites
- **Python Environment**: Virtual environment activated (if used)
- **AWS CLI**: Installed and configured (for data download)
- **Weights & Biases Account**: Required for experiment tracking ([Sign up](https://wandb.ai/))

---

## Step 1: Configure Sample Settings

### File: `src/open_data_pvnet/configs/PVNet_configs/datamodule/streamed_batches.yaml`

Update the following parameters:

```yaml
num_train_samples: 5    # Increase based on available compute resources
num_val_samples: 5      # Increase based on available compute resources
```

**Purpose**: Controls the number of training and validation samples. Start with small values (5) for testing, then scale up (e.g., 1000+) for production training.

---

## Step 2: Configure Batch Processing

### File: `src/open_data_pvnet/configs/PVNet_configs/datamodule/premade_batches.yaml`

Set the configuration path:

```yaml
configuration: <absolute_path_to_project>/src/open_data_pvnet/configs/PVNet_configs/datamodule/configuration/example_configuration.yaml
```

**Purpose**: Links to the data source configuration when running with pre-generated samples.

---

## Step 3: Set Active DataModule

### File: `src/open_data_pvnet/configs/PVNet_configs/config.yaml`

Change line 8 depending on the current step:

**For generating samples** (Step 5):
```yaml
- datamodule: streamed_batches.yaml
```

**For training** (Step 6):
```yaml
- datamodule: premade_batches.yaml
```

---

## Step 4: Configure Experiment Logging

### File: `src/open_data_pvnet/configs/PVNet_configs/logger/wandb.yaml`

1. Create a Weights & Biases account: https://wandb.ai/
2. Update the following fields:

```yaml
project: "GFS_TEST_RUN"
save_dir: "GFS_TEST_RUN"
```

**Purpose**: Enables experiment tracking, metrics logging, and model versioning in Weights & Biases.

---

## Step 5: Generate Training Samples

### 5.1 Download Data Locally (Recommended)

Download GFS and GSP data from AWS S3 to avoid slow streaming during sample generation:

```bash
# Download GFS NWP data (2023)
aws s3 sync s3://ocf-open-data-pvnet/data/gfs/v4/2023.zarr/ ./gfs_2023.zarr --no-sign-request

# Download GSP (PV generation) data (2023)
aws s3 sync s3://ocf-open-data-pvnet/data/uk/pvlive/v2/combined_2023_gsp.zarr ./gsp_2023.zarr --no-sign-request
```

**Storage Requirements**:
- GFS 2023 data: ~XX GB
- GSP 2023 data: ~XX GB

### 5.2 Update Data Configuration

### File: `src/open_data_pvnet/configs/PVNet_configs/datamodule/configuration/example_configuration.yaml`

Update the `zarr_path` attributes to point to local directories:

```yaml
input_data:
  gsp:
    zarr_path: "./gsp_2023.zarr"  # Change from S3 path
    # public: True  # Comment out this line

  nwp:
    gfs:
      zarr_path: "./gfs_2023.zarr"  # Change from S3 path
      # public: True  # Comment out this line
```

### 5.3 Update Streamed Batches Configuration

### File: `src/open_data_pvnet/configs/PVNet_configs/datamodule/streamed_batches.yaml`

Change line 5 from:
```yaml
configuration: null
```

To:
```yaml
configuration: <absolute_path_to_project>/src/open_data_pvnet/configs/PVNet_configs/datamodule/configuration/example_configuration.yaml
```

### 5.4 Run Sample Generation

```bash
# Activate virtual environment (if using one)
source ./venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate     # Windows

# Remove previous sample runs
rm -rf GFS_samples PLACEHOLDER

# Generate samples
python src/open_data_pvnet/scripts/save_samples.py
```

**Output**: Samples will be saved to the `GFS_samples/` directory with `train/` and `val/` subdirectories.

---

## Step 6: Train the Model

### 6.1 Update Config to Use Pre-made Batches

### File: `src/open_data_pvnet/configs/PVNet_configs/config.yaml`

Change line 8:
```yaml
- datamodule: streamed_batches.yaml
```

To:
```yaml
- datamodule: premade_batches.yaml
```

### 6.2 Run Training

```bash
python run.py
```

**What Happens**:
- Model loads pre-generated samples from `GFS_samples/`
- Training and validation loops execute
- Metrics logged to Weights & Biases
- Model checkpoints saved automatically

---

## Configuration Summary

### Key File Paths
| Configuration File | Purpose |
|-------------------|---------|
| `config.yaml` | Main training configuration, datamodule selection |
| `streamed_batches.yaml` | Sample generation settings |
| `premade_batches.yaml` | Training from pre-generated samples |
| `example_configuration.yaml` | Data source paths and parameters |
| `wandb.yaml` | Experiment logging settings |

### Workflow States

| Step | Active DataModule | Action |
|------|------------------|--------|
| Sample Generation | `streamed_batches.yaml` | `save_samples.py` |
| Model Training | `premade_batches.yaml` | `run.py` |

---

## Data Sources

### GFS (Global Forecast System)
- **Provider**: NOAA
- **Resolution**: 3-hour intervals
- **Channels**: 14 weather variables (radiation, cloud cover, temperature, wind, etc.)
- **Coverage**: 2023 dataset

### GSP (Grid Supply Points)
- **Provider**: PVLive
- **Resolution**: 30-minute intervals
- **Usage**: Target solar generation data for UK
- **Coverage**: 2023 dataset


---


**Thank you for joining us on this journey to advance solar forecasting and renewable energy solutions!** 🌞
