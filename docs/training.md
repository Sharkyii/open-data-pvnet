## Training Pipeline

This guide explains how to train PVNet using the open-data-pvnet pipeline. It covers:
- Configuration structure and best practices
- The two supported training workflows (with clear recommendations)
- Common pitfalls and how to resolve them

---

## 1. Configuration Basics

All configuration files live in a structured directory:

```
open-data-pvnet/
└── src/
    └── open_data_pvnet/
        └── configs/
            └── PVNet_configs/
```

The main configuration file is `config.yaml`, which tells the system which sub-configurations to use. Think of it as the master control panel that connects everything together.

---

## 2. Understanding the Key Configuration Parts

### Trainer Configuration
Controls how your model trains:
- GPU/CPU usage
- Training duration
- Precision settings

### Model Configuration
Defines your model's architecture:
- Which encoders to use (GFS, satellite, etc.)
- Forecast horizon
- Optimizer settings

### Data Configuration (Most Important!)
This determines how your data is loaded. PVNet supports two approaches:
- **Streamed batches**: Directly from Zarr files
- **Premade batches**: From pre-generated samples

You must choose only one approach at a time - mixing them will cause errors.

---

## 3. Two Ways to Train PVNet

### Method 1: Streamed Batches (Direct Zarr Loading)

This approach loads data directly from Zarr files during training.

**When to use it:**
- You have sufficient disk space and bandwidth
- You don't want to pre-generate samples

**How to set it up:**
In your `config.yaml`, ensure you have:
```yaml
- datamodule: streamed_batches.yaml
```

**Important settings:**
```yaml
_target_: pvnet.data.DataModule
configuration: /ABSOLUTE/PATH/example_configuration.yaml
batch_size: 8
num_workers: 2
prefetch_factor: 2
train_period:
  - null
  - "2023-06-30"
val_period:
  - "2023-07-01"
  - "2023-12-31"
```

**What to avoid:**
Don't include `sample_output_dir`, `num_train_samples`, or `num_val_samples` in your streamed configuration, as these will cause errors.

**To start training:**
```bash
python run.py experiment=example_simple
```

### Method 2: Premade Batches (Recommended for Beginners)

This approach uses pre-generated samples, making training more stable and reproducible.

**When to use it:**
- You want consistent, reproducible results
- You want faster iteration during development
- You've encountered issues with Zarr or storage

**Step 1: Generate samples**
Navigate to `open-data-pvnet/src/open_data_pvnet/scripts` and run:
```bash
python save_samples.py \
  +datamodule.sample_output_dir="GFS_samples" \
  +datamodule.num_train_samples=10 \
  +datamodule.num_val_samples=2 \
  datamodule.num_workers=2
```

This creates a directory with your samples:
```
scripts/
└── GFS_samples/
    ├── data_configuration.yaml
    ├── train/
    └── val/
```

**Step 2: Switch to premade batches**
In your `config.yaml`, change to:
```yaml
- datamodule: premade_batches.yaml
```

**Step 3: Configure the premade batches**
```yaml
_target_: pvnet.data.DataModule
sample_output_dir: /ABSOLUTE/PATH/TO/GFS_samples
batch_size: 8
num_workers: 2
prefetch_factor: 2
```

**Important:** Always use absolute paths! Hydra changes the working directory at runtime, so relative paths will break.

**Step 4: Train**
```bash
python run.py experiment=example_simple
```

---

## 4. Data Configuration Best Practices

When setting up your data configuration (`example_configuration.yaml`), we strongly recommend:

1. Download data locally
2. Point to local Zarr paths
3. Set `public: false` for local data

Example configuration:
```yaml
gsp:
  zarr_path: C:/data/gsp/combined_2023_gsp.zarr
  public: false

nwp:
  gfs:
    zarr_path: C:/data/nwp/nwp_gfs.zarr
    public: false
```

### Important Note About Local Data

If you're using local data, make sure to set `public: false` in your configuration. When `public: true` is set with a local path, you'll encounter this error:

```
ValueError: storage_options passed with non-fsspec path
```

For example, if your configuration looks like this:
```yaml
gsp:
  zarr_path: "s3://ocf-open-data-pvnet/data/uk/pvlive/v2/combined_2023_gsp.zarr"
  # ... other settings ...
  public: True
```

And you're trying to use local data, change it to:
```yaml
gsp:
  zarr_path: "/path/to/your/local/data/combined_2023_gsp.zarr"
  # ... other settings ...
  public: False
```

---

## 5. Common Hydra Issues

### Hydra Override Error

If you encounter an override error when using `experiment_simple`, it might be due to conflicting override declarations. The `experiment_simple` configuration includes these overrides:

```yaml
defaults:
  - override /trainer: default.yaml
  - override /model: multimodal.yaml
  - override /datamodule: premade_samples.yaml
  - override /callbacks: default.yaml
  - override /logger: wandb.yaml
  - override /hydra: default.yaml
```

If you see errors like "Could not override 'hydra'", you may need to temporarily comment out the hydra override in your config file:

```yaml
# - override /hydra: default.yaml
```

### Working Directory Changes

Remember that Hydra runs experiments inside timestamped directories (e.g., `outputs/YYYY-MM-DD/HH-MM-SS/`). This is why absolute paths are essential - relative paths will break when the working directory changes.

---

## 6. Quick Comparison: Streamed vs. Premade

| Feature                    | Streamed | Premade |
| -------------------------- | -------- | ------- |
| Reads Zarr at runtime      | Yes      | No      |
| Needs pre-generated samples| No       | Yes     |
| Sensitive to storage flags | Yes      | No      |
| Recommended for beginners  | No       | Yes     |

---

## 7. External Requirements

### Google Cloud CLI

Even if you're using local data, some metadata utilities require the Google Cloud CLI. Install it from:
https://cloud.google.com/sdk/docs/install

Then authenticate:
```bash
gcloud auth application-default login
```
