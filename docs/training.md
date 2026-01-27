## Training Pipeline


## 1. Configuration Basics

All configuration files are under PVVet_configs

```
open-data-pvnet/
└── src/
    └── open_data_pvnet/
        └── configs/
            └── PVNet_configs/
```

The main configuration file is `config.yaml`, which tells the system which sub-configurations to use.

---

## 2. Understanding the Key Configuration Parts

### Trainer Configuration

* Controls how your model trains

### Model Configuration

* Defines your model's architecture

### Data Configuration (Most Important!)

This determines how your data is loaded. PVNet supports two approaches:

* **Streamed batches**: Directly from Zarr files
* **Premade batches**: From pre-generated samples

---

## 3. Two Ways to Train PVNet

### Method 1: Streamed Batches (Direct Zarr Loading)

This approach loads data directly from Zarr files during training.

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
---

### Method 2: Premade Batches (Recommended)

This approach uses pre-generated samples, making training more stable and reproducible.


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

Before running the samples, ensure the following configuration updates are made.

Go to `src/open_data_pvnet/configs/PVNet_configs/datamodule/streamed_batches.yaml`
Change values if desired (increase at your discretion):

```
num_train_samples: 5
num_val_samples: 5
```

Update `src/open_data_pvnet/configs/PVNet_configs/datamodule/premade_batches.yaml`
Change this line to:

```
configuration: <your_directory...open-data-pvnet/src/open_data_pvnet/configs/PVNet_configs/datamodule/configuration/example_configuration.yaml>
```

Update `src/open_data_pvnet/configs/PVNet_configs/config.yaml`
Change the line to:

```
- datamodule: premade_batches.yaml
```

---

**Step 2: Configure Weights & Biases**

Create a Weights & Biases account:
[https://wandb.ai/](https://wandb.ai/)

---

**Step 3: Prepare data locally (recommended)**

We recommend you save the samples locally for faster processing.

In your main `open-data-pvnet` directory, run the following command (assumes aws cli is installed locally):

```bash
aws s3 sync s3://ocf-open-data-pvnet/data/gfs/v4/2023.zarr/ ./gfs_2023.zarr --no-sign-request
aws s3 sync s3://ocf-open-data-pvnet/data/uk/pvlive/v2/combined_2023_gsp.zarr ./gsp_2023.zarr --no-sign-request
```

Change the `example_configuration.yaml` `zarr_path` attributes to the local paths you created above.

Comment out both of these lines:

```
public: True
```

If you are going to use the actual s3 buckets then leave alone however this may be really slow.

In `streamed_batches.yaml` change this line:

```
configuration: null
```

to your actual path of the `example_configuration.yaml` file.

---

**Step 4: Train**

Go to `config.yaml` and ensure:

```
- datamodule: premade_batches.yaml
```

Then run:

```bash
python run.py
```

---

## 4. Data Configuration 

When setting up your data configuration (`example_configuration.yaml`):

1. Set `public: false` for local data

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

---

## 7. External Requirements

### Google Cloud CLI

Even if you're using local data, some metadata utilities require the Google Cloud CLI. Install it from:
https://cloud.google.com/sdk/docs/install

Then authenticate:
```bash
gcloud auth application-default login
```

Thank you for joining us on this journey to advance solar forecasting and renewable energy solutions!

