# Detailed Data Download Guide

This guide provides comprehensive instructions for downloading the necessary data to run experiments and use pre-trained models in the DebiasDiffusion project.

## Overview

The DebiasDiffusion project uses a public Cloudflare R2 bucket to distribute datasets and model weights. This ensures that all users have access to the same data, facilitating reproducibility and ease of use.

## Using the public_data_downloader.py Script

The `public_data_downloader.py` script in the `tools/` directory is the primary method for downloading project data.

### Basic Usage

To download all data (recommended for full reproducibility):
```
python tools/public_data_downloader.py all --output_dir ./data
```

To download only model weights:
```
python tools/public_data_downloader.py model_data --output_dir ./data
```

To download data for a specific experiment:
```
python tools/public_data_downloader.py experiment_5.4.1 --output_dir ./data
```

### Available Packages

- `full`: All data, including model weights and all experiment datasets
- `model_data`: Only model weights (FDM_weights and h_space_classifiers)
- `experiment_5.1.3`: Data for experiment 5.1.3
- `experiment_5.4`: Data for experiment 5.4
- `experiment_6.1`: Data for experiment 6.1

### Output Structure

The downloaded data will be automatically extracted to the specified output directory, maintaining the following structure:
```
data/
├── model_data/
│   ├── FDM_weights/
│   └── h_space_classifiers/
└── experiments/
├── section_5.1.3/
├── section_5.4/
└── section_6.1/
```

## Manual Download (Alternative Method)

If you prefer to manually download the data or are experiencing issues with the script, you can directly access the public R2 bucket at:
```
https://pub-e4d660081d944b389609a3d747f5cf10.r2.dev
```

You can use any S3-compatible client to download the data from this URL.

## Troubleshooting

If you encounter issues during the download process:

1. Ensure you have a stable internet connection.
2. Check that you have sufficient disk space in the target directory. The full package contains ~500 GB of data.
3. Verify that you're using the latest version of the `public_data_downloader.py` script from the repository.
4. If using a corporate or university network, ensure that access to the R2 bucket is not blocked by firewalls.

For persistent issues, please contact the author under [maber133@uni-duesseldorf.de](maber133@uni-duesseldorf.de).