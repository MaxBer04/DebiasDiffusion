# DebiasDiffusion Tools

This directory contains utility scripts for managing data and running experiments in the DebiasDiffusion project.

## Scripts

### public_data_downloader.py

This script allows users to download publicly available datasets and model weights required for reproducing experiments or using pre-trained models.

Usage:

  ```
  python public_data_downloader.py <package> --output_dir <dir>
  ```


Packages:
- `full`: Download all data (model weights and all experiment datasets)
- `model_data`: Download only model weights (FDM_weights and h_space_classifiers)
- `experiment_5.1.3`: Download data for experiment 5.1.3
- `experiment_5.4`: Download data for experiment 5.4
- `experiment_6.1`: Download data for experiment 6.1

This script downloads data from a public Cloudflare R2 bucket, which uses the S3 interface. The bucket is maintained by the project author and provides safe, read-only access to necessary files for experiments and models.

### s3_utils.py

This script is for internal use by project maintainers. It provides functionality to upload and download data to/from Amazon S3 and Cloudflare R2 storage.

Usage:

  ```
  python s3_utils.py <action> --dataset_type <type> --storage_type <s3/r2>
  ```

Actions:
- `upload`: Upload datasets or model weights
- `download`: Download datasets or model weights

Note: This script requires appropriate AWS or Cloudflare R2 credentials to be set as environment variables. It should not be used by general users of the project.

## Cloudflare R2

Cloudflare R2 is an object storage service that is compatible with Amazon S3's API. In this project, we use R2 to host our public datasets and model weights. R2 provides:

- S3-compatible API, allowing us to use familiar tools and libraries
- Secure, read-only access to our public data
- Cost-effective storage and bandwidth for large datasets

The public R2 bucket used in this project is maintained by the project author and ensures that all users have access to the necessary data for reproducing experiments and using the DebiasDiffusion models.