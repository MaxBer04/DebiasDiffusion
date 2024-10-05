"""
Public Data Downloader for DebiasDiffusion Project

This script provides functionality to download publicly available datasets and 
model weights for the DebiasDiffusion project. It's designed for users who want 
to reproduce experiments or use pre-trained models.

Usage:
    python tools/public_data_downloader.py <package> --output_dir <dir>

Packages:
    - full_package: Download all data (model weights and all experiment datasets)
    - model_data: Download only model weights (FDM_weights and h_space_classifiers)
    - experiment_5.1.3: Download data for experiment 5.1.3
    - experiment_5.4: Download data for experiment 5.4
    - experiment_6.1: Download data for experiment 6.1

Outputs:
    - Downloaded and extracted data in the specified output directory
    - Directory structure mimics the project's data organization

Note: This script does not require any credentials as it downloads from a public 
      read-only storage bucket.
"""

import os
import argparse
import requests
from tqdm import tqdm
import tarfile
from typing import Optional
from pathlib import Path

# Constants
BUCKET_URL = "https://pub-e4d660081d944b389609a3d747f5cf10.r2.dev"
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR.parent / "data"

def download_file(url: str, local_filename: Path) -> None:
    """Download a file from a given URL with a progress bar."""
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        total_size = int(r.headers.get('content-length', 0))
        with open(local_filename, 'wb') as f, tqdm(
            desc=local_filename.name,
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as progress_bar:
            for chunk in r.iter_content(chunk_size=8192):
                size = f.write(chunk)
                progress_bar.update(size)

def extract_dataset(compressed_file: Path, dataset_dir: Path) -> None:
    """Extract a compressed tar.gz file to the specified directory."""
    print(f"Extracting '{compressed_file}' to '{dataset_dir}'...")
    with tarfile.open(compressed_file, 'r:gz') as tar:
        total_members = len(tar.getmembers())
        with tqdm(total=total_members, unit='file', desc='Extracting') as pbar:
            for member in tar.getmembers():
                tar.extract(member, path=dataset_dir)
                pbar.update(1)
    print("Dataset extracted successfully.")
    compressed_file.unlink()  # Remove the compressed file after extraction

def download_and_extract(package_name: str, output_dir: Path) -> None:
    """Download and extract a specific data package."""
    url = f"{BUCKET_URL}/{package_name}.tar.gz"
    local_filename = output_dir / f"{package_name}.tar.gz"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading {package_name}...")
    download_file(url, local_filename)
    
    print(f"Extracting {package_name}...")
    extract_dataset(local_filename, output_dir)

def download_full_package(output_dir: Path) -> None:
    """Download and extract the full data package."""
    download_and_extract("full_package", output_dir)

def download_model_data(output_dir: Path) -> None:
    """Download and extract model data."""
    download_and_extract("model_data", output_dir / "model_data")

def download_experiment_data(experiment: str, output_dir: Path) -> None:
    """Download and extract data for a specific experiment."""
    download_and_extract(f"experiment_{experiment}", output_dir / "experiments")

def main() -> None:
    parser = argparse.ArgumentParser(description='Download DebiasDiffusion datasets')
    parser.add_argument('package', choices=['all', 'model_data', 'experiment_5.1.3', 'experiment_5.4', 'experiment_6.1'],
                        help='Package to download')
    parser.add_argument('--output_dir', type=Path, default=DEFAULT_OUTPUT_DIR,
                        help='Directory to save the downloaded data')
    
    args = parser.parse_args()
    
    if args.package == 'all':
        download_full_package(args.output_dir)
    elif args.package == 'model_data':
        download_model_data(args.output_dir)
    else:
        experiment = args.package.split('_')[1]
        download_experiment_data(experiment, args.output_dir)

    print(f"Data package '{args.package}' has been successfully downloaded and extracted to {args.output_dir}")

if __name__ == '__main__':
    main()