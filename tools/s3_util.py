"""
S3 and R2 Storage Utility for DebiasDiffusion Project

This script provides functionality to upload and download data to/from Amazon S3 
and Cloudflare R2 storage. It's designed for internal use by project maintainers 
to manage datasets and model weights.

Usage:
    python tools/s3_utils.py upload --dataset_type <type> --storage_type <s3/r2>
    python tools/s3_utils.py download --dataset_type <type> --storage_type <s3/r2>

Functionality:
    - Upload datasets and model weights to S3 or R2 storage
    - Download datasets and model weights from S3 or R2 storage
    - Compress and decompress data for efficient storage and transfer
    - Support for both individual dataset uploads/downloads and bulk operations
    
Dataset types:
    These include either of the following:
    - full: All data for all experiments and models
    - model_data: All data for the models
    - experiment_5.1.3/5.4.1/5.4.2/5.4.3 or experiment_6.1.1/6.1.2: Data for the experiments in the respective sections

Outputs:
    - For uploads: Compressed data uploaded to specified S3 or R2 bucket
    - For downloads: Data downloaded and extracted to appropriate local directories

Note: This script requires appropriate AWS or Cloudflare R2 credentials to be set 
      as environment variables.
"""

import os
import argparse
import shutil
import boto3
from tqdm import tqdm
import tarfile
from typing import Optional, Literal
from pathlib import Path

# Constants
SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent
DATA_DIR = BASE_DIR / "data"
COMPRESSION_FORMAT = "tar"

def get_storage_client(storage_type: Literal['s3', 'r2'], 
                       access_key_id: str, 
                       secret_access_key: str, 
                       endpoint_url: Optional[str] = None) -> boto3.client:
    """Initialize and return a storage client for S3 or R2."""
    if storage_type == 's3':
        return boto3.client('s3', aws_access_key_id=access_key_id, aws_secret_access_key=secret_access_key)
    elif storage_type == 'r2':
        return boto3.client('s3', endpoint_url=endpoint_url,
                            aws_access_key_id=access_key_id,
                            aws_secret_access_key=secret_access_key)
    else:
        raise ValueError(f"Unsupported storage type: {storage_type}")

def compress_directory(source_dir: Path, output_file: Path) -> None:
    """Compress a directory into a tar archive with progress bar."""
    print(f"Compressing {source_dir}...")
    total_size = sum(f.stat().st_size for f in source_dir.glob('**/*') if f.is_file())
    with tarfile.open(output_file, f"w:{COMPRESSION_FORMAT}") as tar:
        with tqdm(total=total_size, unit='B', unit_scale=True, desc="Compressing") as pbar:
            for file_path in source_dir.glob('**/*'):
                if file_path.is_file():
                    tar.add(file_path, arcname=file_path.relative_to(source_dir.parent))
                    pbar.update(file_path.stat().st_size)

def decompress_archive(archive_path: Path, output_dir: Path) -> None:
    """Decompress a tar archive to the specified directory with progress bar."""
    print(f"Decompressing {archive_path}...")
    with tarfile.open(archive_path, f"r:{COMPRESSION_FORMAT}") as tar:
        members = tar.getmembers()
        with tqdm(total=len(members), desc="Decompressing") as pbar:
            for member in members:
                tar.extract(member, path=output_dir)
                pbar.update(1)

def upload_file(client: boto3.client, file_path: Path, bucket: str, object_name: Optional[str] = None) -> None:
    """Upload a file to S3 or R2 storage."""
    if object_name is None:
        object_name = file_path.name

    file_size = file_path.stat().st_size
    with tqdm(total=file_size, unit='B', unit_scale=True, desc=f"Uploading {file_path.name}") as pbar:
        client.upload_file(
            str(file_path), bucket, object_name,
            Callback=lambda bytes_transferred: pbar.update(bytes_transferred)
        )

def download_file(client: boto3.client, bucket: str, object_name: str, file_path: Path) -> None:
    """Download a file from S3 or R2 storage."""
    print(f"HEREEEEE: {object_name}")
    file_size = client.head_object(Bucket=bucket, Key=object_name)['ContentLength']
    with tqdm(total=file_size, unit='B', unit_scale=True, desc=f"Downloading {object_name}") as pbar:
        client.download_file(
            bucket, object_name, str(file_path),
            Callback=lambda bytes_transferred: pbar.update(bytes_transferred)
        )

def upload_dataset(storage_client: boto3.client, dataset_type: str, bucket: str) -> None:
    """Compress and upload a specific dataset or all datasets."""
    if dataset_type == 'all':
        # Process all subdirectories without recursion
        for subdir in DATA_DIR.iterdir():
            if subdir.is_dir():
                if subdir.name == "experiments":
                    for exp_subdir in subdir.iterdir():
                        if exp_subdir.is_dir():
                            process_single_upload(storage_client, exp_subdir.name, bucket)
                else:
                    process_single_upload(storage_client, subdir.name, bucket)
    else:
        process_single_upload(storage_client, dataset_type, bucket)

def process_single_upload(storage_client: boto3.client, dataset_type: str, bucket: str) -> None:
    """Process the upload for a single dataset."""
    if dataset_type.startswith("section_"):
        source_dir = DATA_DIR / "experiments" / dataset_type
    else:
        source_dir = DATA_DIR / dataset_type
    
    if not source_dir.exists():
        print(f"Dataset directory {source_dir} does not exist.")
        return

    compressed_file = DATA_DIR / f"{dataset_type}.{COMPRESSION_FORMAT}"
    compress_directory(source_dir, compressed_file)
    upload_file(storage_client, compressed_file, bucket)
    compressed_file.unlink()  # Remove the local compressed file after upload
    print(f"Uploaded {dataset_type}")

def download_dataset(storage_client: boto3.client, dataset_type: str, bucket: str) -> None:
    """Download and decompress a specific dataset or all datasets."""
    if dataset_type == 'all':
        for obj in storage_client.list_objects(Bucket=bucket)['Contents']:
            if obj['Key'].endswith(f".{COMPRESSION_FORMAT}"):
                download_dataset(storage_client, obj['Key'].split(f'.{COMPRESSION_FORMAT}')[0], bucket)
    else:
        object_name = f"{dataset_type}.{COMPRESSION_FORMAT}"
        local_file = DATA_DIR / object_name
        download_file(storage_client, bucket, object_name, local_file)
        
        if dataset_type.startswith("section_"):
            output_dir = DATA_DIR / "experiments"
            output_dir.mkdir(exist_ok=True)
            decompress_archive(local_file, output_dir)
            extracted_dir = output_dir / dataset_type
        else:
            decompress_archive(local_file, DATA_DIR)
        
        local_file.unlink()  # Remove the local compressed file after extraction
        print(f"Downloaded and extracted {dataset_type}")

def main() -> None:
    parser = argparse.ArgumentParser(description="S3 and R2 Storage Utility for DebiasDiffusion Project")
    parser.add_argument('action', choices=['upload', 'download'], help="Action to perform")
    parser.add_argument('--dataset_type', default='full', help="Type of dataset to process (default: full)")
    parser.add_argument('--bucket', required=True, help="Name of the S3 or R2 bucket")
    parser.add_argument('--storage_type', choices=['s3', 'r2'], required=True, help="Storage type (S3 or R2)")
    parser.add_argument('--endpoint_url', help="Endpoint URL for R2 storage")
    args = parser.parse_args()

    access_key_id = os.environ.get('AWS_ACCESS_KEY_ID')
    secret_access_key = os.environ.get('AWS_SECRET_ACCESS_KEY')

    if not access_key_id or not secret_access_key:
        raise ValueError("AWS credentials not found in environment variables.")
    
    os.makedirs(DATA_DIR, exist_ok=True)

    storage_client = get_storage_client(args.storage_type, access_key_id, secret_access_key, args.endpoint_url)

    if args.action == 'upload':
        upload_dataset(storage_client, args.dataset_type, args.bucket)
    elif args.action == 'download':
        download_dataset(storage_client, args.dataset_type, args.bucket)

if __name__ == "__main__":
    main()