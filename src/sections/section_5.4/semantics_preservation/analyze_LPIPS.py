"""
LPIPS Analysis for Semantic Preservation in DebiasDiffusion

This script analyzes the semantic preservation of generated images using LPIPS (Learned Perceptual Image Patch Similarity).
It compares generated images to the original model's outputs to measure perceptual similarity.

Usage:
    python src/sections/section_5.4/semantics_preservation/analyze_LPIPS.py [--args]

Arguments:
    --datasets_dir: Directory containing all datasets (default: BASE_DIR / "data/experiments/section_5.4.1/5.4.1_datasets")
    --original_dataset: Name of the original dataset (default: "outputs_original_NEW")
    --output_dir: Directory to save results (default: BASE_DIR / "results/section_5.4.1/LPIPS_results")
    --batch_size: Batch size for LPIPS processing (default: 256)
    --dataset_type: Type of dataset: 'occupation' or 'laion' (default: 'occupation')

Outputs:
    - CSV files with LPIPS scores for each dataset
    - Overall results CSV file
    - Console output with analysis summary
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import set_seed

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent.parent.parent.parent
sys.path.append(str(BASE_DIR))

from src.utils.lpips_utils import LPIPSEncoder

class ImagePairDataset(Dataset):
    def __init__(self, metadata, dataset_path, original_metadata, original_dataset_path, dataset_type):
        self.metadata = metadata
        self.dataset_path = Path(dataset_path)
        self.original_metadata = original_metadata
        self.original_dataset_path = Path(original_dataset_path)
        self.dataset_type = dataset_type
        self.mismatched_items = self.find_mismatched_items()

    def find_mismatched_items(self):
        mismatched = {}
        if self.dataset_type == 'occupation':
            for occupation in self.metadata['occupation'].unique():
                current_seeds = set(self.metadata[self.metadata['occupation'] == occupation]['seed'])
                original_seeds = set(self.original_metadata[self.original_metadata['occupation'] == occupation]['seed'])
                if current_seeds != original_seeds:
                    mismatched[occupation] = (list(current_seeds), list(original_seeds))
        else:  # LAION dataset
            for prompt in self.metadata['prompt'].unique():
                current_seeds = set(self.metadata[self.metadata['prompt'] == prompt]['seed'])
                original_seeds = set(self.original_metadata[self.original_metadata['prompt'] == prompt]['seed'])
                if current_seeds != original_seeds:
                    mismatched[prompt] = (list(current_seeds), list(original_seeds))
        return mismatched

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        if self.dataset_type == 'occupation':
            group = row['occupation']
            image_path = self.dataset_path / group / f"{row['seed']}.png"
            if group in self.mismatched_items:
                current_seeds, original_seeds = self.mismatched_items[group]
                current_index = current_seeds.index(row['seed'])
                original_seed = original_seeds[current_index % len(original_seeds)]
                original_image_path = self.original_dataset_path / group / f"{original_seed}.png"
            else:
                original_image_path = self.original_dataset_path / group / f"{row['seed']}.png"
        else:  # LAION dataset
            group = row['prompt']
            image_path = self.dataset_path / group / f"{row['seed']}.png"
            if group in self.mismatched_items:
                current_seeds, original_seeds = self.mismatched_items[group]
                current_index = current_seeds.index(row['seed'])
                original_seed = original_seeds[current_index % len(original_seeds)]
                original_image_path = self.original_dataset_path / group / f"{original_seed}.png"
            else:
                original_image_path = self.original_dataset_path / group / f"{row['seed']}.png"
        
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        if not original_image_path.exists():
            raise FileNotFoundError(f"Original image not found: {original_image_path}")
        
        return str(image_path), str(original_image_path), group, row['seed']

def custom_collate(batch):
    image_paths, original_image_paths, groups, seeds = zip(*batch)
    return list(image_paths), list(original_image_paths), list(groups), list(seeds)

def load_metadata(dataset_path, dataset_type):
    metadata_path = Path(dataset_path) / 'metadata.csv'
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    if dataset_type == 'occupation':
        return pd.read_csv(metadata_path, header=0, names=['occupation', 'seed', 'prompt'])
    else:  # LAION dataset
        return pd.read_csv(metadata_path, header=0, names=['seed', 'prompt'])

def preview_dataloader(dataloader, num_samples=5):
    print(f"\nPreview of {num_samples} samples from the DataLoader:")
    sample_iter = iter(dataloader)
    for i in range(num_samples):
        try:
            sample = next(sample_iter)
            print(f"\nSample {i + 1}:")
            print(f"Image paths: {sample[0][:2]}...")
            print(f"Original image paths: {sample[1][:2]}...")
            print(f"Groups: {sample[2][:2]}...")
            print(f"Seeds: {sample[3][:2]}...")
        except StopIteration:
            print("Reached end of dataloader")
            break
    print("\nEnd of preview\n")

def process_batch(encoder, image_paths, original_image_paths):
    images = [Image.open(path).convert("RGB") for path in image_paths]
    original_images = [Image.open(path).convert("RGB") for path in original_image_paths]
    lpips_similarities = encoder(images, original_images)
    return lpips_similarities

def run_analysis(args):
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
    set_seed(42)

    encoder = LPIPSEncoder().to(accelerator.device)
    encoder = accelerator.prepare(encoder)

    datasets = [d for d in os.listdir(args.datasets_dir) if os.path.isdir(os.path.join(args.datasets_dir, d))]
    original_dataset_path = Path(args.datasets_dir) / args.original_dataset
    original_metadata = load_metadata(original_dataset_path, args.dataset_type)

    results_dir = Path(args.output_dir)
    if accelerator.is_main_process:
        results_dir.mkdir(parents=True, exist_ok=True)

    overall_results = []

    for dataset in datasets:
        if dataset == args.original_dataset:
            continue  # Skip the original dataset

        dataset_path = Path(args.datasets_dir) / dataset
        try:
            metadata = load_metadata(dataset_path, args.dataset_type)
        except FileNotFoundError as e:
            accelerator.print(f"Error loading metadata for dataset {dataset}: {e}")
            continue
        
        image_dataset = ImagePairDataset(metadata, dataset_path, original_metadata, original_dataset_path, args.dataset_type)
        dataloader = DataLoader(image_dataset, batch_size=args.batch_size, collate_fn=custom_collate, num_workers=4)
        dataloader = accelerator.prepare(dataloader)

        if accelerator.is_main_process:
            preview_dataloader(dataloader)

        lpips_scores = []

        for batch in tqdm(dataloader, desc=f"Analyzing {dataset_path.name}", disable=not accelerator.is_main_process):
            image_paths, original_image_paths, groups, seeds = batch
            
            with accelerator.autocast():
                lpips_similarities = process_batch(encoder, image_paths, original_image_paths)

            lpips_similarities = accelerator.gather(lpips_similarities).cpu().numpy()

            for i in range(len(image_paths)):
                lpips_scores.append({
                    'group': groups[i],
                    'seed': seeds[i],
                    'lpips_similarity': lpips_similarities[i].item()
                })

        accelerator.wait_for_everyone()

        if accelerator.is_main_process:
            lpips_df = pd.DataFrame(lpips_scores)

            # Save and print LPIPS results
            save_results(lpips_df, results_dir / f"{dataset}_lpips.csv")
            accelerator.print(f"\n--- LPIPS Results for {dataset} ---")
            print_summary(lpips_df, 'lpips_similarity')

            overall_results.append({
                'dataset': dataset,
                'mean_lpips': lpips_df['lpips_similarity'].mean(),
                'std_lpips': lpips_df['lpips_similarity'].std(),
            })

            if image_dataset.mismatched_items:
                accelerator.print("\nItems with mismatched seeds:")
                for item in image_dataset.mismatched_items:
                    accelerator.print(f"- {item}")

    if accelerator.is_main_process:
        # Save and print overall results
        overall_df = pd.DataFrame(overall_results)
        save_results(overall_df, results_dir / "overall_lpips_results.csv")
        accelerator.print("\n--- Overall LPIPS Results ---")
        accelerator.print(overall_df.to_string(index=False))

def save_results(results, output_path):
    results.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")

def print_summary(results, metric_name):
    overall_mean = results[metric_name].mean()
    overall_std = results[metric_name].std()
    print(f"\nOverall mean {metric_name}: {overall_mean:.4f}")
    print(f"Overall standard deviation {metric_name}: {overall_std:.4f}")
    
    print(f"\nMean and standard deviation {metric_name} by group:")
    group_stats = results.groupby('group')[metric_name].agg(['mean', 'std'])
    for group, stats in group_stats.iterrows():
        print(f"{group}: Mean = {stats['mean']:.4f}, Std = {stats['std']:.4f}")

def main(args):
    run_analysis(args)

def parse_args():
    parser = argparse.ArgumentParser(description="Analyze datasets using LPIPS with Accelerate for multi-GPU support")
    parser.add_argument("--datasets_dir", type=str, default=BASE_DIR / "data/experiments/section_5.4.2/5.4.2_datasets", help="Directory containing all datasets")
    parser.add_argument("--original_dataset", type=str, default="SD", help="Name of the original dataset (only name not path)")
    parser.add_argument("--output_dir", type=str, default=BASE_DIR / "results/section_5.4.2/LPIPS_results", help="Directory to save results")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for LPIPS processing")
    parser.add_argument("--dataset_type", type=str, choices=['occupation', 'laion'], default='laion', help="Type of dataset: occupation (templated) or laion (non-templated)")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args)