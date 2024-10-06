"""
CLIP Analysis for Semantic Preservation in DebiasDiffusion

This script analyzes the semantic preservation of generated images using CLIP embeddings.
It compares generated images to their original prompts and to the original model's outputs.

Usage:
    python src/sections/section_5.4/semantics_preservation/analyze_CLIP.py [--args]

Arguments:
    --datasets_dir: Directory containing all datasets (default: BASE_DIR / "data/experiments/section_5.4.2/5.4.2_datasets")
    --original_dataset: Name of the original dataset (default: "SD")
    --output_dir: Directory to save results (default: BASE_DIR / "results/section_5.4.2/CLIP_results")
    --batch_size: Batch size for CLIP processing (default: 1024)
    --dataset_type: Type of dataset: 'occupation' or 'laion' (default: 'laion')

Outputs:
    - CSV files with CLIP scores for each dataset
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
from typing import Any, Callable, Dict, List, Optional, Union, Tuple
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from accelerate import Accelerator
from accelerate.utils import set_seed

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.append(str(BASE_DIR))

from src.utils.clip_utils import CLIPEncoder

class ImageTextDataset(Dataset):
    def __init__(self, metadata, dataset_path, original_metadata, original_dataset_path, dataset_type, is_original=False):
        self.metadata = metadata
        self.dataset_path = Path(dataset_path)
        self.original_metadata = original_metadata
        self.original_dataset_path = Path(original_dataset_path)
        self.dataset_type = dataset_type
        self.is_original = is_original
        self.mismatched_items = self.find_mismatched_items() if not is_original else {}

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
        else:  # LAION dataset
            group = row['prompt']

        image_path = self.dataset_path / group / f"{row['seed']}.png"
        
        if not self.is_original:
            if group in self.mismatched_items:
                current_seeds, original_seeds = self.mismatched_items[group]
                current_index = current_seeds.index(row['seed'])
                original_seed = original_seeds[current_index % len(original_seeds)]
                original_image_path = self.original_dataset_path / group / f"{original_seed}.png"
            else:
                original_image_path = self.original_dataset_path / group / f"{row['seed']}.png"
        else:
            original_image_path = None

        return str(image_path), row['prompt'], str(original_image_path) if original_image_path else None, group, row['seed']

def custom_collate(batch):
    image_paths, prompts, original_image_paths, groups, seeds = zip(*batch)
    return list(image_paths), list(prompts), list(original_image_paths), list(groups), list(seeds)

def load_metadata(dataset_path: Path, dataset_type: str) -> pd.DataFrame:
    metadata_path = dataset_path / 'metadata.csv'
    if dataset_type == 'occupation':
        return pd.read_csv(metadata_path, header=0, names=['occupation', 'seed', 'prompt'])
    else:  # LAION dataset
        return pd.read_csv(metadata_path, header=0, names=['seed', 'prompt'])

def process_batch(encoder: CLIPEncoder, image_paths: List[str], prompts: List[str], original_image_paths: List[str] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    images = [Image.open(path).convert("RGB") for path in image_paths]
    prompt_similarities = encoder(images, prompts)
    
    if original_image_paths and all(path is not None for path in original_image_paths):
        original_images = [Image.open(path).convert("RGB") for path in original_image_paths]
        image_similarities = encoder(images, original_images)
    else:
        image_similarities = None
    
    return prompt_similarities, image_similarities

def run_analysis(args: argparse.Namespace) -> None:
    accelerator = Accelerator()
    set_seed(42)

    encoder = CLIPEncoder().to(accelerator.device)
    encoder = accelerator.prepare(encoder)

    datasets = [d for d in os.listdir(args.datasets_dir) if os.path.isdir(os.path.join(args.datasets_dir, d))]
    original_dataset_path = Path(args.datasets_dir) / args.original_dataset
    original_metadata = load_metadata(original_dataset_path, args.dataset_type)

    results_dir = Path(args.output_dir)
    if accelerator.is_main_process:
        results_dir.mkdir(parents=True, exist_ok=True)

    overall_results = []

    # Always include the original dataset for text similarity
    datasets = [args.original_dataset] + [d for d in datasets if d != args.original_dataset]

    for dataset in datasets:
        dataset_path = Path(args.datasets_dir) / dataset
        metadata = load_metadata(dataset_path, args.dataset_type)
        
        is_original = (dataset == args.original_dataset)
        image_dataset = ImageTextDataset(metadata, dataset_path, original_metadata, original_dataset_path, args.dataset_type, is_original)
        dataloader = DataLoader(image_dataset, batch_size=args.batch_size, collate_fn=custom_collate, num_workers=4)
        dataloader = accelerator.prepare(dataloader)

        prompt_adherence_scores = []
        image_similarities = []

        for batch in tqdm(dataloader, desc=f"Analyzing {dataset_path.name}", disable=not accelerator.is_main_process):
            image_paths, prompts, original_image_paths, groups, seeds = batch
            
            prompt_similarities, image_similarity_batch = process_batch(encoder, image_paths, prompts, original_image_paths if not is_original else None)

            prompt_similarities = accelerator.gather(prompt_similarities).cpu().numpy()
            if image_similarity_batch is not None:
                image_similarity_batch = accelerator.gather(image_similarity_batch).cpu().numpy()

            for i in range(len(image_paths)):
                prompt_adherence_scores.append({
                    'group': groups[i],
                    'seed': seeds[i],
                    'prompt_adherence': prompt_similarities[i].item()
                })

                if image_similarity_batch is not None:
                    image_similarities.append({
                        'group': groups[i],
                        'seed': seeds[i],
                        'image_similarity': image_similarity_batch[i].item()
                    })

        accelerator.wait_for_everyone()

        if accelerator.is_main_process:
            prompt_adherence = pd.DataFrame(prompt_adherence_scores)
            save_results(prompt_adherence, results_dir / f"{dataset}_prompt_adherence.csv")

            mean_prompt_adherence = prompt_adherence['prompt_adherence'].mean()
            std_prompt_adherence = prompt_adherence['prompt_adherence'].std()

            result = {
                'dataset': dataset,
                'mean_prompt_adherence': mean_prompt_adherence,
                'std_prompt_adherence': std_prompt_adherence,
            }

            if not is_original:
                image_similarity = pd.DataFrame(image_similarities)
                save_results(image_similarity, results_dir / f"{dataset}_image_similarity.csv")

                mean_image_similarity = image_similarity['image_similarity'].mean()
                std_image_similarity = image_similarity['image_similarity'].std()

                result.update({
                    'mean_image_similarity': mean_image_similarity,
                    'std_image_similarity': std_image_similarity
                })

            overall_results.append(result)

            print(f"\n--- Summary for {dataset} ---")
            print(f"Mean Prompt Adherence: {mean_prompt_adherence:.4f}")
            print(f"Std Prompt Adherence: {std_prompt_adherence:.4f}")
            if not is_original:
                print(f"Mean Image Similarity: {mean_image_similarity:.4f}")
                print(f"Std Image Similarity: {std_image_similarity:.4f}")

            if not is_original:
                if args.dataset_type == 'occupation':
                    print(f"\nOccupations with mismatched seeds in {dataset}:")
                    for occupation in image_dataset.mismatched_items:
                        print(f"- {occupation}")
                else:  # LAION dataset
                    print(f"\nPrompts with mismatched seeds in {dataset}:")
                    for prompt in image_dataset.mismatched_items:
                        print(f"- {prompt}")

    if accelerator.is_main_process:
        overall_df = pd.DataFrame(overall_results)
        save_results(overall_df, results_dir / "overall_results.csv")
        accelerator.print("\n--- Overall Results ---")
        accelerator.print(overall_df.to_string(index=False))

def save_results(results: pd.DataFrame, output_path: Path) -> None:
    results.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")

def main(args: argparse.Namespace) -> None:
    run_analysis(args)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze datasets using CLIP with Accelerate for multi-GPU support")
    parser.add_argument("--datasets_dir", type=str, default=BASE_DIR / "data/experiments/section_5.4.2/5.4.2_datasets", help="Directory containing all datasets")
    parser.add_argument("--original_dataset", type=str, default="SD", help="Name of the original dataset")
    parser.add_argument("--output_dir", type=str, default=BASE_DIR / "results/section_5.4.2/CLIP_results", help="Directory to save results")
    parser.add_argument("--batch_size", type=int, default=1024, help="Batch size for CLIP processing")
    parser.add_argument("--dataset_type", type=str, choices=['occupation', 'laion'], default='laion', help="Type of dataset: occupation (templated) or laion (non-templated)")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args)