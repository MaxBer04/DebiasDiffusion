"""
FairFace Dataset Analysis for DebiasDiffusion

This script analyzes the FairFace dataset, which is used for evaluating fairness
in the DebiasDiffusion project. It provides statistics and visualizations of the
dataset's demographics.

Usage:
    python src/sections/section_5.4/quality/fairface_analysis.py [--args]

Arguments:
    --dataset: Name of the dataset on Hugging Face (default: "HuggingFaceM4/FairFace")
    --config: Configuration of the dataset (choices: '0.25', '1.25', default: '1.25')
    --split: Dataset split to use (default: "train")
    --category: Categorical column to visualize distribution (default: "gender")
    --output_dir: Directory to save results (default: BASE_DIR / "results/section_5.4/fairface_analysis")

Outputs:
    - Console output with dataset statistics
    - Distribution plot of the specified category (saved as PNG in the output directory)
"""

import argparse
import os
from typing import Dict, Any

from datasets import load_dataset
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent.parent.parent.parent.parent

def download_dataset(dataset_name: str, config: str, split: str) -> Any:
    """Download the specified dataset from Hugging Face with the given configuration."""
    return load_dataset(dataset_name, config, split=split)

def analyze_dataset_structure(dataset: Any) -> None:
    """Analyze the structure of the dataset."""
    print(f"Dataset info:\n{dataset}")
    print(f"\nFeatures:\n{dataset.features}")
    print(f"\nNumber of rows: {len(dataset)}")

def analyze_column_statistics(dataset: Any) -> None:
    """Analyze statistics for each column in the dataset."""
    df = dataset.to_pandas()
    print("\nColumn statistics:")
    print(df.describe())

def visualize_category_distribution(dataset: Any, category: str, output_dir: Path) -> None:
    """Visualize the distribution of a categorical column."""
    df = dataset.to_pandas()
    value_counts = df[category].value_counts()
    
    plt.figure(figsize=(10, 6))
    value_counts.plot(kind='bar')
    plt.title(f'Distribution of {category}')
    plt.xlabel(category)
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    output_file = output_dir / f'{category}_distribution.png'
    plt.savefig(output_file)
    print(f"Distribution plot saved as {output_file}")

def main(args: argparse.Namespace) -> None:
    """Main function to execute the FairFace dataset analysis."""
    dataset = download_dataset(args.dataset, args.config, args.split)
    
    analyze_dataset_structure(dataset)
    analyze_column_statistics(dataset)
    
    if args.category:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        visualize_category_distribution(dataset, args.category, args.output_dir)

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Download and analyze a Hugging Face dataset.")
    parser.add_argument("--dataset", type=str, default="HuggingFaceM4/FairFace", help="Name of the dataset on Hugging Face")
    parser.add_argument("--config", type=str, choices=['0.25', '1.25'], default='1.25', help="Configuration of the dataset")
    parser.add_argument("--split", type=str, default="train", help="Dataset split to use (e.g., 'train', 'test')")
    parser.add_argument("--category", type=str, default="gender", help="Categorical column to visualize distribution")
    parser.add_argument("--output_dir", type=Path, default=BASE_DIR / "results/section_5.4/fairface_example", 
                        help="Directory to save results")
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args)