"""
Fairness Discrepancy (FD) Analysis for Debiasing Methods

This script analyzes the Fairness Discrepancy (FD) metric for various debiasing methods
based on attribute classifications. It processes CSV files containing attribute probabilities
and generates comparative visualizations and statistics.

Usage:
    python src/sections/section_5.4/fairness/analyze_FD.py [--args]

Arguments:
    --input_dir: Directory containing input CSV files (default: BASE_DIR / "data/experiments/section_5.4.12/5.4.1_attribute_classification_results")
    --output_file: Output file path for results (default: BASE_DIR / "results/section_5.4.1/FD_values.csv")
    --attributes: Attributes to evaluate (default: gender race age)
    --target_gender: Target distribution for gender [male, female] (default: 0.5 0.5)
    --target_race: Target distribution for race [white, black, asian, indian] (default: 0.25 0.25 0.25 0.25)
    --target_age: Target distribution for age [young, old] (default: 0.75 0.25)

Outputs:
    - CSV file with FD results for each method and attribute
    - Text file with a tabular summary of results
    - Heatmap visualization of FD values
"""

import os
import pandas as pd
import numpy as np
import torch
import argparse
import json
from pathlib import Path
from tabulate import tabulate
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent.parent.parent.parent

def FD(probs: np.ndarray, p_tar: np.ndarray) -> float:
    """Calculate the Fairness Discrepancy metric."""
    if isinstance(p_tar, torch.Tensor):
        p_tar = p_tar.cpu().numpy()
    elif isinstance(p_tar, list):
        p_tar = np.array(p_tar)
    
    expected_distribution = np.mean(probs, axis=0)
    diff = expected_distribution - p_tar
    metric_value = np.linalg.norm(diff) ** 2
    
    return metric_value

def calculate_fd_for_dataset(df: pd.DataFrame, attribute: str, p_tar: np.ndarray) -> Tuple[float, float]:
    """Calculate overall FD and standard deviation for a dataset."""
    probs = np.array([json.loads(p) for p in df[f'{attribute}_probs']])
    
    overall_fd = np.round(FD(probs, p_tar), decimals=3)
    
    group_fds = df.groupby('group').apply(
        lambda x: FD(np.array([json.loads(p) for p in x[f'{attribute}_probs']]), p_tar)
    )
    
    fd_std = np.round(np.std(group_fds), decimals=3)
    
    return overall_fd, fd_std

def custom_sort_key(filename: str) -> Tuple[int, str]:
    """Custom sorting key for model filenames."""
    prefixes = ['SD_', 'FTF_', 'FD_', 'AS_', 'DD_']
    for i, prefix in enumerate(prefixes):
        if filename.startswith(prefix):
            return (i, filename)
    return (len(prefixes), filename)

def process_datasets(input_dir: Path, output_file: Path, attributes: List[str], target_distributions: Dict[str, np.ndarray]):
    """Process all datasets and calculate FD metrics."""
    results = []
    
    for file in sorted(os.listdir(input_dir), key=custom_sort_key):
        if file.endswith('.csv'):
            file_path = input_dir / file
            df = pd.read_csv(file_path)
            
            dataset_results = {'dataset': file}
            
            for attr in attributes:
                if attr in target_distributions:
                    fd_value, fd_std = calculate_fd_for_dataset(df, attr, target_distributions[attr])
                    dataset_results[f'{attr}_FD'] = fd_value
                    dataset_results[f'{attr}_FD_std'] = fd_std
            
            results.append(dataset_results)
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_file, index=False, float_format='%.3f')
    print(f"Results saved to {output_file}")
    
    # Create a table output
    table = tabulate(results_df, headers='keys', tablefmt='grid', floatfmt=".3f")
    table_file = output_file.with_suffix('.txt')
    with open(table_file, 'w') as f:
        f.write(table)
    print(f"Table saved to {table_file}")
    print("\nSummary Table:")
    print(table)
    
    # Create a vertically oriented heatmap
    plt.figure(figsize=(max(12, len(results_df.columns) * 0.8), 18))
    sns.heatmap(results_df.set_index('dataset').T, annot=True, cmap='YlOrRd', fmt='.3f', cbar_kws={'label': 'Fairness Discrepancy'})
    plt.title('Fairness Discrepancy Heatmap')
    plt.xlabel('Datasets')
    plt.ylabel('Attributes')
    plt.tight_layout()
    heatmap_file = output_file.with_suffix('.png')
    plt.savefig(heatmap_file, dpi=300, bbox_inches='tight')
    print(f"Heatmap saved to {heatmap_file}")

def main(args: argparse.Namespace):
    input_dir = Path(args.input_dir)
    output_file = Path(args.output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    target_distributions = {
        'gender': np.array(args.target_gender),
        'race': np.array(args.target_race),
        'age': np.array(args.target_age)
    }
    
    process_datasets(input_dir, output_file, args.attributes, target_distributions)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calculate Fairness Discrepancy for multiple datasets.")
    parser.add_argument("--input_dir", type=str, default=BASE_DIR / "data/experiments/section_5.4.1/5.4.1_attribute_classification_results", 
                        help="Directory containing input CSV files")
    parser.add_argument("--output_file", type=str, default=BASE_DIR / "results/section_5.4.1/FD_values.csv", 
                        help="Output file path for results")
    parser.add_argument("--attributes", nargs='+', default=['gender', 'race', 'age'], 
                        help="Attributes to evaluate (default: gender race age)")
    parser.add_argument("--target_gender", nargs=2, type=float, default=[0.5, 0.5], 
                        help="Target distribution for gender (male, female)")
    parser.add_argument("--target_race", nargs=4, type=float, default=[0.25, 0.25, 0.25, 0.25], 
                        help="Target distribution for race (white, black, asian, indian)")
    parser.add_argument("--target_age", nargs=2, type=float, default=[0.75, 0.25], 
                        help="Target distribution for age (young, old)")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args)