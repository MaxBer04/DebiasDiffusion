"""
Performance Analysis for Debiasing Methods

This script analyzes the time and memory usage of various debiasing methods for text-to-image diffusion models.
It processes performance statistics generated during image creation and produces comparative visualizations.

Usage:
    python src/sections/section_5.4/analyze_time_and_memory.py [--args]

Arguments:
    --input_dir: Directory containing CSV files with performance data (default: data/datasets)
    --output_dir: Directory to save results (default: BASE_DIR / "data/experiments/section_5.4.1/5.4.1_datasets")
    --models: List of model prefixes to analyze (default: SD, DD, FD, FDM, AS)
    --title_size: Font size for plot titles (default: 16)
    --label_size: Font size for axis labels (default: 14)
    --tick_size: Font size for tick labels (default: 12)
    --legend_size: Font size for legend text (default: 14)
    --legend_title_size: Font size for legend title (default: 14)
    --no_titles: Flag to omit titles from plots (default: True)
    --no_svg: Flag to not save plots as SVG (default: False)

Outputs:
    - CSV file with aggregated performance metrics
    - PNG and optionally SVG plots visualizing time and memory usage across methods
    - Text file with a tabular summary of results
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import argparse
from pathlib import Path
import numpy as np
from typing import List, Dict, Any
from tabulate import tabulate

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent

def read_performance_stats(root_dir: Path, model_prefixes: List[str]) -> pd.DataFrame:
    """Read and process performance statistics from CSV files."""
    data = defaultdict(list)
    for subdir in os.listdir(root_dir):
        subdir_path = root_dir / subdir
        if subdir_path.is_dir():
            csv_path = subdir_path / 'performance_stats.csv'
            if csv_path.exists():
                df = pd.read_csv(csv_path)
                model_info = parse_model_info(subdir, model_prefixes)
                if model_info:
                    numeric_columns = ['avg_time_per_image', 'avg_gpu_memory_usage']
                    for col in numeric_columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                    
                    data['model'].append(model_info['model'])
                    data['attributes'].append(model_info['attributes'])
                    data['batch_size'].append(model_info['batch_size'])
                    for col in numeric_columns:
                        data[f'{col}_mean'].append(df[col].mean())
                        data[f'{col}_std'].append(df[col].std())
    
    result_df = pd.DataFrame(data)
    
    # Adjust memory values for batch size 32
    mask = (result_df['model'] == 'FD') & (result_df['batch_size'] == 32) & (result_df['attributes'].isin(['r', 'rg', 'rag']))
    result_df.loc[mask, 'avg_gpu_memory_usage_mean'] *= 2
    result_df.loc[mask, 'avg_gpu_memory_usage_std'] *= 2
    
    print(result_df.head())
    return result_df

def parse_model_info(dirname: str, model_prefixes: List[str]) -> Dict[str, Any]:
    """Parse model information from directory name."""
    parts = dirname.split('_')
    if len(parts) < 2:
        return None
    if parts[0] in model_prefixes:
        batch_size = next((int(p[2:]) for p in parts if p.startswith('bs')), 64)
        return {
            'model': parts[0],
            'attributes': parts[1] if len(parts) > 1 else 'standard',
            'batch_size': batch_size
        }
    return None

def create_plots(df: pd.DataFrame, output_dir: Path, font_sizes: Dict[str, int], show_titles: bool, save_svg: bool) -> None:
    """Create and save performance analysis plots."""
    attr_order = ['g', 'r', 'rg', 'rag']
    attr_labels = {'g': 'Gender', 'r': 'Race', 'rg': 'G. x R.', 'rag': 'G. x R. x A.'}
    df['attributes'] = pd.Categorical(df['attributes'], categories=attr_order, ordered=True)
    df = df.sort_values(['model', 'attributes'])

    sd_data = df[df['model'] == 'SD']
    sd_time = sd_data['avg_time_per_image_mean'].values[0] if not sd_data.empty else None
    sd_memory = sd_data['avg_gpu_memory_usage_mean'].values[0] if not sd_data.empty else None

    plot_df = df[df['model'] != 'SD'].copy()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={'wspace': 0.3})
    sns.set_style("whitegrid")
    palette = sns.color_palette("colorblind")

    def plot_barplot(ax, y, ylabel, sd_value):
        sns.barplot(x='model', y=y, hue='attributes', data=plot_df, ax=ax, palette=palette, hue_order=attr_order)

        x_coords = []
        width = 0.8 / len(attr_order)
        for i, model in enumerate(plot_df['model'].unique()):
            for j, attr in enumerate(attr_order):
                mask = (plot_df['model'] == model) & (plot_df['attributes'] == attr)
                if mask.any():
                    x = i + (j - 1.5) * width
                    x_coords.append(x)
                    y_val = plot_df.loc[mask, y].values[0]
                    yerr = plot_df.loc[mask, y.replace('mean', 'std')].values[0]
                    ax.errorbar(x, y_val, yerr=yerr, fmt='none', c='k', capsize=5)

        if sd_value is not None:
            ax.axhline(y=sd_value, color='r', linestyle='--', label='SD')
        ax.set_ylabel(ylabel, fontsize=font_sizes['label'], labelpad=10)
        ax.set_xlabel('Model', fontsize=font_sizes['label'], labelpad=10)
        ax.tick_params(axis='both', which='major', labelsize=font_sizes['tick'])

        if y == 'avg_gpu_memory_usage_mean':
            ax.set_ylim(bottom=1.6e6)

        return x_coords

    plot_barplot(ax1, 'avg_time_per_image_mean', 'Time (seconds)', sd_time)
    if show_titles:
        ax1.set_title('Average Time per Image by Model and Attributes', fontsize=font_sizes['title'])

    plot_barplot(ax2, 'avg_gpu_memory_usage_mean', 'Memory (bytes)', sd_memory)
    if show_titles:
        ax2.set_title('Average GPU Memory Usage by Model and Attributes', fontsize=font_sizes['title'])

    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, [attr_labels.get(label, label) for label in labels], 
               title='', bbox_to_anchor=(0.5, -0.08), 
               loc='lower center', ncol=len(attr_order)+1, 
               fontsize=font_sizes['legend'], title_fontsize=font_sizes['legend_title'])

    ax1.get_legend().remove()
    ax2.get_legend().remove()

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.18)
    output_file = output_dir / 'performance_analysis.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    if save_svg:
        svg_output_file = output_dir / 'performance_analysis.svg'
        plt.savefig(svg_output_file, format='svg', bbox_inches='tight')
    plt.close()
    print(f"Plots saved to '{output_file}' and '{svg_output_file if save_svg else ''}'")

def main(args: argparse.Namespace) -> None:
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory '{input_dir}' does not exist.")

    output_dir.mkdir(parents=True, exist_ok=True)

    df = read_performance_stats(input_dir, args.models)
    if df.empty:
        print("No data found. Please check input directories and model prefixes.")
        return

    font_sizes = {
        'title': args.title_size,
        'label': args.label_size,
        'tick': args.tick_size,
        'legend': args.legend_size,
        'legend_title': args.legend_title_size
    }

    create_plots(df, output_dir, font_sizes, show_titles=not args.no_titles, save_svg=not args.no_svg)
    
    # Save tabular results
    table = tabulate(df, headers='keys', tablefmt='grid', floatfmt=".3f")
    table_file = output_dir / 'performance_summary.txt'
    with open(table_file, 'w') as f:
        f.write(table)
    print(f"Tabular summary saved to '{table_file}'")
    
    print("Analysis completed successfully.")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze performance of image generation models")
    parser.add_argument("--input_dir", type=str, default=BASE_DIR / "data/experiments/section_5.4.1/5.4.1_datasets", help="Path to directory with datasets")
    parser.add_argument("--output_dir", type=str, default=BASE_DIR / "results/section_5.4.1/time_and_memory", help="Path to output directory for plots")
    parser.add_argument("--models", nargs='+', default=['SD', 'DD', 'FD', 'FDM', 'AS'], help="List of model prefixes")
    parser.add_argument("--title_size", type=int, default=16, help="Font size for titles")
    parser.add_argument("--label_size", type=int, default=14, help="Font size for axis labels")
    parser.add_argument("--tick_size", type=int, default=12, help="Font size for tick labels")
    parser.add_argument("--legend_size", type=int, default=14, help="Font size for legend text")
    parser.add_argument("--legend_title_size", type=int, default=14, help="Font size for legend title")
    parser.add_argument("--no_titles", action="store_true", help="Do not show titles")
    parser.add_argument("--no_svg", action="store_true", help="Do not save SVG files")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args)