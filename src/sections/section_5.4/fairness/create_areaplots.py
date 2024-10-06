"""
Area Plot Creation for Attribute Distributions

This script generates area plots to visualize the distribution of attributes (gender, race, age)
across different debiasing methods. It processes CSV files containing attribute probabilities
and creates comparative visualizations.

Usage:
    python src/sections/section_5.4/fairness/create_areaplots.py [--args]

Arguments:
    --debias_group: Debiasing group to analyze (r, g, rg, rag) (default: rg)
    --data_dir: Directory containing CSV files (default: /root/DebiasDiffusion/data/5.4.1_attribute_classification_results)
    --num_groups: Number of random groups to analyze. Use -1 for all groups. (default: -1)
    --output_dir: Directory to save output plots (default: outputs/section_5.4/areaplots)
    --save_svg: Save plots as SVG in addition to PNG (default: True)
    --seed: Random seed for reproducibility (default: 1904)
    --gender_target: Target distribution for gender [male, female] (default: 0.5 0.5)
    --race_target: Target distribution for race [white, black, asian, indian] (default: 0.25 0.25 0.25 0.25)
    --age_target: Target distribution for age [young, old] (default: 0.75 0.25)
    --fontsize_tick: Font size for tick labels (default: 16)
    --fontsize_label: Font size for axis labels (default: 22)
    --fontsize_title: Font size for titles (default: 26)
    --dataset_type: Type of dataset: occupation or laion (default: occupation)

Outputs:
    - PNG and optionally SVG area plots for each attribute, showing distributions across methods
"""

import argparse
import os
import random
from pathlib import Path
from typing import List, Dict, Tuple

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent

MODEL_PREFIXES = ['SD', 'FDM', 'FD', 'AS', 'DD']
ATTRIBUTE_LABELS = {
    'gender': ['Male', 'Female'],
    'race': ['White', 'Black', 'Asian', 'Indian'],
    'age': ['Young', 'Old']
}
MODEL_NAMES = {
    'SD': 'Stable Diff.',
    'FDM': 'FDM',
    'FD': 'Fair Diff.',
    'AS': 'Attr. Swit.',
    'DD': 'Debias Diff.'
}

def load_csv_data(file_path: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(file_path)
        for col in ['gender_probs', 'race_probs', 'age_probs']:
            df[col] = df[col].apply(lambda x: eval(x) if isinstance(x, str) else x)
        return df
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return pd.DataFrame()

def get_common_groups(data_frames: List[pd.DataFrame], num: int, seed: int = None, dataset_type: str = 'laion') -> List[str]:
    random.seed(seed)
    group_column = 'occupation' if dataset_type == 'occupation' else 'group'
    common_groups = set(data_frames[0][group_column])
    for df in data_frames[1:]:
        common_groups &= set(df[group_column])

    if num == -1 or len(common_groups) <= num:
        return list(common_groups)
    else:
        return random.sample(list(common_groups), num)

def calculate_average_probs(df: pd.DataFrame, groups: List[str], attribute: str, dataset_type: str) -> Dict[str, List[float]]:
    group_column = 'occupation' if dataset_type == 'occupation' else 'group'
    probs = df[df[group_column].isin(groups)].groupby(group_column)[f'{attribute}_probs'].apply(list).to_dict()
    return {group: [sum(p[i] for p in probs[group]) / len(probs[group]) for i in range(len(probs[group][0]))] for group in groups}

def create_area_plots(data: Dict[str, Dict[str, Dict[str, List[float]]]], output_dir: Path, save_svg: bool, 
                      target_distributions: Dict[str, List[float]], fontsize_tick: int, fontsize_label: int, 
                      fontsize_title: int, dataset_type: str):
    plt.style.use('default')
    
    colors = ['#4e79a7', '#f28e2b', '#e15759', '#76b7b2', '#59a14f', '#edc948', '#b07aa1', '#ff9da7', '#9c755f', '#bab0ac']

    for attribute, classes in ATTRIBUTE_LABELS.items():
        fig, axes = plt.subplots(len(MODEL_PREFIXES), len(classes), figsize=(5*len(classes), 3*len(MODEL_PREFIXES)), squeeze=False)

        for i, model in enumerate(MODEL_PREFIXES):
            for j, class_label in enumerate(classes):
                ax = axes[i, j]

                target_value = target_distributions[attribute][j]
                values = np.array(sorted([data[attribute][model][group][j] for group in data[attribute][model]]))
                x = np.arange(len(values))

                ax.plot(x, values, linewidth=2, color=colors[j % len(colors)])

                above_target = values > target_value
                ax.fill_between(x, values, target_value, where=above_target,
                                alpha=0.3, color=colors[j % len(colors)])
                ax.fill_between(x, values, target_value, where=~above_target,
                                alpha=0.3, color=colors[j % len(colors)])

                ax.axhline(y=target_value, color='#d62728', linestyle='--', linewidth=1.5, label='Target' if i == 0 and j == 0 else '')

                average_value = np.mean(values)
                ax.axhline(y=average_value, color='#5B3417', linestyle='-', linewidth=2, 
                           label='Average' if i == 0 and j == 0 else '')

                ax.set_xlim(0, len(x) - 1)
                ax.set_ylim(0, 1)

                ax.grid(True, linestyle=':', color='#909497', alpha=.9, linewidth=1)
                ax.set_axisbelow(True)

                ax.tick_params(axis='both', which='major', labelsize=fontsize_tick)
                ax.set_xticklabels([])

                if j == 0:
                    ax.set_ylabel('Probability', fontsize=fontsize_label, labelpad=20)
                else:
                    ax.set_yticklabels([])

                if i == len(MODEL_PREFIXES) - 1:
                    ax.set_xlabel('Occupations' if dataset_type == 'occupation' else 'Groups', fontsize=fontsize_label, labelpad=14)

        for i, model in enumerate(MODEL_PREFIXES):
            y_pos = [5, 3.48, 2.6, 1.75, 0.84][i]
            fig.text(0.036, y_pos / len(MODEL_PREFIXES), MODEL_NAMES[model],
                    va='center', ha='left', rotation='vertical', fontsize=fontsize_title,
                    transform=fig.transFigure)

        for j, class_label in enumerate(classes):
            x_pos = [0.222, 0.444, 0.666, 0.887][j] if attribute == 'race' else [0.345, 0.78][j]
            fig.text(x_pos, 0.96, class_label,
                     ha='center', va='bottom', fontsize=fontsize_title)

        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.02),
                   ncol=3, fontsize=fontsize_label)

        plt.tight_layout(rect=[0.05, 0.05, 1, 0.96])

        output_path = output_dir / f"{attribute}_areaplot.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        if save_svg:
            plt.savefig(output_path.with_suffix('.svg'), format='svg', bbox_inches='tight')

        plt.close(fig)

def main(args: argparse.Namespace):
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    target_distributions = {
        'gender': args.gender_target,
        'race': args.race_target,
        'age': args.age_target
    }

    all_dfs = {}

    sd_file_name = "SD_bs64_occs500_legacy-gender.csv" if args.dataset_type == 'occupation' else "SD.csv"
    sd_file_path = data_dir / sd_file_name
    sd_df = load_csv_data(sd_file_path)
    if sd_df.empty:
        print(f"Error: Unable to load SD data from {sd_file_path}. Exiting.")
        return
    all_dfs['SD'] = sd_df

    for prefix in tqdm(['FD', 'FDM', 'DD', 'AS'], desc="Loading data"):
        if args.dataset_type == 'occupation':
            if prefix == 'FD':
                if args.debias_group in ['rg', 'rag']:
                    file_name = f"{prefix}_{args.debias_group}_bs32_occs500_legacy-gender.csv"
                else:
                    file_name = f"{prefix}_{args.debias_group}_bs64_occs500_legacy-gender.csv"
            elif prefix == 'DD':
                file_name = f"{prefix}_{args.debias_group}_s17_bs64_occs500_legacy-gender.csv"
            else:  # FDM and AS
                file_name = f"{prefix}_{args.debias_group}_bs64_occs500_legacy-gender.csv"
        else:  # LAION dataset
            file_name = f"{prefix}_{args.debias_group}.csv"

        file_path = data_dir / file_name

        if not file_path.exists():
            print(f"Warning: File not found: {file_path}")
            continue

        df = load_csv_data(file_path)
        if not df.empty:
            all_dfs[prefix] = df

    if len(all_dfs) < 2:
        print("Error: Not enough valid data loaded. Exiting.")
        return

    groups = get_common_groups(list(all_dfs.values()), args.num_groups, args.seed, args.dataset_type)
    print(f"Analyzing {len(groups)} groups")

    data = {attr: {} for attr in ATTRIBUTE_LABELS.keys()}
    for prefix, df in all_dfs.items():
        for attr in ATTRIBUTE_LABELS.keys():
            data[attr][prefix] = calculate_average_probs(df, groups, attr, args.dataset_type)

    create_area_plots(data, output_dir, args.save_svg, target_distributions,
                      args.fontsize_tick, args.fontsize_label, args.fontsize_title, args.dataset_type)
    print(f"Plots saved in {output_dir}")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate comparison area plots for debiasing models")
    parser.add_argument("--debias_group", type=str, choices=['r', 'g', 'rg', 'rag'], default="rg",
                        help="Debiasing group to analyze (r: race, g: gender, rg: race and gender, rag: race, age, and gender)")
    parser.add_argument("--data_dir", type=str, default="/root/DebiasDiffusion/data/section_5.4.1/5.4.1_attribute_classification_results", help="Directory containing CSV files")
    parser.add_argument("--num_groups", type=int, default=-1, help="Number of random groups to analyze. Use -1 for all groups.")
    parser.add_argument("--output_dir", type=str, default=BASE_DIR / "results/section_5.4.1/group_level_areaplots", help="Directory to save output plots")
    parser.add_argument("--save_svg", action="store_true", default=True, help="Save plots as SVG in addition to PNG")
    parser.add_argument("--seed", type=int, default=1904, help="Random seed for reproducibility")
    parser.add_argument("--gender_target", type=float, nargs=2, default=[0.5, 0.5],
                        help="Target distribution for gender (Male, Female)")
    parser.add_argument("--race_target", type=float, nargs=4, default=[0.25, 0.25, 0.25, 0.25],
                        help="Target distribution for race (White, Black, Asian, Indian)")
    parser.add_argument("--age_target", type=float, nargs=2, default=[0.75, 0.25],
                        help="Target distribution for age (Young, Old)")
    parser.add_argument("--fontsize_tick", type=int, default=16, help="Font size for tick labels")
    parser.add_argument("--fontsize_label", type=int, default=22, help="Font size for axis labels")
    parser.add_argument("--fontsize_title", type=int, default=26, help="Font size for titles")
    parser.add_argument("--dataset_type", type=str, choices=['occupation', 'laion'], default='occupation',
                        help="Type of dataset: occupation (templated) or laion (non-templated)")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args)