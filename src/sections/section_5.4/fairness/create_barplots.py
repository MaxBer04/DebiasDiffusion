import argparse
import os
import random
from pathlib import Path
from typing import List, Dict, Tuple

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tqdm import tqdm

# Constants
SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_PREFIXES = ['FD', 'FTF', 'DD', 'AS', 'SD']
ATTRIBUTE_LABELS = {
    'gender': ['Male', 'Female'],
    'race': ['White', 'Black', 'Asian', 'Indian'],
    'age': ['Young', 'Old']
}
MODEL_NAMES = {
    'FD': 'Fair Diffusion',
    'FTF': 'DFT',
    'DD': 'Debias Diffusion',
    'AS': 'Attribute Switching',
    'SD': 'Stable Diffusion'
}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate comparison plots for debiasing models")
    parser.add_argument("--debias_group", type=str, choices=['r', 'g', 'rg', 'rag'], default="r",
                        help="Debiasing group to analyze (r: race, g: gender, rg: race and gender, rag: race, age, and gender)")
    parser.add_argument("--data_dir", type=str, default="outputs_2", help="Directory containing CSV files relative to script")
    parser.add_argument("--num_occupations", type=int, default=100, help="Number of random occupations to analyze")
    parser.add_argument("--attribute", type=str, choices=['race', 'gender', 'age'], default="race",
                        help="Attribute to analyze")
    parser.add_argument("--target_distribution", type=float, nargs='+', default=[0.25, 0.25, 0.25, 0.25],
                        help="Target distribution for the chosen attribute")
    parser.add_argument("--output_dir", type=str, default="barplots_race_r", help="Directory to save output plots")
    parser.add_argument("--save_svg", action="store_true", default=True, help="Save plots as SVG in addition to PNG")
    parser.add_argument("--show_x_labels", action="store_true", default=False, help="Show x-axis labels (occupation names)")
    parser.add_argument("--seed", type=int, default=2000, help="Random seed for reproducibility")
    parser.add_argument("--fontsize_small", type=int, default=8, help="Font size for small text elements")
    parser.add_argument("--fontsize_medium", type=int, default=10, help="Font size for medium text elements")
    parser.add_argument("--fontsize_large", type=int, default=12, help="Font size for large text elements")
    parser.add_argument("--alternative_mode", action="store_true", default=True, help="Use alternative plotting mode")
    return parser.parse_args()

def load_csv_data(file_path: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(file_path)
        for col in ['gender_probs', 'race_probs', 'age_probs']:
            df[col] = df[col].apply(lambda x: eval(x) if isinstance(x, str) else x)
        return df
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return pd.DataFrame()

def get_common_occupations(data_frames: List[pd.DataFrame], num: int, seed: int = None) -> List[str]:
    random.seed(seed)
    common_occupations = set(data_frames[0]['occupation'])
    for df in data_frames[1:]:
        common_occupations &= set(df['occupation'])

    if len(common_occupations) < num:
        print(f"Warning: Only {len(common_occupations)} common occupations found. Using all available.")
        return list(common_occupations)

    return random.sample(list(common_occupations), num)

def calculate_average_probs(df: pd.DataFrame, occupations: List[str], attribute: str) -> Dict[str, List[float]]:
    probs = df[df['occupation'].isin(occupations)].groupby('occupation')[f'{attribute}_probs'].apply(list).to_dict()
    return {occ: [sum(p[i] for p in probs[occ]) / len(probs[occ]) for i in range(len(probs[occ][0]))] for occ in occupations}

def create_plots(data: Dict[str, Dict[str, List[float]]], occupations: List[str], attribute: str, 
                 target_distribution: List[float], output_dir: Path, save_svg: bool, show_x_labels: bool,
                 fontsize_small: int, fontsize_medium: int, fontsize_large: int, alternative_mode: bool):
    num_classes = len(ATTRIBUTE_LABELS[attribute])
    num_models = 5 if alternative_mode else 4
    fig, axes = plt.subplots(num_models, num_classes, figsize=(20, 6*num_models), constrained_layout=True)

    plt.style.use('default')
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3
    plt.rcParams['axes.axisbelow'] = True

    try:
        colors = sns.color_palette("husl", 2)
    except:
        colors = ['#1f77b4', '#ff7f0e']

    model_order = MODEL_PREFIXES if alternative_mode else MODEL_PREFIXES[:-1]

    for i, model in enumerate(model_order):
        for j, class_label in enumerate(ATTRIBUTE_LABELS[attribute]):
            ax = axes[i, j]
            model_probs = [data[model][occ][j] for occ in occupations]

            if alternative_mode:
                sorted_indices = sorted(range(len(model_probs)), key=lambda k: model_probs[k])
            else:
                sd_probs = [data['SD'][occ][j] for occ in occupations]
                sorted_indices = sorted(range(len(sd_probs)), key=lambda k: sd_probs[k])

            sorted_occupations = [occupations[i] for i in sorted_indices]
            sorted_model_probs = [model_probs[i] for i in sorted_indices]

            x = range(len(sorted_occupations))
            width = 0.7 if alternative_mode else 0.35

            ax.bar(x, sorted_model_probs, width, label=model, color=colors[0], alpha=0.8)

            if not alternative_mode and model != 'SD':
                sorted_sd_probs = [sd_probs[i] for i in sorted_indices]
                ax.bar([i + width for i in x], sorted_sd_probs, width, label='SD', color=colors[1], alpha=0.5)

            ax.axhline(y=target_distribution[j], color='r', linestyle='--', label='Target')

            ax.set_ylim(0, 1)
            ax.set_ylabel('Probability', fontsize=fontsize_medium)
            if show_x_labels:
                ax.set_xticks(x)
                ax.set_xticklabels(sorted_occupations, rotation=90, ha='right', fontsize=fontsize_small)
            else:
                ax.set_xticks([])

            ax.tick_params(axis='y', labelsize=fontsize_small)

            if i == num_models - 1:  # Only show x-label for bottom row
                ax.set_xlabel('Occupation', fontsize=fontsize_medium)

            if j == 0:  # Add model name to the left of the first plot in each row
                ax.text(-0.3, 0.5, MODEL_NAMES[model], rotation=90, va='center', ha='right', transform=ax.transAxes, fontsize=fontsize_large, fontweight='bold')

            if i == 0:  # Add class label above the first row
                ax.text(0.5, 1.1, class_label, ha='center', va='bottom', transform=ax.transAxes, fontsize=fontsize_large, fontweight='bold')

            legend = ax.get_legend()
            if legend is not None:
                legend.remove()

    plt.tight_layout()
    fig.subplots_adjust(top=0.95, bottom=0.05, left=0.05, right=0.98)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.98),
               ncol=3, fontsize=fontsize_medium)

    combined_path = output_dir / f"{attribute}_comparison{'_alt' if alternative_mode else ''}.png"
    plt.savefig(combined_path, dpi=300, bbox_inches='tight')
    if save_svg:
        plt.savefig(combined_path.with_suffix('.svg'), format='svg', bbox_inches='tight')

    plt.close('all')

def main():
    args = parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    data_dir = SCRIPT_DIR / args.data_dir
    output_dir = SCRIPT_DIR / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    all_dfs = {}

    sd_file_path = data_dir / "SD_bs64_occs500_legacy-gender.csv"
    sd_df = load_csv_data(sd_file_path)
    if sd_df.empty:
        print("Error: Unable to load SD data. Exiting.")
        return
    all_dfs['SD'] = sd_df

    for prefix in tqdm(['FD', 'FTF', 'DD', 'AS'], desc="Loading data"):
        if prefix == 'FD':
            if args.debias_group in ['rg', 'rag']:
                file_path = data_dir / f"{prefix}_{args.debias_group}_bs32_occs500_legacy-gender.csv"
            else:
                file_path = data_dir / f"{prefix}_{args.debias_group}_bs64_occs500_legacy-gender.csv"
        elif prefix == 'DD':
            file_path = data_dir / f"{prefix}_{args.debias_group}_s17_bs64_occs500_legacy-gender.csv"
        else:  # FTF and AS
            file_path = data_dir / f"{prefix}_{args.debias_group}_bs64_occs500_legacy-gender.csv"

        if not file_path.exists():
            print(f"Warning: File not found: {file_path}")
            continue

        df = load_csv_data(file_path)
        if not df.empty:
            all_dfs[prefix] = df

    if len(all_dfs) < 2:
        print("Error: Not enough valid data loaded. Exiting.")
        return

    occupations = get_common_occupations(list(all_dfs.values()), args.num_occupations, args.seed)

    data = {}
    for prefix, df in all_dfs.items():
        data[prefix] = calculate_average_probs(df, occupations, args.attribute)

    if len(args.target_distribution) != len(ATTRIBUTE_LABELS[args.attribute]):
        print(f"Error: Target distribution must have {len(ATTRIBUTE_LABELS[args.attribute])} values for {args.attribute}.")
        return

    create_plots(data, occupations, args.attribute, args.target_distribution, output_dir, args.save_svg, args.show_x_labels,
                 args.fontsize_small, args.fontsize_medium, args.fontsize_large, args.alternative_mode)
    print(f"Plots saved in {output_dir}")

if __name__ == "__main__":
    main()