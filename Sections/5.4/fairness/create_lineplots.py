import argparse
import os
import random
from pathlib import Path
from typing import List, Dict, Tuple

import pandas as pd
import matplotlib.pyplot as plt
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
    parser = argparse.ArgumentParser(description="Generate comparison line plots for debiasing models")
    parser.add_argument("--debias_group", type=str, choices=['r', 'g', 'rg', 'rag'], default="r",
                        help="Debiasing group to analyze (r: race, g: gender, rg: race and gender, rag: race, age, and gender)")
    parser.add_argument("--data_dir", type=str, default="outputs_2", help="Directory containing CSV files relative to script")
    parser.add_argument("--num_occupations", type=int, default=500, help="Number of random occupations to analyze")
    parser.add_argument("--output_dir", type=str, default="lineplots_r", help="Directory to save output plots")
    parser.add_argument("--save_svg", action="store_true", default=True, help="Save plots as SVG in addition to PNG")
    parser.add_argument("--seed", type=int, default=2000, help="Random seed for reproducibility")
    parser.add_argument("--gender_target", type=float, nargs=2, default=[0.5, 0.5],
                        help="Target distribution for gender (Male, Female)")
    parser.add_argument("--race_target", type=float, nargs=4, default=[0.25, 0.25, 0.25, 0.25],
                        help="Target distribution for race (White, Black, Asian, Indian)")
    parser.add_argument("--age_target", type=float, nargs=2, default=[0.75, 0.25],
                        help="Target distribution for age (Young, Old)")
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

def create_line_plots(data: Dict[str, Dict[str, Dict[str, List[float]]]], output_dir: Path, save_svg: bool, target_distributions: Dict[str, List[float]]):
    plt.style.use('default')
    colors = plt.cm.get_cmap('Set2')(np.linspace(0, 1, len(MODEL_PREFIXES)))
    
    for attribute, classes in ATTRIBUTE_LABELS.items():
        fig, axes = plt.subplots(1, len(classes), figsize=(6*len(classes), 5), squeeze=False)
        fig.suptitle(f'{attribute.capitalize()} Probability Distribution', fontsize=16)
        
        for i, class_label in enumerate(classes):
            ax = axes[0, i]
            
            for j, model in enumerate(MODEL_PREFIXES):
                values = [data[attribute][model][occ][i] for occ in data[attribute][model]]
                values.sort()
                ax.plot(values, label=MODEL_NAMES[model], linewidth=2, color=colors[j])

            # Add target distribution line
            target_value = target_distributions[attribute][i]
            ax.axhline(y=target_value, color='r', linestyle='--', label='Target' if j == 0 else '')

            ax.set_xlabel('Sorted Sample Index', fontsize=10)
            ax.set_ylabel('Probability', fontsize=10)
            ax.set_title(f'{class_label}', fontsize=12)
            ax.set_xlim(0, len(values) - 1)
            ax.set_ylim(0, 1)
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.tick_params(axis='both', which='major', labelsize=8)
        
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.05),
                   ncol=len(MODEL_PREFIXES) + 1, fontsize=10)
        
        plt.tight_layout(rect=[0, 0.1, 1, 0.95])
        
        output_path = output_dir / f"{attribute}_lineplot.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        if save_svg:
            plt.savefig(output_path.with_suffix('.svg'), format='svg', bbox_inches='tight')
        
        plt.close(fig)

def main():
    args = parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    data_dir = SCRIPT_DIR / args.data_dir
    output_dir = SCRIPT_DIR / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    target_distributions = {
        'gender': args.gender_target,
        'race': args.race_target,
        'age': args.age_target
    }

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

    data = {attr: {} for attr in ATTRIBUTE_LABELS.keys()}
    for prefix, df in all_dfs.items():
        for attr in ATTRIBUTE_LABELS.keys():
            data[attr][prefix] = calculate_average_probs(df, occupations, attr)

    create_line_plots(data, output_dir, args.save_svg, target_distributions)
    print(f"Plots saved in {output_dir}")

if __name__ == "__main__":
    main()