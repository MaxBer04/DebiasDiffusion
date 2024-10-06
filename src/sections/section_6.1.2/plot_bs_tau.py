"""
Batch Size and Tau Bias Analysis for DebiasDiffusion

This script analyzes and visualizes the relationship between semantic preservation (measured by LPIPS)
and fairness (measured by FD metric) for different batch sizes and tau_bias values in the DebiasDiffusion
pipeline. It processes data from generated image datasets and creates plots to illustrate these relationships.

Usage:
    python src/sections/section_6.1.2/plot_bs_tau.py [--args]

Data:
    The script uses pre-computed LPIPS and FD values. The original datasets are located in:
    data/experiments/section_6.1.2/6.1.2_datasets

    Pre-computed results used in the thesis are available in:
    data/experiments/section_6.1.2

    To generate new results from different datasets:
    1. Use scripts in src/sections/section_5.4/ to evaluate LPIPS and FD metrics
    2. Update the 'lpips' and 'bias' dictionaries in this script with the new values

Arguments:
    --output_dir: Directory to save output plots (default: BASE_DIR / "results/section_6.1.2/batch_size_tau_analysis")
    --fontsize_tick: Font size for tick labels (default: 16)
    --fontsize_label: Font size for axis labels (default: 18)
    --fontsize_title: Font size for titles (default: 20)
    --fontsize_legend: Font size for legend (default: 20)
    --markersize: Size of markers (default: 60)
    --save_svg: Save plots as SVG in addition to PNG (default: True)

Outputs:
    - PNG and optionally SVG plots illustrating the relationship between LPIPS and FD for different attributes
    - Console output with plot generation progress
"""

import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from matplotlib.lines import Line2D
from typing import Dict, List, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent.parent.parent

def create_dataframe(lpips: Dict[str, float], bias: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    """
    Create a pandas DataFrame from LPIPS and bias data.
    
    Args:
        lpips (Dict[str, float]): Dictionary of LPIPS scores
        bias (Dict[str, Dict[str, float]]): Dictionary of bias scores for different attributes
    
    Returns:
        pd.DataFrame: DataFrame containing the combined data
    """
    data = []
    for key in lpips.keys():
        timestep, batch_size = key.split('_')
        data.append({
            'Timestep': int(timestep),
            'Batch Size': int(batch_size),
            'LPIPS': lpips[key],
            'GenderBias': bias[key]['gender'],
            'RaceBias': bias[key]['race'],
            'AgeBias': bias[key]['age']
        })
    return pd.DataFrame(data)

def plot_lpips_vs_bias(df: pd.DataFrame, attribute: str, ax: plt.Axes, font_sizes: Dict[str, int], 
                       markersize: int, colors: List[str], markers: List[str], fix_parameter: str) -> None:
    """
    Plot LPIPS vs Bias for a specific attribute.
    
    Args:
        df (pd.DataFrame): DataFrame containing the data
        attribute (str): Attribute to plot (Gender, Race, or Age)
        ax (plt.Axes): Matplotlib axes object to plot on
        font_sizes (Dict[str, int]): Dictionary of font sizes
        markersize (int): Size of markers in the plot
        colors (List[str]): List of colors for different series
        markers (List[str]): List of markers for different data points
        fix_parameter (str): Parameter to fix ('batch_size' or 'timestep')
    """
    if fix_parameter == 'batch_size':
        fixed_values = sorted(df['Batch Size'].unique())
        variable_values = sorted(df['Timestep'].unique())
        fixed_column = 'Batch Size'
        variable_column = 'Timestep'
    else:  # fix_parameter == 'timestep'
        fixed_values = sorted(df['Timestep'].unique())
        variable_values = sorted(df['Batch Size'].unique())
        fixed_column = 'Timestep'
        variable_column = 'Batch Size'
    
    for i, fixed_value in enumerate(fixed_values):
        data = df[df[fixed_column] == fixed_value].sort_values(by=variable_column)
        ax.plot(data['LPIPS'], data[f'{attribute}Bias'], 
                color=colors[i], linestyle='-', linewidth=1.5,
                label=f'{fixed_column}={fixed_value}')
        for j, variable_value in enumerate(variable_values):
            point_data = data[data[variable_column] == variable_value]
            if not point_data.empty:
                ax.scatter(point_data['LPIPS'], point_data[f'{attribute}Bias'], 
                           color=colors[i], marker=markers[j],
                           s=markersize, zorder=10)

    ax.set_xlabel('LPIPS $\\rightarrow$', fontsize=font_sizes['label'])
    ax.set_ylabel('$\\leftarrow$ Bias (FD)', fontsize=font_sizes['label'])
    
    ax.tick_params(axis='both', which='major', labelsize=font_sizes['tick'])
    
    ax.set_xticks(np.arange(0.75, 0.9, 0.03))
    ax.set_xlim(0.74, 0.88)

def create_custom_legend(fixed_values: List[int], variable_values: List[int], fixed_column: str, 
                         variable_column: str, colors: List[str], markers: List[str], 
                         font_sizes: Dict[str, int]) -> Tuple[List[Line2D], List[Line2D]]:
    """
    Create custom legend elements for the plot.
    
    Args:
        fixed_values (List[int]): List of fixed parameter values
        variable_values (List[int]): List of variable parameter values
        fixed_column (str): Name of the fixed parameter column
        variable_column (str): Name of the variable parameter column
        colors (List[str]): List of colors for different series
        markers (List[str]): List of markers for different data points
        font_sizes (Dict[str, int]): Dictionary of font sizes
    
    Returns:
        Tuple[List[Line2D], List[Line2D]]: Legend elements for fixed and variable parameters
    """
    legend_elements = []
    
    for i, value in enumerate(fixed_values):
        if fixed_column == 'Timestep':
            label = f'$\\tau_{{bias}}$ = {value+1}'
        else:
            label = f'{fixed_column} = {value}'
        legend_elements.append(Line2D([0], [0], color=colors[i], lw=1.5, label=label))
    
    for j, value in enumerate(variable_values):
        if variable_column == 'Timestep':
            label = f'$\\tau_{{bias}}$ = {value+1}'
        else:
            label = f'{variable_column} = {value}'
        legend_elements.append(Line2D([0], [0], color='gray', marker=markers[j], linestyle='None',
                                      markersize=5, label=label))
    
    if fixed_column == 'Timestep':
        fixed_legend = legend_elements[:len(fixed_values)][::-1]
    else:
        fixed_legend = legend_elements[:len(fixed_values)]
    
    if variable_column == 'Timestep':
        variable_legend = legend_elements[len(fixed_values):][::-1]
    else:
        variable_legend = legend_elements[len(fixed_values):]
    
    return fixed_legend, variable_legend

def create_plot(df: pd.DataFrame, output_dir: Path, font_sizes: Dict[str, int], markersize: int, 
                save_svg: bool, fix_parameter: str) -> None:
    """
    Create and save the main plot.
    
    Args:
        df (pd.DataFrame): DataFrame containing the data
        output_dir (Path): Directory to save the output plots
        font_sizes (Dict[str, int]): Dictionary of font sizes
        markersize (int): Size of markers in the plot
        save_svg (bool): Whether to save the plot as SVG
        fix_parameter (str): Parameter to fix ('batch_size' or 'timestep')
    """
    attributes = ['Gender', 'Race', 'Age']
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=300)
    
    if fix_parameter == 'batch_size':
        fixed_values = sorted(df['Batch Size'].unique())
        variable_values = sorted(df['Timestep'].unique())
        fixed_column = 'Batch Size'
        variable_column = 'Timestep'
    else:  # fix_parameter == 'timestep'
        fixed_values = sorted(df['Timestep'].unique())
        variable_values = sorted(df['Batch Size'].unique())
        fixed_column = 'Timestep'
        variable_column = 'Batch Size'
    
    colors = sns.color_palette("husl", n_colors=len(fixed_values))
    if fix_parameter == 'timestep':
        colors = colors[::-1]
    markers = ['o', 's', 'D', '^', 'X']
    
    for col, attribute in enumerate(attributes):
        ax = axes[col]
        plot_lpips_vs_bias(df, attribute, ax, font_sizes, markersize, colors, markers, fix_parameter)
        
        ax.set_title(attribute, fontsize=font_sizes['title'], pad=10)
        
        if col != 0:
            ax.set_ylabel('')
    
    fixed_legend, variable_legend = create_custom_legend(fixed_values, variable_values, fixed_column, variable_column, colors, markers, font_sizes)
    
    fig.legend(handles=fixed_legend, loc='lower center', bbox_to_anchor=(0.5, -0.15),
               fontsize=font_sizes['legend'], ncol=len(fixed_values))
    fig.legend(handles=variable_legend, loc='lower center', bbox_to_anchor=(0.5, -0.275),
               fontsize=font_sizes['legend'], ncol=len(variable_values))

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)  # Adjust bottom margin for legend
    output_path = output_dir / f"lpips_vs_bias_{fix_parameter}_fixed.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved as {output_path}")
    
    if save_svg:
        svg_path = output_path.with_suffix('.svg')
        plt.savefig(svg_path, format='svg', bbox_inches='tight')
        print(f"SVG plot saved as {svg_path}")
    
    plt.close(fig)

def main(args: argparse.Namespace) -> None:
    lpips = {
        "45_128": 0.785, "40_128": 0.852, "35_128": 0.878, "30_128": 0.849, "25_128": 0.870,
        "45_64": 0.778, "40_64": 0.820, "35_64": 0.871, "30_64": 0.845, "25_64": 0.862,
        "45_32": 0.770, "40_32": 0.796, "35_32": 0.855, "30_32": 0.821, "25_32": 0.843,
        "45_16": 0.756, "40_16": 0.764, "35_16": 0.829, "30_16": 0.787, "25_16": 0.812
    }

    bias = {
        "45_128": {"gender": 0.006, "race": 0.003, "age": 0.021},
        "40_128": {"gender": 0.006, "race": 0.004, "age": 0.015},
        "35_128": {"gender": 0.016, "race": 0.008, "age": 0.008},
        "30_128": {"gender": 0.007, "race": 0.006, "age": 0.010},
        "25_128": {"gender": 0.013, "race": 0.010, "age": 0.007},
        "45_64": {"gender": 0.004, "race": 0.003, "age": 0.020},
        "40_64": {"gender": 0.003, "race": 0.004, "age": 0.012},
        "35_64": {"gender": 0.012, "race": 0.005, "age": 0.009},
        "30_64": {"gender": 0.006, "race": 0.005, "age": 0.010},
        "25_64": {"gender": 0.011, "race": 0.007, "age": 0.008},
        "45_32": {"gender": 0.004, "race": 0.001, "age": 0.020},
        "40_32": {"gender": 0.003, "race": 0.002, "age": 0.014},
        "35_32": {"gender": 0.014, "race": 0.007, "age": 0.009},
        "30_32": {"gender": 0.004, "race": 0.002, "age": 0.014},
        "25_32": {"gender": 0.014, "race": 0.004, "age": 0.010},
        "45_16": {"gender": 0.006, "race": 0.002, "age": 0.020},
        "40_16": {"gender": 0.001, "race": 0.002, "age": 0.017},
        "35_16": {"gender": 0.016, "race": 0.005, "age": 0.010},
        "30_16": {"gender": 0.002, "race": 0.003, "age": 0.018},
        "25_16": {"gender": 0.015, "race": 0.004, "age": 0.011}
    }

    df = create_dataframe(lpips, bias)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    font_sizes = {
        'tick': args.fontsize_tick,
        'label': args.fontsize_label,
        'title': args.fontsize_title,
        'legend': args.fontsize_legend
    }
    
    create_plot(df, output_dir, font_sizes, args.markersize, args.save_svg, 'batch_size')
    create_plot(df, output_dir, font_sizes, args.markersize, args.save_svg, 'timestep')
    
    print(f"All plots saved in {output_dir}")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate LPIPS vs Bias plots for DebiasDiffusion")
    parser.add_argument("--output_dir", type=Path, default=BASE_DIR / "results/section_6.1.2/batch_size_tau_analysis",
                        help="Directory to save output plots")
    parser.add_argument("--fontsize_tick", type=int, default=16, help="Font size for tick labels")
    parser.add_argument("--fontsize_label", type=int, default=18, help="Font size for axis labels")
    parser.add_argument("--fontsize_title", type=int, default=20, help="Font size for titles")
    parser.add_argument("--fontsize_legend", type=int, default=20, help="Font size for legend")
    parser.add_argument("--markersize", type=int, default=60, help="Size of markers")
    parser.add_argument("--save_svg", action="store_true", default=True, help="Save plots as SVG in addition to PNG")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args)