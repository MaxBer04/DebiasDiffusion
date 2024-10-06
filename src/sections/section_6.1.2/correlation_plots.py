import torch
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import argparse
from matplotlib.gridspec import GridSpec

script_dir = os.path.dirname(os.path.abspath(__file__))

def main(args):
    args.input_file = os.path.join(script_dir, args.input_file)
    
    with open(args.input_file, 'r') as f:
        all_results = json.load(f)

    plots_dir = os.path.join(os.path.dirname(args.input_file), "plots_new")
    os.makedirs(plots_dir, exist_ok=True)

    plt.rcParams.update({
        'font.size': args.fontsize_base,
        'axes.labelsize': args.fontsize_label,
        'axes.titlesize': args.fontsize_title,
        'xtick.labelsize': args.fontsize_tick,
        'ytick.labelsize': args.fontsize_tick,
        'legend.fontsize': args.fontsize_legend
    })

    plot_correlation_lines(all_results, plots_dir, args)
    plot_correlation_grids(all_results, plots_dir, args)

def plot_correlation_lines(all_results, plots_dir, args):
    fig, ax = plt.subplots(figsize=(11, 5))
    vertical_line_color = '#808080'  # Gray color for vertical lines
    
    for t in args.plot_timesteps:
        ax.axvline(x=t, color=vertical_line_color, linestyle='--', alpha=0.5)
    
    for attr in args.supported_attributes:
        timesteps = len(all_results[attr][list(all_results[attr].keys())[0]]['estimated_probs'])
        correlations = []
        for t in range(timesteps):
            estimated_biases = []
            real_biases = []
            for occupation, results in all_results[attr].items():
                p_tar = [0.5, 0.5] if attr != 'race' else [0.25, 0.25, 0.25, 0.25]
                estimated_bias = bias_metric(results['estimated_probs'], p_tar)[t]
                real_bias = bias_metric([results['real_probs']], p_tar)[0]
                estimated_biases.append(estimated_bias)
                real_biases.append(real_bias)
            correlation, _ = pearsonr(estimated_biases, real_biases)
            correlations.append(correlation)
        
        ax.plot(range(50, 0, -1), correlations, label=attr.capitalize(), linewidth=2)

    ax.invert_xaxis()
    ax.set_xticks(range(50, 0, -5))
    ax.set_xticklabels(range(50, 0, -5))

    ax.set_xlabel('Timestep', fontsize=args.fontsize_label)
    ax.set_ylabel('Pearson Correlation r', fontsize=args.fontsize_label)
    ax.xaxis.set_label_coords(0.5, -0.1) 
    ax.yaxis.set_label_coords(-0.075, 0.5)
    if args.show_legend:
        ax.legend(fontsize=args.fontsize_legend)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'pearson_correlation_all_attributes.png'), dpi=300, bbox_inches='tight')
    if args.save_svg:
        plt.savefig(os.path.join(plots_dir, 'pearson_correlation_all_attributes.svg'), bbox_inches='tight')
    plt.close()

def plot_correlation_grids(all_results, plots_dir, args):
    timesteps = sorted(args.plot_timesteps, reverse=True)
    num_plots = len(timesteps)
    
    for attr in args.supported_attributes:
        if args.split_grid:
            mid_point = num_plots // 2
            plot_ranges = [timesteps[:mid_point], timesteps[mid_point:]]
        else:
            plot_ranges = [timesteps]
        
        for idx, plot_range in enumerate(plot_ranges):
            ncols = min(5, len(plot_range))
            nrows = (len(plot_range) + ncols - 1) // ncols
            
            figsize_width = 3 * ncols
            figsize_height = 3 * nrows
            
            fig = plt.figure(figsize=(figsize_width, figsize_height))
            gs = GridSpec(nrows, ncols, figure=fig)
            
            for i, t in enumerate(plot_range):
                ax = fig.add_subplot(gs[i // ncols, i % ncols])
                plot_correlation_scatter(all_results, ax, t, attr, args)
                ax.set_title(f'Timestep = {50-t}', fontsize=args.fontsize_title, pad=10)
                
                if args.hide_inner_ticks:
                    if i // ncols != nrows - 1:  # Not bottom row
                        ax.set_xticklabels([])
                    if i % ncols != 0:  # Not left column
                        ax.set_yticklabels([])
                
                if i // ncols == nrows - 1:  # Bottom row
                    ax.set_xlabel('Estimated Bias', fontsize=args.fontsize_label)
                else:
                    ax.set_xlabel('')
                
                if i % ncols == 0:  # Left column
                    ax.set_ylabel('Real Bias', fontsize=args.fontsize_label)
                else:
                    ax.set_ylabel('')
                
                ax.xaxis.set_label_coords(0.5, -0.2) 
                ax.yaxis.set_label_coords(-0.2, 0.5)
                
                # Set consistent axis limits
                ax.set_xlim(-0.025, 0.525 if attr == "gender" else 0.775)
                ax.set_ylim(-0.025, 0.525 if attr == "gender" else 0.775)

            plt.tight_layout()
            suffix = f'_part{idx+1}' if args.split_grid else ''
            plt.savefig(os.path.join(plots_dir, f'correlation_grid_{attr}{suffix}.png'), dpi=300, bbox_inches='tight')
            if args.save_svg:
                plt.savefig(os.path.join(plots_dir, f'correlation_grid_{attr}{suffix}.svg'), bbox_inches='tight')
            plt.close()

def plot_correlation_scatter(all_results, ax, timestep, attr, args):
    estimated_biases = []
    real_biases = []
    for occupation, results in all_results[attr].items():
        p_tar = [0.5, 0.5] if attr != 'race' else [0.25, 0.25, 0.25, 0.25]
        estimated_bias = bias_metric(results['estimated_probs'], p_tar)[timestep]
        real_bias = bias_metric([results['real_probs']], p_tar)[0]
        estimated_biases.append(estimated_bias)
        real_biases.append(real_bias)
    
    ax.scatter(estimated_biases, real_biases, alpha=0.6)

    m, b = np.polyfit(estimated_biases, real_biases, 1)
    x_line = np.array([-0.025, 0.525 if attr == "gender" else 0.775])
    y_line = m * x_line + b
    ax.plot(x_line, y_line, color='r', linestyle='--')
    
    # Set axis limits
    ax.set_xlim(-0.025, 0.525 if attr == "gender" else 0.775)
    ax.set_ylim(-0.025, 0.525 if attr == "gender" else 0.775)
    
    r_squared = np.corrcoef(estimated_biases, real_biases)[0, 1]**2
    print(f"ATTR: {attr} | TIMESTEP: {timestep} | CORRELATION: {r_squared:.3f}")

def bias_metric(probs_list, p_tar):
    if isinstance(p_tar, torch.Tensor):
        p_tar = p_tar.cpu().numpy()
    elif isinstance(p_tar, list):
        p_tar = np.array(p_tar)
    
    metric_values = []

    for probs in probs_list:
        probs = np.array(probs)
        expected_distribution = np.mean(probs, axis=0)
        diff = expected_distribution - p_tar
        metric_value = np.linalg.norm(diff) ** 2
        metric_values.append(metric_value)
    
    return metric_values

def parse_args():
    parser = argparse.ArgumentParser(description="Analyze and plot biased image data")
    parser.add_argument("--input_file", type=str, default="output_correlations/all_results.json", help="Path to the input JSON file containing the results")
    parser.add_argument("--show_legend", action="store_true", default=True, help="Show legends in plots")
    parser.add_argument("--save_svg", action="store_true", default=True, help="Save plots in SVG format")
    parser.add_argument("--supported_attributes", default=['gender', 'race'], nargs='+', help="The list of attributes to evaluate")
    parser.add_argument("--plot_timesteps", nargs='+', type=int, default=[46, 41, 36, 31, 26], help="Timesteps to plot in the correlation grid and mark in the line plot")
    parser.add_argument("--fontsize_base", type=int, default=14, help="Base font size")
    parser.add_argument("--fontsize_label", type=int, default=16, help="Font size for axis labels")
    parser.add_argument("--fontsize_title", type=int, default=16, help="Font size for plot titles")
    parser.add_argument("--fontsize_tick", type=int, default=14, help="Font size for tick labels")
    parser.add_argument("--fontsize_legend", type=int, default=16, help="Font size for legends")
    parser.add_argument("--fontsize_text", type=int, default=14, help="Font size for additional text in plots")
    parser.add_argument("--hide_inner_ticks", action="store_true", default=True, help="Hide tick labels for inner plots in the grid")
    parser.add_argument("--split_grid", action="store_true", default=True, help="Split the grid plot into two separate plots")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args)