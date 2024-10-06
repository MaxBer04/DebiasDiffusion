import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from matplotlib.lines import Line2D

SCRIPT_DIR = Path(__file__).resolve().parent

def parse_args():
    parser = argparse.ArgumentParser(description="Generate LPIPS vs Bias plots")
    parser.add_argument("--output_dir", type=str, default=SCRIPT_DIR / "results_Bias_v_LPIPS",
                        help="Directory to save output plots")
    parser.add_argument("--fontsize_tick", type=int, default=16, help="Font size for tick labels")
    parser.add_argument("--fontsize_label", type=int, default=18, help="Font size for axis labels")
    parser.add_argument("--fontsize_title", type=int, default=20, help="Font size for titles")
    parser.add_argument("--fontsize_legend", type=int, default=20, help="Font size for legend")
    parser.add_argument("--markersize", type=int, default=60, help="Size of markers")
    parser.add_argument("--save_svg", action="store_true", default=True, help="Save plot as SVG in addition to PNG")
    return parser.parse_args()

def create_dataframe(lpips, bias):
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

def plot_lpips_vs_bias(df, attribute, ax, font_sizes, markersize, colors, markers, fix_parameter):
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
    
    # Set more natural x-axis ticks
    ax.set_xticks(np.arange(0.75, 0.9, 0.03))
    ax.set_xlim(0.74, 0.88)

def create_custom_legend(fig, fixed_values, variable_values, fixed_column, variable_column, colors, markers, font_sizes):
    legend_elements = []
    
    # Fixed parameter legend
    for i, value in enumerate(fixed_values):
        if fixed_column == 'Timestep':
            label = f'$\\tau_{{bias}}$ = {value+1}'
        else:
            label = f'{fixed_column} = {value}'
        legend_elements.append(Line2D([0], [0], color=colors[i], lw=1.5, label=label))
    
    # Variable parameter legend
    for j, value in enumerate(variable_values):
        if variable_column == 'Timestep':
            label = f'$\\tau_{{bias}}$ = {value+1}'
        else:
            label = f'{variable_column} = {value}'
        legend_elements.append(Line2D([0], [0], color='gray', marker=markers[j], linestyle='None',
                                      markersize=5, label=label))
    
    # Reverse the order of τ_bias elements in the legend
    if fixed_column == 'Timestep':
        fixed_legend = legend_elements[:len(fixed_values)][::-1]
    else:
        fixed_legend = legend_elements[:len(fixed_values)]
    
    if variable_column == 'Timestep':
        variable_legend = legend_elements[len(fixed_values):][::-1]
    else:
        variable_legend = legend_elements[len(fixed_values):]
    
    return fixed_legend, variable_legend

def create_plot(df, output_dir, font_sizes, markersize, save_svg, fix_parameter):
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
    
    fixed_legend, variable_legend = create_custom_legend(fig, fixed_values, variable_values, fixed_column, variable_column, colors, markers, font_sizes)
    
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

def main():
    args = parse_args()
    
    lpips = {
        "45_128": 0.785, "40_128": 0.852, "35_128": 0.878, "30_128": 0.849, "25_128": 0.870,  #"35_128": 0.878, "30_128": 0.849, "25_128": 0.870, 
        "45_64": 0.778, "40_64": 0.820, "35_64": 0.871, "30_64": 0.845, "25_64": 0.862, #"35_64": 0.871, "30_64": 0.845, "25_64": 0.862,
        "45_32": 0.770, "40_32": 0.796, "35_32": 0.855, "30_32": 0.821, "25_32": 0.843,  #"35_32": 0.855, "30_32": 0.821, "25_32": 0.843, 
        "45_16": 0.756, "40_16": 0.764, "35_16": 0.829, "30_16": 0.787, "25_16": 0.812 #"35_16": 0.829, "30_16": 0.787, "25_16": 0.812 
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

if __name__ == "__main__":
    main()