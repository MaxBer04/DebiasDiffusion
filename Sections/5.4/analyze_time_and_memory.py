import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import argparse
from pathlib import Path
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = Path(__file__).resolve().parent.parent.parent

def read_performance_stats(root_dir, model_prefixes):
    data = defaultdict(list)
    for subdir in os.listdir(root_dir):
        if os.path.isdir(os.path.join(root_dir, subdir)):
            csv_path = os.path.join(root_dir, subdir, 'performance_stats.csv')
            if os.path.exists(csv_path):
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
    #result_df.loc[mask, 'avg_gpu_memory_usage_mean'] *= 2
    #result_df.loc[mask, 'avg_gpu_memory_usage_std'] *= 2
    
    # Adjust time values for 'rag' in DD and FTF
    #mask = (result_df['model'].isin(['DD', 'FTF', 'AS'])) & (result_df['attributes'] == 'rag')
    #result_df.loc[mask, 'avg_time_per_image_mean'] *= 1.69
    #result_df.loc[mask, 'avg_time_per_image_std'] *= 1.69
    #mask = (result_df['model'].isin(['AS'])) & (result_df['attributes'] == 'g')
    #result_df.loc[mask, 'avg_time_per_image_std'] *= 1.69
    
    print(result_df.head())
    return result_df

def parse_model_info(dirname, model_prefixes):
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

def create_plots(df, output_dir, font_sizes, show_titles=True, save_svg=True):
    # Sortieren der Attribute in der gewünschten Reihenfolge
    attr_order = ['g', 'r', 'rg', 'rag']
    attr_labels = {'g': 'Gender', 'r': 'Race', 'rg': 'G. x R.', 'rag': 'G. x R. x A.'}
    df['attributes'] = pd.Categorical(df['attributes'], categories=attr_order, ordered=True)
    df = df.sort_values(['model', 'attributes'])

    # SD-Werte extrahieren
    sd_data = df[df['model'] == 'SD']
    sd_time = sd_data['avg_time_per_image_mean'].values[0] if not sd_data.empty else None
    sd_memory = sd_data['avg_gpu_memory_usage_mean'].values[0] if not sd_data.empty else None

    # Daten für alle Modelle außer SD
    plot_df = df[df['model'] != 'SD'].copy()
    #plot_df['model'] = plot_df['model'].replace('FD', 'FDiff')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={'wspace': 0.3}) 
    sns.set_style("whitegrid")
    palette = sns.color_palette("colorblind")

    def plot_barplot(ax, y, ylabel, sd_value):
        sns.barplot(x='model', y=y, hue='attributes', data=plot_df, ax=ax, palette=palette, hue_order=attr_order)

        # Correct placement of error bars
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

        # Set y-axis limits for memory plot
        if y == 'avg_gpu_memory_usage_mean':
            ax.set_ylim(bottom=1.6e6)

        return x_coords

    # Plot 1: Average Time per Image
    x_coords = plot_barplot(ax1, 'avg_time_per_image_mean', 'Time (seconds)', sd_time)
    if show_titles:
        ax1.set_title('Average Time per Image by Model and Attributes', fontsize=font_sizes['title'])

    # Plot 2: Average GPU Memory Usage
    plot_barplot(ax2, 'avg_gpu_memory_usage_mean', 'Memory (bytes)', sd_memory)
    if show_titles:
        ax2.set_title('Average GPU Memory Usage by Model and Attributes', fontsize=font_sizes['title'])

    # Shared legend
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, [attr_labels.get(label, label) for label in labels], 
               title='', bbox_to_anchor=(0.5, -0.08), 
               loc='lower center', ncol=len(attr_order)+1, 
               fontsize=font_sizes['legend'], title_fontsize=font_sizes['legend_title'])

    # Remove individual legends
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
    print(f"Plots wurden in '{output_file}' gespeichert.")
    if save_svg:
        print(f"SVG-Version wurde in '{svg_output_file}' gespeichert.")

def main():
    parser = argparse.ArgumentParser(description="Analyse der Performance von Bildgenerierungsmodellen")
    parser.add_argument("--input", type=str, default=BASE_DIR / "data" / "datasets", help="Pfad zum Verzeichnis mit den Datasets")
    parser.add_argument("--output", type=str, default=BASE_DIR / "outputs/section_5.4/time_and_memory", help="Pfad zum Ausgabeverzeichnis für die Plots")
    parser.add_argument("--models", nargs='+', default=['SD', 'DD', 'FD', 'FDM', 'AS'], help="Liste der Modellkürzel (z.B. FD, SD)")
    parser.add_argument("--title-size", type=int, default=16, help="Schriftgröße für Titel")
    parser.add_argument("--label-size", type=int, default=14, help="Schriftgröße für Achsenbeschriftungen")
    parser.add_argument("--tick-size", type=int, default=12, help="Schriftgröße für Tick-Labels")
    parser.add_argument("--legend-size", type=int, default=14, help="Schriftgröße für Legendentext")
    parser.add_argument("--legend-title-size", type=int, default=14, help="Schriftgröße für Legendentitel")
    parser.add_argument("--no-titles", action="store_true", default=True, help="Keine Titel anzeigen")
    parser.add_argument("--no-svg", action="store_true", default=False, help="Keine SVG-Datei speichern")
    args = parser.parse_args()

    input_dir = SCRIPT_DIR / args.input
    output_dir = BASE_DIR / args.output

    if not input_dir.exists():
        print(f"Fehler: Das Eingabeverzeichnis '{input_dir}' existiert nicht.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    df = read_performance_stats(input_dir, args.models)
    if df.empty:
        print("Keine Daten gefunden. Überprüfen Sie die Eingabeverzeichnisse und Modellkürzel.")
        return

    font_sizes = {
        'title': args.title_size,
        'label': args.label_size,
        'tick': args.tick_size,
        'legend': args.legend_size,
        'legend_title': args.legend_title_size
    }

    create_plots(df, output_dir, font_sizes, show_titles=not args.no_titles, save_svg=not args.no_svg)
    print("Analyse abgeschlossen.")

if __name__ == "__main__":
    main()