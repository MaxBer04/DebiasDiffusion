import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import argparse

script_dir = os.path.dirname(os.path.abspath(__file__))

def load_data(csv_file):
    return pd.read_csv(csv_file)

def calculate_percentages(data, classes):
    total = sum(data.values())
    return [100 * data[cls] / total for cls in classes]

def group_data(data, groupings):
    print(data)
    grouped = {group: sum(data[cls] for cls in classes) for group, classes in groupings.items()}
    return grouped

def plot_distribution(data, classes, title, output_dir, fig_size, equal_line=False, save_svg=False):
    plt.figure(figsize=fig_size)
    percentages = calculate_percentages(data, classes)

    bars = plt.bar(classes, percentages, color='cornflowerblue')
    plt.ylabel('Percentage (%)', fontsize=22, labelpad=20)
    plt.title(f'{title} Distribution', fontsize=22, pad=20)
    plt.xticks(rotation=45 if title == 'Age' else 55, fontsize=16)
    plt.yticks(fontsize=16)
    plt.ylim(0, 100)

    if equal_line:
        equal_percentage = 100 / len(classes) if title != 'Age' else 75
        plt.axhline(y=equal_percentage, color='gray', linestyle='--', linewidth=1, alpha=0.7)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{title.lower()}_dist.png"), bbox_inches='tight')
    if save_svg:
        plt.savefig(os.path.join(output_dir, f"{title.lower()}_dist.svg"), bbox_inches='tight', format='svg')
    plt.close()

def print_class_counts(data, classes, attribute):
    print(f"\n{attribute} Class Counts:")
    for cls in classes:
        count = data[cls]
        print(f"{cls}: {count}")

def main(args):
    args.input_dir = os.path.join(script_dir, args.input_dir)
    args.output_dir = os.path.join(script_dir, args.output_dir)
    df = load_data(args.input_dir)

    GENDER_CLASSES = ['Male', 'Female']
    RACE_GROUPINGS = {
        "White": ["White"], #"WMELH": ["White", "Middle Eastern", "Latino Hispanic"],
        "Black": ["Black"],
        "Asian": ["Asian"], #["East Asian", "Southeast Asian"],
        "Indian": ["Indian"]
    }
    AGE_GROUPINGS = {
        "Young": ["Young"], #["0-2", "3-9", "10-19", "20-29", "30-39"],
        "Old": ["Old"], #["40-49", "50-59", "60-69", "70+"]
    }

    os.makedirs(args.output_dir, exist_ok=True)

    # Gender plot remains the same
    gender_counts = df['gender'].value_counts().to_dict()
    plot_distribution(gender_counts, GENDER_CLASSES, 'Gender', args.output_dir, args.fig_size, args.equal_line, args.save_svg)

    # Race plot with new groupings
    race_counts = df['race'].value_counts().to_dict()
    grouped_race_counts = group_data(race_counts, RACE_GROUPINGS)
    plot_distribution(grouped_race_counts, list(RACE_GROUPINGS.keys()), 'Race', args.output_dir, args.fig_size, args.equal_line, args.save_svg)

    # Age plot with new groupings
    age_counts = df['age'].value_counts().to_dict()
    grouped_age_counts = group_data(age_counts, AGE_GROUPINGS)
    plot_distribution(grouped_age_counts, list(AGE_GROUPINGS.keys()), 'Age', args.output_dir, args.fig_size, args.equal_line, args.save_svg)

    if args.print_counts:
        print_class_counts(gender_counts, GENDER_CLASSES, 'Gender')
        print_class_counts(grouped_race_counts, RACE_GROUPINGS.keys(), 'Race')
        print_class_counts(grouped_age_counts, AGE_GROUPINGS.keys(), 'Age')

    print(f"Plots have been saved in the '{args.output_dir}' directory.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate distribution plots from CSV data.")
    parser.add_argument("--input_dir", default="outputs/FTF_r_bs64_occs500_legacy-gender.csv", type=str, help="Path to the input CSV file")
    parser.add_argument("--output_dir", default="evaluations/legacy_gender/FTF_r_bs64_occs500", type=str, help="Directory to save the output plots")
    parser.add_argument("--fig_size", default=[16, 10], type=float, nargs=2, help="Figure size (width height)")
    parser.add_argument("--equal_line", default=True, action="store_true", help="Add a line for equal distribution")
    parser.add_argument("--save_svg", default=True, action="store_true", help="Save plots as SVG in addition to PNG")
    parser.add_argument("--print_counts", default=True, action="store_true", help="Print absolute counts for each class")

    args = parser.parse_args()
    main(args)