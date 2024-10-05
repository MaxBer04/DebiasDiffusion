import argparse
import os
from datasets import load_dataset
import pandas as pd
import matplotlib.pyplot as plt

def download_dataset(dataset_name, config, split):
    """
    Download the specified dataset from Hugging Face with the given configuration.
    """
    return load_dataset(dataset_name, config, split=split)

def analyze_dataset_structure(dataset):
    """
    Analyze the structure of the dataset.
    """
    print(f"Dataset info:\n{dataset}")
    print(f"\nFeatures:\n{dataset.features}")
    print(f"\nNumber of rows: {len(dataset)}")

def analyze_column_statistics(dataset):
    """
    Analyze statistics for each column in the dataset.
    """
    df = dataset.to_pandas()
    print("\nColumn statistics:")
    print(df.describe())

def visualize_category_distribution(dataset, category):
    """
    Visualize the distribution of a categorical column.
    """
    df = dataset.to_pandas()
    value_counts = df[category].value_counts()
    
    plt.figure(figsize=(10, 6))
    value_counts.plot(kind='bar')
    plt.title(f'Distribution of {category}')
    plt.xlabel(category)
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'{category}_distribution.png')
    print(f"Distribution plot saved as {category}_distribution.png")

def main(args):
    dataset = download_dataset(args.dataset, args.config, args.split)
    
    analyze_dataset_structure(dataset)
    analyze_column_statistics(dataset)
    
    if args.category:
        visualize_category_distribution(dataset, args.category)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and analyze a Hugging Face dataset.")
    parser.add_argument("--dataset", type=str, default="HuggingFaceM4/FairFace", help="Name of the dataset on Hugging Face")
    parser.add_argument("--config", type=str, choices=['0.25', '1.25'], default='1.25', help="Configuration of the dataset")
    parser.add_argument("--split", type=str, default="train", help="Dataset split to use (e.g., 'train', 'test')")
    parser.add_argument("--category", type=str, default="gender", help="Categorical column to visualize distribution")
    
    args = parser.parse_args()
    main(args)