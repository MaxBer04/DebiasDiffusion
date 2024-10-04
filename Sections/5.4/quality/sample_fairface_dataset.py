import os
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from datasets import load_dataset, concatenate_datasets
from tqdm import tqdm
from collections import Counter

SCRIPT_DIR = Path(__file__).resolve().parent

def load_fairface_dataset(subset="1.25"):
    train_dataset = load_dataset("HuggingFaceM4/FairFace", subset, split="train")
    validation_dataset = load_dataset("HuggingFaceM4/FairFace", subset, split="validation")
    return concatenate_datasets([train_dataset, validation_dataset])

def process_attributes(dataset):
    processed_data = []
    for idx, item in enumerate(tqdm(dataset, desc="Processing attributes")):
        age = 'Young' if item['age'] in [0, 1, 2, 3, 4, 5] else 'Old'
        race = item['race']
        if race in [0, 6]:  # East Asian or Southeast Asian
            race = 'Asian'
        elif race in [3, 4, 5]:  # White, Middle Eastern, or Latino_Hispanic
            race = 'White'
        elif race == 1:  # Indian
            race = 'Indian'
        elif race == 2:  # Black
            race = 'Black'
        else:
            continue  # Skip if race is not in our categories

        processed_data.append({
            'idx': idx,
            'gender': 'Male' if item['gender'] == 0 else 'Female',
            'age': age,
            'race': race
        })
    return pd.DataFrame(processed_data)

def load_target_distribution(csv_path):
    if csv_path is None:
        return None
    df = pd.read_csv(csv_path)
    target_dist = {}

    for column in df.columns:
        counts = df[column].value_counts()
        total = counts.sum()
        target_dist[column] = {category: count / total for category, count in counts.items()}

    return target_dist

def create_balanced_distribution(df, selected_attributes):
    target_dist = {}
    for column in selected_attributes:
        categories = df[column].unique()
        target_dist[column] = {category: 1/len(categories) for category in categories}
    return target_dist

def sample_dataset(df, total_samples, target_dist):
    sampled_df = pd.DataFrame()

    for attribute, dist in target_dist.items():
        target_counts = {cls: int(prob * total_samples) for cls, prob in dist.items()}

        for cls, target_count in target_counts.items():
            available = df[df[attribute] == cls]
            if len(available) < target_count:
                print(f"Warning: Not enough {cls} samples for {attribute}. "
                      f"Requested {target_count}, but only {len(available)} available.")
                sampled = available
            else:
                sampled = available.sample(target_count, replace=False)

            sampled_df = pd.concat([sampled_df, sampled], ignore_index=True)

    # Remove duplicates (as some samples might have been selected multiple times)
    sampled_df = sampled_df.drop_duplicates(subset='idx')

    # If we don't have enough samples, add random samples to reach the total
    if len(sampled_df) < total_samples:
        remaining = total_samples - len(sampled_df)
        print(f"Warning: Adding {remaining} random samples to reach the total.")
        additional_samples = df[~df.index.isin(sampled_df.index)].sample(remaining)
        sampled_df = pd.concat([sampled_df, additional_samples], ignore_index=True)

    # If we have too many samples, randomly remove some
    elif len(sampled_df) > total_samples:
        print(f"Warning: Removing {len(sampled_df) - total_samples} random samples to reach the target total.")
        sampled_df = sampled_df.sample(total_samples)

    return sampled_df

def save_sampled_images(dataset, sampled_df, output_dir, use_group_folders, selected_attributes):
    os.makedirs(output_dir, exist_ok=True)
    for _, row in tqdm(sampled_df.iterrows(), total=len(sampled_df), desc="Saving images"):
        img = dataset[row['idx']]['image']
        group_name = "_".join([str(row[attr]) for attr in selected_attributes])
        file_name = f"{group_name}_{row['idx']}.jpg"

        if use_group_folders:
            group_dir = os.path.join(output_dir, group_name)
            os.makedirs(group_dir, exist_ok=True)
            file_path = os.path.join(group_dir, file_name)
        else:
            file_path = os.path.join(output_dir, file_name)

        img.save(file_path)

def print_statistics(df, sampled_df, target_dist):
    print("\nDataset Statistics:")
    print(f"{'Category':<10} {'Class':<10} {'Original %':<12} {'Sampled %':<12} {'Target %':<10}")
    print("-" * 60)

    for category in target_dist.keys():
        original_dist = df[category].value_counts(normalize=True)
        sampled_dist = sampled_df[category].value_counts(normalize=True)

        for cls in target_dist[category].keys():
            original_pct = original_dist.get(cls, 0) * 100
            sampled_pct = sampled_dist.get(cls, 0) * 100
            target_pct = target_dist[category][cls] * 100

            print(f"{category:<10} {cls:<10} {original_pct:.2f}%{' ':6} {sampled_pct:.2f}%{' ':6} {target_pct:.2f}%")
        print()

def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading dataset...")
    dataset = load_fairface_dataset(subset=args.subset)

    print("Processing attributes...")
    processed_df = process_attributes(dataset)

    selected_attributes = [attr for attr in ['gender', 'race', 'age'] if getattr(args, attr)]

    if not selected_attributes:
        print("Error: At least one attribute (gender, race, or age) must be selected.")
        return

    if args.target_dist:
        print(f"Loading target distribution from {args.target_dist}")
        target_dist = load_target_distribution(args.target_dist)
        target_dist = {k: v for k, v in target_dist.items() if k in selected_attributes}
    else:
        print("Creating balanced distribution...")
        target_dist = create_balanced_distribution(processed_df, selected_attributes)

    print("Sampling dataset according to distribution...")
    sampled_df = sample_dataset(processed_df, args.num_samples, target_dist)

    print(f"Saving {len(sampled_df)} images to {args.output_dir}")
    save_sampled_images(dataset, sampled_df, args.output_dir, args.use_group_folders, selected_attributes)

    print_statistics(processed_df, sampled_df, target_dist)

    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sample dataset from FairFace with target or balanced distribution")
    parser.add_argument("--num_samples", type=int, default=9023, help="Total number of samples to generate")
    parser.add_argument("--subset", type=str, default="1.25", choices=["0.25", "1.25"], help="FairFace subset to use")
    parser.add_argument("--output_dir", type=str, default=SCRIPT_DIR / "datasets_sampled/AS_g", help="Output directory for sampled images")
    parser.add_argument("--use_group_folders", action="store_true", default=False, help="Save images in group-specific folders")
    parser.add_argument("--target_dist", type=str, default=None, help="Path to CSV file with target distribution")
    parser.add_argument("--gender", action="store_true", default=True, help="Include gender in sampling")
    parser.add_argument("--race", action="store_true", default=False, help="Include race in sampling")
    parser.add_argument("--age", action="store_true", default=False, help="Include age in sampling")
    args = parser.parse_args()
    main(args)