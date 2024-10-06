"""
H-Space Classifier Evaluation for DebiasDiffusion

This script evaluates the performance of h-space classifiers used in the DebiasDiffusion project.
It loads a pre-created dataset of h-vectors with labels, applies the classifiers, and generates
performance plots and metrics.

Usage:
    python src/sections/section_6.2/evaluate_dataset.py [--args]

Arguments:
    --dataset_path: Path to the dataset file (default: BASE_DIR / "data/experiments/section_6.2/h_space_data/dataset_5k.pt")
    --output_path: Directory to save results (default: BASE_DIR / "results/section_6.2/h_space_evaluation")
    --batch_size: Batch size for evaluation (default: 256)
    --use_fp16: Use half precision for evaluation (default: True)
    --attributes: List of attributes to evaluate (default: gender race age)
    --dataset_sizes: List of dataset sizes to evaluate (default: 5k)
    --methods: List of methods to evaluate (default: qq qqff qqff_v2)
    --fontsize_base: Base font size for plots (default: 14)
    --fontsize_label: Font size for axis labels (default: 16)
    --fontsize_title: Font size for plot titles (default: 16)
    --fontsize_tick: Font size for tick labels (default: 14)
    --fontsize_legend: Font size for legends (default: 16)
    --fontsize_text: Font size for additional text in plots (default: 14)

Outputs:
    - PNG and SVG plots of classifier accuracy over timesteps
    - CSV file with evaluation results
    - Console output with evaluation summary
"""

import argparse
import os
import sys
import json
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import numpy as np
import torch
import pandas as pd
from typing import Dict, List, Tuple
from pathlib import Path

# Add project root to Python path
SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent.parent.parent
sys.path.append(str(BASE_DIR))

from src.utils.classifier import make_classifier_model
from src.utils.plotting_utils import save_plot

def load_classifier(classifier_path: str, device: torch.device, num_classes: int, model_type: str, use_fp16: bool = True) -> torch.nn.Module:
    """
    Load a classifier model from a file.

    Args:
        classifier_path (str): Path to the classifier model file.
        device (torch.device): Device to load the model on.
        num_classes (int): Number of output classes for the classifier.
        model_type (str): Type of classifier model.
        use_fp16 (bool): Whether to use half precision.

    Returns:
        torch.nn.Module: Loaded classifier model.
    """
    classifier = make_classifier_model(
        in_channels=1280,
        image_size=8,
        out_channels=num_classes,
        model_type=model_type
    )
    state_dict = torch.load(classifier_path, map_location=device)
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    classifier.load_state_dict(new_state_dict)
    
    if use_fp16:
        classifier = classifier.half()
    else:
        classifier = classifier.float()
    
    return classifier.to(device)

def load_dataset(dataset_path: str) -> torch.Tensor:
    """
    Load the evaluation dataset.

    Args:
        dataset_path (str): Path to the dataset file.

    Returns:
        torch.Tensor: Loaded dataset.
    """
    return torch.load(dataset_path)

def mask_h_vectors(h_debiased: torch.Tensor, timestep: int) -> torch.Tensor:
    """
    Mask h-space vectors for timesteps after the current one.

    Args:
        h_debiased (torch.Tensor): H-space vectors.
        timestep (int): Current timestep.

    Returns:
        torch.Tensor: Masked h-space vectors.
    """
    masked = h_debiased.clone()
    masked[:, timestep+1:] = 0
    return masked

def evaluate_classifiers(classifiers: Dict[str, Tuple[torch.nn.Module, str, str]], 
                         dataset: List[Dict[str, torch.Tensor]], 
                         device: torch.device, 
                         batch_size: int, 
                         use_fp16: bool) -> Dict[str, np.ndarray]:
    """
    Evaluate the performance of classifiers on the dataset.

    Args:
        classifiers (Dict[str, Tuple[torch.nn.Module, str, str]]): Dictionary of classifiers.
        dataset (List[Dict[str, torch.Tensor]]): Evaluation dataset.
        device (torch.device): Device to run evaluation on.
        batch_size (int): Batch size for evaluation.
        use_fp16 (bool): Whether to use half precision.

    Returns:
        Dict[str, np.ndarray]: Evaluation results for each classifier.
    """
    results = {name: [] for name in classifiers}
    
    num_batches = len(dataset) // batch_size
    
    for batch_idx in tqdm(range(num_batches), desc="Evaluating classifiers"):
        start_idx = batch_idx * batch_size
        end_idx = start_idx + batch_size
        batch_data = dataset[start_idx:end_idx]
        
        h_debiased = torch.stack([data['h_debiased'] for data in batch_data]).to(device)
        if use_fp16:
            h_debiased = h_debiased.half()
        labels = {attr: torch.tensor([data['labels'][attr] for data in batch_data], device=device) 
                  for attr in batch_data[0]['labels']}
        
        for name, (classifier, attr, model_type) in classifiers.items():
            if model_type == "linear":
                predictions = []
                for t in range(50):
                    masked_h = mask_h_vectors(h_debiased, t)
                    logits = classifier(masked_h[:, t], torch.full((batch_size,), t, device=device))
                    preds = torch.argmax(logits, dim=1)
                    correct = (preds == labels[attr]).float()
                    predictions.append(correct.mean().item())
                results[name].append(predictions)
            else:  # "multi_layer" or "resnet18"
                predictions = []
                for t in range(50):
                    masked_h = mask_h_vectors(h_debiased, t)
                    logits = classifier(masked_h[:, t], torch.full((batch_size,), t, device=device))
                    preds = torch.argmax(logits, dim=1)
                    correct = (preds == labels[attr]).float()
                    predictions.append(correct.mean().item())
                results[name].append(predictions)
    
    return {name: np.mean(preds, axis=0) for name, preds in results.items()}

def plot_results(results: Dict[str, np.ndarray], 
                 output_path: Path, 
                 font_sizes: Dict[str, int]) -> None:
    """
    Plot the evaluation results.

    Args:
        results (Dict[str, np.ndarray]): Evaluation results for each classifier.
        output_path (Path): Path to save the plot.
        font_sizes (Dict[str, int]): Font sizes for different plot elements.
    """
    plt.figure(figsize=(12, 8))
    sns.set_style("whitegrid")

    color_palette = plt.cm.Set2.colors + plt.cm.Set3.colors + plt.cm.tab20.colors

    attributes = set(attr.split('_')[0] for attr in results.keys())
    colors = plt.cm.get_cmap('tab10')(np.linspace(0, 1, len(attributes)))
    color_dict = dict(zip(attributes, colors))

    linestyles = [
        'solid', 'dotted', 'dashed', 'dashdot', 
        (0, (1, 1)), (0, (5, 5)), (0, (3, 5, 1, 5)), (0, (3, 1, 1, 1)),
        (0, (1, 10)), (0, (5, 10)), (0, (3, 10, 1, 10)), (0, (3, 5, 1, 5, 1, 5)),
        (0, (3, 1, 1, 1, 1, 1)), (0, (5, 1)), (0, (1, 1, 5, 5)), (0, (5, 5, 1, 5))
    ]

    grouped_results = {}
    for name, accuracies in results.items():
        attr = name.split('_')[0]
        if attr not in grouped_results:
            grouped_results[attr] = []
        grouped_results[attr].append((name, accuracies))

    for attr, group in grouped_results.items():
        color = color_dict[attr]
        for i, (name, accuracies) in enumerate(group):
            linestyle = linestyles[i % len(linestyles)]
            plt.plot(range(50), accuracies * 100, label=f"{attr.capitalize()} - {name}",
                     color=color, linestyle=linestyle)

    plt.xlabel("Timestep", fontsize=font_sizes['label'])
    plt.ylabel("Accuracy (%)", fontsize=font_sizes['label'])
    plt.ylim(0, 100)
    plt.title("Classifier Accuracy over Diffusion Timesteps", fontsize=font_sizes['title'])
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=font_sizes['legend'])
    plt.tight_layout()
    
    save_plot(plt, output_path, dpi=300)

def main(args: argparse.Namespace) -> None:
    """
    Main function to run the evaluation.

    Args:
        args (argparse.Namespace): Command-line arguments.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    dataset = load_dataset(args.dataset_path)
    
    classifiers = {}
    for attr in args.attributes:
        for size in args.dataset_sizes:
            for method in args.methods:
                name = f"{attr}_{size}_{method}"
                path = BASE_DIR / f"data/model_data/h_space_classifiers/{method}/{size}/{attr}_{size}_e100_bs256_lr0.0001_tv0.8/best_model.pt"
                num_classes = 2 if attr in ['gender', 'age'] else 4
                classifiers[name] = (load_classifier(str(path), device, num_classes, "linear", args.use_fp16), attr, "linear")
    
    results = evaluate_classifiers(classifiers, dataset, device, args.batch_size, args.use_fp16)
    
    output_dir = Path(args.output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    font_sizes = {
        'base': args.fontsize_base,
        'label': args.fontsize_label,
        'title': args.fontsize_title,
        'tick': args.fontsize_tick,
        'legend': args.fontsize_legend,
        'text': args.fontsize_text
    }

    plot_results(results, output_dir / "classifier_accuracy", font_sizes)

    # Save results to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_dir / "classifier_accuracy.csv", index=False)

    print(f"Evaluation completed. Results saved to {output_dir}")

def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Evaluate h-space classifiers for DebiasDiffusion")
    parser.add_argument("--dataset_path", type=str, default=str(BASE_DIR / "data/experiments/section_6.2/h_space_data/dataset_5k.pt"), 
                        help="Path to the dataset file")
    parser.add_argument("--output_path", type=str, default=str(BASE_DIR / "results/section_6.2/h_space_evaluation"), 
                        help="Directory to save results")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for evaluation")
    parser.add_argument("--use_fp16", action="store_true", default=True, help="Use half precision for evaluation")
    parser.add_argument("--attributes", nargs='+', default=['gender', 'race', 'age'], 
                        help="List of attributes to evaluate")
    parser.add_argument("--dataset_sizes", nargs='+', default=['5k'], 
                        help="List of dataset sizes to evaluate")
    parser.add_argument("--methods", nargs='+', default=['qq', 'qqff', 'qqff_v2'], 
                        help="List of methods to evaluate")
    parser.add_argument("--fontsize_base", type=int, default=14, help="Base font size")
    parser.add_argument("--fontsize_label", type=int, default=16, help="Font size for axis labels")
    parser.add_argument("--fontsize_title", type=int, default=16, help="Font size for plot titles")
    parser.add_argument("--fontsize_tick", type=int, default=14, help="Font size for tick labels")
    parser.add_argument("--fontsize_legend", type=int, default=16, help="Font size for legends")
    parser.add_argument("--fontsize_text", type=int, default=14, help="Font size for additional text in plots")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args)