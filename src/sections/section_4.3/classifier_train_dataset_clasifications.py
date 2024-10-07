"""
H-Space Classifier Training for DebiasDiffusion

This script trains h-space classifiers for attribute prediction in the DebiasDiffusion project.
It supports training on self-labeled, classifier-labeled, and one-hot encoded datasets for
gender, race, and age attributes.

Usage:
    python src/sections/section_4.3/classifier_train_dataset_classifications.py [--args]

Arguments:
    --dataset_path: Path to the dataset file (default: BASE_DIR / "data/experiments/section_4.3/h_space_data/dataset_5k.pt")
    --output_path: Directory to save results (default: BASE_DIR / "results/section_4.3/h_space_classifiers")
    --batch_size: Batch size for training (default: 256)
    --use_fp16: Use half precision for training (default: True)
    --attributes: List of attributes to train classifiers for (default: gender race age)
    --dataset_sizes: List of dataset sizes to train on (default: 5k)
    --methods: List of methods to train (default: self_labeled cls_labeled cls_labeled_oh)
    --epochs: Number of training epochs (default: 100)
    --lr: Learning rate (default: 1e-4)
    --save_interval: Interval for saving model checkpoints (default: 3)
    --seed: Random seed for reproducibility (default: 42)

Outputs:
    - Trained h-space classifier models saved as PyTorch files
    - Training logs and metrics saved as CSV files
    - Plots of training progress and evaluation results
"""

import os
import sys
import torch
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Tuple
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# Add project root to Python path
SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent.parent.parent
sys.path.append(str(BASE_DIR))

from src.utils.classifier import make_classifier_model
from src.utils.logger import get_logger
from src.utils.plotting_utils import plot_accuracy, plot_confusion_matrix, save_plot

def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_dataset(dataset_path: Path) -> Dict[str, torch.Tensor]:
    """Load the h-space dataset."""
    data = torch.load(dataset_path)
    return {
        'h_vectors': torch.stack([sample['h_debiased'] for sample in data]),
        'labels': {attr: torch.stack([sample['labels'][attr] for sample in data]) for attr in data[0]['labels']}
    }

def prepare_data(data: Dict[str, torch.Tensor], attribute: str, method: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """Prepare data for training based on the specified method."""
    h_vectors = data['h_vectors']
    labels = data['labels'][attribute]
    
    if method == 'cls_labeled_oh':
        labels = torch.argmax(labels, dim=1)
    elif method == 'self_labeled':
        labels = torch.argmax(labels, dim=1)
    
    return h_vectors, labels

def train_epoch(model: torch.nn.Module, dataloader: DataLoader, optimizer: torch.optim.Optimizer, 
                criterion: torch.nn.Module, device: torch.device) -> float:
    """Train the model for one epoch."""
    model.train()
    total_loss = 0
    for h_vectors, labels in dataloader:
        h_vectors, labels = h_vectors.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(h_vectors)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def evaluate(model: torch.nn.Module, dataloader: DataLoader, criterion: torch.nn.Module, 
             device: torch.device) -> Tuple[float, float, np.ndarray, Dict[str, float]]:
    """Evaluate the model on the given dataset."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for h_vectors, labels in dataloader:
            h_vectors, labels = h_vectors.to(device), labels.to(device)
            outputs = model(h_vectors)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    cm = confusion_matrix(all_labels, all_preds)
    cr = classification_report(all_labels, all_preds, output_dict=True)
    return total_loss / len(dataloader), accuracy, cm, cr

def main(args: argparse.Namespace):
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    
    logger = get_logger(args.output_path, f"{args.attribute}_{args.method}_{args.dataset_size}")
    
    data = load_dataset(args.dataset_path)
    h_vectors, labels = prepare_data(data, args.attribute, args.method)
    
    dataset = TensorDataset(h_vectors, labels)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    
    num_classes = len(torch.unique(labels))
    model = make_classifier_model(in_channels=1280, image_size=8, out_channels=num_classes)
    model = model.to(device)
    
    if args.use_fp16:
        model = model.half()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss()
    
    best_val_accuracy = 0
    train_losses, val_losses, train_accuracies, val_accuracies = [], [], [], []
    
    for epoch in tqdm(range(args.epochs), desc="Training"):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_accuracy, val_cm, val_cr = evaluate(model, val_loader, criterion, device)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        _, train_accuracy, _, _ = evaluate(model, train_loader, criterion, device)
        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)
        
        logger.info(f"Epoch {epoch+1}/{args.epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                    f"Train Acc: {train_accuracy:.4f}, Val Acc: {val_accuracy:.4f}")
        
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), args.output_path / f"best_model_{args.attribute}_{args.method}_{args.dataset_size}.pt")
        
        if (epoch + 1) % args.save_interval == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, args.output_path / f"checkpoint_{args.attribute}_{args.method}_{args.dataset_size}_epoch_{epoch+1}.pt")
    
    # Plot training progress
    plot_accuracy({'train': train_accuracies, 'val': val_accuracies}, 
                  args.output_path / f"accuracy_{args.attribute}_{args.method}_{args.dataset_size}.png")
    
    # Plot confusion matrix
    plot_confusion_matrix(val_cm, list(range(num_classes)), 
                          args.output_path / f"confusion_matrix_{args.attribute}_{args.method}_{args.dataset_size}.png")
    
    # Save classification report
    pd.DataFrame(val_cr).transpose().to_csv(args.output_path / f"classification_report_{args.attribute}_{args.method}_{args.dataset_size}.csv")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train h-space classifiers for DebiasDiffusion")
    parser.add_argument("--dataset_path", type=Path, default=BASE_DIR / "data/experiments/section_4.3/h_space_data/dataset_5k.pt", 
                        help="Path to the dataset file")
    parser.add_argument("--output_path", type=Path, default=BASE_DIR / "results/section_4.3/h_space_classifiers", 
                        help="Directory to save results")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for training")
    parser.add_argument("--use_fp16", action="store_true", help="Use half precision for training")
    parser.add_argument("--attributes", nargs='+', default=['gender', 'race', 'age'], help="List of attributes to train classifiers for")
    parser.add_argument("--dataset_sizes", nargs='+', default=['5k'], help="List of dataset sizes to train on")
    parser.add_argument("--methods", nargs='+', default=['self_labeled', 'cls_labeled', 'cls_labeled_oh'], 
                        help="List of methods to train")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--save_interval", type=int, default=3, help="Interval for saving model checkpoints")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--no_cuda", action="store_true", help="Disable CUDA")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args)