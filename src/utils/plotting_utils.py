"""
Plotting Utilities for DebiasDiffusion

This module provides various plotting functions used in the DebiasDiffusion project
for visualizing results, creating image grids, and saving plots.

Functions:
    plot_images_with_attributes: Create a grid of images with attribute probabilities.
    plot_accuracy: Plot accuracy over time for different attributes.
    plot_confusion_matrix: Create and save a confusion matrix plot.
    plot_loss: Plot and save a loss curve.
    save_image_row: Save a row of images.
    save_pil_image_row: Save a row of PIL images.
    save_image_grid_with_borders: Save a grid of images with colored borders.
    save_image_grid: Save a grid of images.
    plot_attention_map_histogram: Plot and save a histogram of attention map values.

Usage:
    from src.utils.plotting_utils import plot_images_with_attributes, save_image_grid

    plot_images_with_attributes(images, probs_dict, attribute_labels_dict, save_path)
    save_image_grid(images, prompt, seed, output_directory)

Note:
    This module requires matplotlib, seaborn, PIL, and torch to be installed.
"""

import sys
import os
import math
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
import numpy as np
import seaborn as sns
import torch
from torchvision.utils import make_grid
from typing import List, Dict, Tuple, Union, Optional

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from utils.general import ensure_directory, normalize_img_for_imshow, is_image_file, remove_image_filename


def save_plot(fig: plt.Figure, output_path: Path, dpi: int = 300) -> None:
    """
    Save the given matplotlib figure as both PNG and SVG files.

    Args:
        fig (plt.Figure): The matplotlib figure to save.
        output_path (Path): The base path to save the files (without extension).
        dpi (int, optional): The resolution in dots per inch. Defaults to 300.

    Returns:
        None
    """
    # Ensure the directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save as PNG
    png_path = output_path.with_suffix('.png')
    fig.savefig(png_path, dpi=dpi, bbox_inches='tight')
    print(f"Plot saved as PNG: {png_path}")

    # Save as SVG
    svg_path = output_path.with_suffix('.svg')
    fig.savefig(svg_path, format='svg', bbox_inches='tight')
    print(f"Plot saved as SVG: {svg_path}")

    # Close the plot to free up memory
    plt.close(fig)

def plot_images_with_attributes(
    images: torch.Tensor,
    probs_dict: Dict[str, torch.Tensor],
    attribute_labels_dict: Dict[str, List[str]],
    save_path: Union[str, Path],
    nrow: int = 6,
    show_legend: bool = True,
    dpi: int = 300
) -> None:
    """
    Create a grid of images with attribute probabilities.

    Args:
        images (torch.Tensor): Tensor of images.
        probs_dict (Dict[str, torch.Tensor]): Dictionary of attribute probabilities.
        attribute_labels_dict (Dict[str, List[str]]): Dictionary of attribute labels.
        save_path (Union[str, Path]): Path to save the output image.
        nrow (int, optional): Number of images per row. Defaults to 6.
        show_legend (bool, optional): Whether to show the legend. Defaults to True.
        dpi (int, optional): DPI for the output image. Defaults to 300.
    """
    num_images, num_attributes = images.shape[0], len(probs_dict)
    img_w, img_h = images.shape[3], images.shape[2]
    bar_width, bar_height = img_w // 4, img_h
    bar_padding = 2
    bars_total_width = (bar_width + bar_padding) * num_attributes
    
    grid_w = nrow * (img_w + bars_total_width + 10)
    grid_h = ((num_images - 1) // nrow + 1) * (img_h + 10)
    full_h = grid_h + (100 if show_legend else 0)
    
    full_img = Image.new('RGB', (grid_w, full_h), color='white')
    draw = ImageDraw.Draw(full_img)
    
    color_maps = {
        'gender': {'male': (100, 149, 237), 'female': (255, 182, 193)},
        'race': {'white': (255, 255, 255), 'black': (0, 0, 0), 'asian': (255, 255, 0), 'indian': (165, 42, 42)},
        'age': {'young': (152, 251, 152), 'old': (169, 169, 169)}
    }
    
    for idx in range(num_images):
        row, col = idx // nrow, idx % nrow
        x, y = col * (img_w + bars_total_width + 10), row * (img_h + 10)
        
        img_np = images[idx].mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).byte().cpu().numpy()
        full_img.paste(Image.fromarray(img_np), (x, y))
        
        for attr_idx, (attr_type, probs) in enumerate(probs_dict.items()):
            bar_x = x + img_w + attr_idx * (bar_width + bar_padding)
            draw.rectangle([bar_x, y, bar_x + bar_width, y + bar_height], fill='lightgrey', outline='grey')
            
            cum_height = 0
            for class_idx, (attr, prob) in enumerate(zip(attribute_labels_dict[attr_type], probs[idx].squeeze())):
                section_height = int(bar_height * prob.item())
                fill_color = color_maps[attr_type][attr.lower()]
                draw.rectangle([bar_x, y + cum_height, bar_x + bar_width, y + cum_height + section_height], 
                            fill=fill_color, outline='black')
                cum_height += section_height
    
    if show_legend:
        legend_y = grid_h + 10
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
        x_offset = 0
        for attr_type, labels in attribute_labels_dict.items():
            for i, label in enumerate(labels):
                draw.rectangle([10 + x_offset, legend_y, 40 + x_offset, legend_y + 30], 
                               fill=color_maps[attr_type][label.lower()], outline='black')
                draw.text((45 + x_offset, legend_y + 5), label, fill=(0, 0, 0), font=font)
                x_offset += 120
            x_offset += 40
    
    full_img.save(save_path, dpi=(dpi, dpi))

def plot_accuracy(accuracies_dict: Dict[str, List[float]], save_path: Union[str, Path]) -> None:
    """
    Plot accuracy over time for each attribute.

    Args:
        accuracies_dict (Dict[str, List[float]]): Dictionary of accuracies for each attribute.
        save_path (Union[str, Path]): Path to save the plot.
    """
    plt.figure(figsize=(10, 6))
    for attr, accuracies in accuracies_dict.items():
        plt.plot(range(1, len(accuracies)+1), accuracies, label=attr)
    
    plt.title("Accuracy over Time", fontsize=16)
    plt.xlabel("Timestep", fontsize=14)
    plt.ylabel("Accuracy (%)", fontsize=14)
    plt.ylim(0, 100)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_confusion_matrix(y_pred: np.ndarray, y_true: np.ndarray, labels: List[str], save_path: Union[str, Path]) -> None:
    """
    Create and save a confusion matrix plot.

    Args:
        y_pred (np.ndarray): Predicted labels.
        y_true (np.ndarray): True labels.
        labels (List[str]): Label names.
        save_path (Union[str, Path]): Path to save the plot.
    """
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(save_path)
    plt.close()

def plot_loss(losses: List[float], filename: str = 'loss_plot.png', figsize: Tuple[int, int] = (12, 5), show: bool = False) -> None:
    """
    Plot and save a loss curve.

    Args:
        losses (List[float]): List of loss values.
        filename (str, optional): Name of the output file. Defaults to 'loss_plot.png'.
        figsize (Tuple[int, int], optional): Figure size. Defaults to (12, 5).
        show (bool, optional): Whether to display the plot. Defaults to False.
    """
    ensure_directory(filename)
    
    plt.figure(figsize=figsize)
    plt.plot(losses, label='Training Loss')
    plt.title('Training Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    if show:
        plt.show()
    plt.close()

def save_image_row(tensor_list: List[torch.Tensor], save_path: Union[str, Path]) -> None:
    """
    Save a row of images from a list of tensors.

    Args:
        tensor_list (List[torch.Tensor]): List of image tensors.
        save_path (Union[str, Path]): Path to save the image row.
    """
    ensure_directory(save_path)
    
    image_arrays = [tensor.numpy().transpose(1, 2, 0) for tensor in tensor_list]
    image_arrays = [normalize_img_for_imshow(img) for img in image_arrays]

    fig, axes = plt.subplots(1, len(image_arrays), figsize=(len(image_arrays) * 2, 2))
    for ax, img in zip(axes, image_arrays):
        ax.imshow(img)
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def save_pil_image_row(images: List[Image.Image], prompt: str, seed: int, output_directory: Union[str, Path]) -> None:
    """
    Save a row of PIL images.

    Args:
        images (List[Image.Image]): List of PIL Images.
        prompt (str): Prompt used to generate the images.
        seed (int): Seed used for image generation.
        output_directory (Union[str, Path]): Directory to save the image row.
    """
    file_prefix = remove_image_filename(output_directory)
    os.makedirs(file_prefix, exist_ok=True)

    num_images = len(images)
    max_width = max(image.width for image in images)
    max_height = max(image.height for image in images)

    row_image = Image.new('RGB', size=(max_width * num_images, max_height))

    for idx, image in enumerate(images):
        row_image.paste(image, box=(idx * max_width, 0))

    if not is_image_file(output_directory):
        file_name = f"{output_directory}/{prompt}_{seed}.png"
    else:
        file_name = output_directory
    row_image.save(file_name)

def save_image_grid_with_borders(
    images: List[Image.Image],
    prompt: str,
    seed: int,
    output_directory: Union[str, Path],
    num_cols: int = 10,
    tau_1: int = 5,
    tau_2: int = 27
) -> None:
    """
    Save a grid of images with colored borders.

    Args:
        images (List[Image.Image]): List of PIL Images.
        prompt (str): Prompt used to generate the images.
        seed (int): Seed used for image generation.
        output_directory (Union[str, Path]): Directory to save the image grid.
        num_cols (int, optional): Number of columns in the grid. Defaults to 10.
        tau_1 (int, optional): First tau value for border coloring. Defaults to 5.
        tau_2 (int, optional): Second tau value for border coloring. Defaults to 27.
    """
    num_images = len(images)
    num_rows = (num_images + num_cols - 1) // num_cols
    
    img_width, img_height = images[0].size
    
    line_width = 14
    padding = int(line_width / 2)
    
    grid_width = int(num_cols * (img_width + padding) - padding)
    grid_height = int(num_rows * (img_height + padding) - padding)
    
    grid_img = Image.new('RGB', (grid_width, grid_height), (255, 255, 255))
    
    for i, img in enumerate(images):
        row = i // num_cols
        col = i % num_cols
        x = col * (img_width + padding)
        y = row * (img_height + padding)
        grid_img.paste(img, (x, y))
    
    draw = ImageDraw.Draw(grid_img)
    
    def draw_individual_borders(start_idx, end_idx, color):
        for i in range(start_idx, end_idx + 1):
            row = i // num_cols
            col = i % num_cols
            x = col * (img_width + padding)
            y = row * (img_height + padding)
            draw.rectangle(
                [x, y, x + img_width, y + img_height],
                outline=color, width=line_width
            )
    
    draw_individual_borders(0, tau_1 - 2, 'blue')
    draw_individual_borders(tau_1 - 1, tau_2 - 2, 'red')
    draw_individual_borders(tau_2 - 1, num_images - 1, 'green')
    
    file_name = f"{output_directory}/{prompt}_{seed}.png"
    os.makedirs(output_directory, exist_ok=True)
    grid_img.save(file_name)
    print(f"Image saved at: {file_name}")


def save_image_grid(images: list[Image.Image], prompt: str, seed: int, output_directory: str, num_cols: int = None) -> None:
    """
    Saves a list of PIL Images as a grid.

    :param images: List of PIL Images to be arranged in a grid.
    :param prompt: String to identify the prompt or source of the images.
    :param seed: Integer seed value used in the generation of images, appended to the file name.
    :param output_directory: Directory path where the resulting image will be saved.
    :param num_cols: Number of columns in the grid.
    """
    # Create output directory if it does not exist
    file_prefix = remove_image_filename(output_directory)
    if not os.path.exists(file_prefix):
        os.makedirs(file_prefix)

    # Calculate the size of the grid based on the number of images
    num_images = len(images)
    if num_cols:
        grid_cols = num_cols
        grid_rows = math.ceil(num_images / num_cols)
    else:
        grid_cols = grid_rows = math.ceil(math.sqrt(num_images))

    # Get max dimensions to maintain uniformity
    max_width = max(image.width for image in images)
    max_height = max(image.height for image in images)

    # Create a new blank canvas for the grid
    grid_image = Image.new('RGB', (max_width * grid_cols, max_height * grid_rows))

    # Insert each image into its corresponding position
    for idx, image in enumerate(images):
        row = idx // grid_cols
        col = idx % grid_cols
        grid_image.paste(image, (col * max_width, row * max_height))

    # Save the final image
    if not is_image_file(output_directory):
        file_name = f"{output_directory}/{prompt}_{seed}.png"
    else:
        file_name = output_directory
    grid_image.save(file_name)


