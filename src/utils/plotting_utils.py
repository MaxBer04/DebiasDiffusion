from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from torchvision.utils import make_grid
import os
import sys
import math
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from utils.general import ensure_directory, normalize_img_for_imshow, is_image_file, remove_image_filename

def plot_images_with_attributes(images, probs_dict, attribute_labels_dict, save_path, nrow=6, show_legend=True, dpi=300):
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

def plot_accuracy(accuracies_dict, save_path):
    """Plot accuracy over time for each attribute."""
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

def plot_confusion_matrix(y_pred, y_true, labels, save_path):
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
    Plots a simple loss.
    
    Args:
    losses (list of float): List containing loss values for each epoch.
    filename (str): Name of the file to save the plot. Defaults to 'loss_plot.png'.
    figsize(tuple of ints): The size of the figure.
    show (bool): Show the plot in the end.
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


def save_image_row(tensor_list, save_path):
    """
    Takes a list of Torch tensors with shape [batch size, channels, resolution, resolution]
    and saves them as an image row in a specified path.

    :param tensor_list: List of Torch tensors. Each tensor has shape [channels, resolution, resolution],
                        where `channels` can be 1 (for grayscale) or 3 (for RGB).
    :param save_path:   String. The file path where the resulting image row should be saved.
    """
    ensure_directory(save_path)
    
    # Convert tensors to NumPy arrays
    image_arrays = [tensor.numpy().transpose(1, 2, 0) for tensor in tensor_list]

    # Normalize each image
    image_arrays = [normalize_img_for_imshow(img) for img in image_arrays]

    # Create a single row of images
    fig, axes = plt.subplots(1, len(image_arrays), figsize=(len(image_arrays) * 2, 2))

    # Plot each image in its own subplot
    for ax, img in zip(axes, image_arrays):
        ax.imshow(img)
        ax.axis('off')

    # Save the figure
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    

def save_pil_image_row(images, prompt, seed, output_directory):
    """
    Takes a list of PIL Images and saves them as an image row in a specified path.

    :param pil_image_list: List of PIL Images.
    :param save_path: String. The file path where the resulting image row should be saved.
    """
    """
    Saves a list of PIL Images as a grid.

    :param images: List of PIL Images to be arranged in a grid.
    :param prompt: String to identify the prompt or source of the images.
    :param seed: Integer seed value used in the generation of images, appended to the file name.
    :param output_directory: Directory path where the resulting image will be saved.

    Creates a new PNG image that consists of a grid of the provided images.
    The grid dimensions are calculated to fit all images with equal width and height.
    """
    # Create output directory if it does not exist
    file_prefix = remove_image_filename(output_directory)
    if not os.path.exists(file_prefix):
        os.makedirs(file_prefix)

    # Calculate the size of the grid based on the number of images
    num_images = len(images)
    grid_size = num_images

    # Get max dimensions to maintain uniformity
    max_width = max(image.width for image in images)
    max_height = max(image.height for image in images)

    # Create a new blank canvas for the grid
    row_image = Image.new('RGB', (max_width * grid_size, max_height))

    # Insert each image into its corresponding position
    for idx, image in enumerate(images):
        row_image.paste(image, (idx * max_width, 0))

    # Save the final image
    if not is_image_file(output_directory):
        file_name = f"{output_directory}/{prompt}_{seed}.png"
    else:
        file_name = output_directory
    row_image.save(file_name) 
    
    

def save_image_grid_with_borders(images, prompt, seed, output_directory, num_cols=10, tau_1=5, tau_2=27):
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
    print(f"Bild gespeichert unter: {file_name}")
    


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



def plot_attention_map_histogram(attention_map, filename):
    ensure_directory(filename)

    # Flatten the attention map if it's a matrix
    if len(attention_map.shape) > 1:
        attention_map = attention_map.flatten()
    
    # Create histogram
    plt.figure()
    plt.hist(attention_map, bins=50, alpha=0.75, color='blue')
    plt.title('Histogram of Attention Map Values')
    plt.xlabel('Attention Value')
    plt.ylabel('Frequency')
    
    # Save plot to file
    plt.savefig(filename)
    plt.close()