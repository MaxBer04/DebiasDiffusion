"""
Utility functions for image handling and manipulation in the DebiasDiffusion project.

This module provides functions for creating image grids and saving images,
which are commonly used across various experiments in the project.
"""

import os
from pathlib import Path
from typing import List

import torch
from PIL import Image
from torchvision.utils import make_grid


def create_image_grid(images: List[Image.Image], rows: int, cols: int) -> Image.Image:
    """
    Create a grid of images from a list of PIL Images.

    Args:
        images (List[Image.Image]): List of PIL Images to arrange in a grid.
        rows (int): Number of rows in the grid.
        cols (int): Number of columns in the grid.

    Returns:
        Image.Image: A new PIL Image containing the grid of input images.
    """
    w, h = images[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    for i, image in enumerate(images):
        grid.paste(image, box=(i%cols*w, i//cols*h))
    return grid


def save_images(images: List[Image.Image], 
                output_dir: Path, 
                prompts: List[str], 
                num_images_per_prompt: int,
                grid_cols: int, 
                seed: int) -> None:
    """
    Save individual images and create image grids for each prompt.

    Args:
        images (List[Image.Image]): List of generated images.
        output_dir (Path): Directory to save the images and grids.
        prompts (List[str]): List of prompts used to generate the images.
        num_images_per_prompt (int): Number of images generated for each prompt.
        grid_cols (int): Number of columns in the image grid.
        seed (int): Random seed used for image generation.

    Returns:
        None
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for i, prompt in enumerate(prompts):
        prompt_slug = "_".join(prompt.split()[:10])  # Use first 10 words of prompt for filename
        prompt_dir = output_dir / prompt_slug
        prompt_dir.mkdir(parents=True, exist_ok=True)
        
        prompt_images = images[i*num_images_per_prompt : (i+1)*num_images_per_prompt]
        
        # Save individual images
        for j, image in enumerate(prompt_images):
            image.save(prompt_dir / f"{prompt_slug}_{seed}_{j:04d}.png")
        
        # Create and save image grid
        rows = (len(prompt_images) + grid_cols - 1) // grid_cols
        grid = create_image_grid(prompt_images, rows, grid_cols)
        grid.save(output_dir / f"{prompt_slug}_{seed}_grid.png")


def tensor_to_pil(images: torch.Tensor) -> List[Image.Image]:
    """
    Convert a batch of tensor images to a list of PIL Images.

    Args:
        images (torch.Tensor): Tensor of shape (N, C, H, W) containing the images.

    Returns:
        List[Image.Image]: List of PIL Images.
    """
    images = (images / 2 + 0.5).clamp(0, 1)
    images = images.cpu().permute(0, 2, 3, 1).numpy()
    return [Image.fromarray((image * 255).astype("uint8")) for image in images]


def make_grid_from_tensors(images: torch.Tensor, nrow: int = 8) -> Image.Image:
    """
    Create an image grid from a batch of tensor images.

    Args:
        images (torch.Tensor): Tensor of shape (N, C, H, W) containing the images.
        nrow (int): Number of images displayed in each row of the grid.

    Returns:
        Image.Image: PIL Image containing the grid of images.
    """
    grid = make_grid(images, nrow=nrow, padding=2, normalize=True)
    return tensor_to_pil(grid.unsqueeze(0))[0]