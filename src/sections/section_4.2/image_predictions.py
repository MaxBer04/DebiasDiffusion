"""
Generate Image Predictions for Diffusion Models

This script generates image predictions and saves those for various timesteps during the diffusion process.
It's designed to visualize how image prediction of the model evolve throughout the denoising process, as discussed in
Section 4.2 of the associated thesis.

Usage:
    python src/sections/section_4.2/image_predictions.py [--args]

Arguments:
    --model_id: Hugging Face model ID or path to local model (default: "PalionTech/debias-diffusion-orig")
    --prompts: List of prompts to generate images from (default: ["A photo of a diplomat"])
    --output_dir: Output directory for generated images (default: "results/section_4.2/image_predictions")
    --num_images_per_prompt: Number of images to generate per prompt (default: 1)
    --num_inference_steps: Number of denoising steps (default: 50)
    --guidance_scale: Guidance scale for classifier-free guidance (default: 7.5)
    --grid_cols: Number of columns in the output grid (default: 10)
    --seed: Random seed for reproducibility (default: 9374)
    --device: Device to run the model on (default: "cuda" if available, else "cpu")

Outputs:
    - Individual image predictions over all timesteps
    - Image grid showing all predictions from first (T=50) to last (T=1) denoising step
"""

import argparse
import random
import sys
from pathlib import Path
from typing import List, Dict

import torch
from tqdm.auto import tqdm

# Add project root to Python path
SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent.parent.parent
sys.path.append(str(BASE_DIR))

from src.pipelines.testing_diffusion_pipeline import TestingDiffusionPipeline
from src.utils.plotting_utils import save_image_grid_with_borders

def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def setup_pipeline(model_id: str, device: str) -> TestingDiffusionPipeline:
    """Set up the diffusion pipeline."""
    pipeline = TestingDiffusionPipeline.from_pretrained(model_id).to(device)
    return pipeline

def generate_images(pipeline: TestingDiffusionPipeline, prompts: List[str], args: argparse.Namespace) -> List[List[torch.Tensor]]:
    """Generate images using the provided pipeline and prompts."""
    all_images = []
    for prompt in tqdm(prompts, desc="Generating images"):
        for _ in tqdm(range(args.num_images_per_prompt), desc="Per prompt"):
            img_predictions = pipeline(
                prompt, 
                num_inference_steps=args.num_inference_steps, 
                guidance_scale=args.guidance_scale, 
                generator=torch.Generator(device=pipeline.device).manual_seed(args.seed),
                return_image_predictions=True
            )[1]
            all_images.append(img_predictions)
    return all_images

def save_images(images_list: List[List[torch.Tensor]], output_dir: Path, args: argparse.Namespace) -> None:
    """Save individual images and image grids."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for i, prompt in enumerate(args.prompts):
        for run in range(args.num_images_per_prompt):
            prompt_slug = "_".join(prompt.split()[:10]) + f"_{run}"
            prompt_dir = output_dir / prompt_slug
            single_images_dir = prompt_dir / "single_images"
            single_images_dir.mkdir(parents=True, exist_ok=True)
            
            images = images_list[i * args.num_images_per_prompt + run][:-1]
            for j, image in enumerate(images):
                image.save(single_images_dir / f"{prompt_slug}_{args.seed}_{j:04d}.png")
            
            save_image_grid_with_borders(images, prompt, args.seed, prompt_dir, num_cols=args.grid_cols)

def main(args: argparse.Namespace) -> None:
    """Main function to run the image prediction generation process."""
    set_seed(args.seed)
    
    print(f"Setting up pipeline...")
    pipeline = setup_pipeline(args.model_id, args.device)
    
    print(f"Generating images...")
    images = generate_images(pipeline, args.prompts, args)
    
    print(f"Saving images...")
    save_images(images, Path(args.output_dir), args)
    
    print(f"Images saved to {args.output_dir}")
    
    # Clear CUDA cache to free up memory
    if args.device == "cuda":
        torch.cuda.empty_cache()

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Generate image predictions for diffusion models")
    parser.add_argument("--model_id", type=str, default="PalionTech/debias-diffusion-orig",
                        help="Hugging Face model ID or path to local model")
    parser.add_argument("--prompts", nargs="+", default=["A photo of a diplomat"],
                        help="List of prompts to generate images from")
    parser.add_argument("--output_dir", type=str, default="results/section_4.2/image_predictions",
                        help="Output directory for generated images")
    parser.add_argument("--num_images_per_prompt", type=int, default=1,
                        help="Number of images to generate per prompt")
    parser.add_argument("--num_inference_steps", type=int, default=50,
                        help="Number of denoising steps")
    parser.add_argument("--guidance_scale", type=float, default=7.5,
                        help="Guidance scale for classifier-free guidance")
    parser.add_argument("--grid_cols", type=int, default=10,
                        help="Number of columns in the output grid")
    parser.add_argument("--seed", type=int, default=9374,
                        help="Random seed for reproducibility")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to run the model on")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args)