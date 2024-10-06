"""
Re-Diffusion Experiment for Diffusion Models

This script implements the re-diffusion experiment described in Section 4.2 of the associated thesis.
It generates images using a diffusion model, then re-diffuses these images to various noise levels
and attempts to recover the original image, demonstrating the stability of the generative process.

Usage:
    python src/sections/section_4.3/re_diffusion.py [--args]

Arguments:
    --model_id: Hugging Face model ID or path to local model (default: "PalionTech/debias-diffusion-orig")
    --prompts: List of prompts to generate images from (default: ["A photo of a diplomat"])
    --output_dir: Output directory for generated images (default: "results/section_4.3/re_diffusion")
    --num_images_per_prompt: Number of images to generate per prompt (default: 1)
    --num_inference_steps: Number of denoising steps (default: 50)
    --guidance_scale: Guidance scale for classifier-free guidance (default: 7.5)
    --grid_cols: Number of columns in the output grid (default: 10)
    --seed: Random seed for reproducibility (default: 9374)
    --device: Device to run the model on (default: "cuda" if available, else "cpu")

Outputs:
    - Individual images for each re-diffusion step and prompt
    - Image grids showing the re-diffusion process for each prompt
"""

import argparse
import random
import sys
from pathlib import Path
from typing import List, Tuple

import torch
from torchvision.transforms import ToTensor
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

def generate_and_rediffuse(pipeline: TestingDiffusionPipeline, prompts: List[str], args: argparse.Namespace) -> List[List[torch.Tensor]]:
    """Generate images and perform re-diffusion experiment."""
    all_images = []
    for prompt in tqdm(prompts, desc="Processing prompts"):
        for _ in tqdm(range(args.num_images_per_prompt), desc="Per prompt"):
            # Generate initial image
            image = pipeline(
                prompt,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                generator=torch.Generator(device=pipeline.device).manual_seed(args.seed)
            )[0][0]
            
            image_tensor = ToTensor()(image).to(pipeline.device, dtype=pipeline.vae.dtype).unsqueeze(0)

            re_diffused_images = [image]
            for t in tqdm(pipeline.scheduler.timesteps, desc="Re-diffusing"):
                t_tensor = torch.tensor([int(t)], dtype=torch.long, device=pipeline.device)
                latents = pipeline.vae.encode(image_tensor).latent_dist.sample().detach()
                latents = latents * pipeline.vae.config.scaling_factor
                noise = torch.randn_like(latents)
                noisy_latents = pipeline.scheduler.add_noise(latents, noise, t_tensor)
                noisy_image = pipeline(
                    prompt,
                    latents=noisy_latents,
                    num_inference_steps=args.num_inference_steps, 
                    guidance_scale=args.guidance_scale, 
                    generator=torch.Generator(device=pipeline.device).manual_seed(args.seed),
                    stop_t=t
                )[0][0]
                re_diffused_images.append(noisy_image)
            all_images.append(re_diffused_images)
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
            
            images = images_list[i * args.num_images_per_prompt + run]
            for j, image in enumerate(images):
                image.save(single_images_dir / f"{prompt_slug}_{args.seed}_{j:04d}.png")
            
            save_image_grid_with_borders(images, prompt, args.seed, prompt_dir, num_cols=args.grid_cols)

def main(args: argparse.Namespace) -> None:
    """Main function to run the re-diffusion experiment."""
    set_seed(args.seed)
    
    print(f"Setting up pipeline...")
    pipeline = setup_pipeline(args.model_id, args.device)
    
    print(f"Generating and re-diffusing images...")
    images = generate_and_rediffuse(pipeline, args.prompts, args)
    
    print(f"Saving images...")
    save_images(images, Path(args.output_dir), args)
    
    print(f"Images saved to {args.output_dir}")
    
    # Clear CUDA cache to free up memory
    if args.device == "cuda":
        torch.cuda.empty_cache()

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Re-diffusion experiment for diffusion models")
    parser.add_argument("--model_id", type=str, default="PalionTech/debias-diffusion-orig",
                        help="Hugging Face model ID or path to local model")
    parser.add_argument("--prompts", nargs="+", default=["A photo of a diplomat"],
                        help="List of prompts to generate images from")
    parser.add_argument("--output_dir", type=str, default="results/section_4.2/re_diffusion",
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