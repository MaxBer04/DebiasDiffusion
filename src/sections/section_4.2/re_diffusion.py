import argparse
import random
import sys
from pathlib import Path
from typing import List, Dict

import torch
from diffusers import StableDiffusionPipeline
from diffusers.loaders import LoraLoaderMixin
from transformers import CLIPTextModel, CLIPTokenizer
from PIL import Image
from torchvision.transforms import ToTensor, ToPILImage
from tqdm.auto import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(BASE_DIR))

from pipelines.testing_diffusion_pipeline import TestingDiffusionPipeline
from utils.plotting_utils import save_image_grid_with_borders

def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def setup_pipeline(model_id: str, device: str) -> StableDiffusionPipeline:
    pipeline = TestingDiffusionPipeline.from_pretrained(model_id).to(device)
    return pipeline

def generate_images(pipeline, prompts: List[str], args: argparse.Namespace) -> List[Image.Image]:
    """Generate images using the provided pipeline and prompts."""
    all_images = []
    for prompt in tqdm(prompts, desc="Generating images"):
        for run in tqdm(range(args.num_images_per_prompt), desc="Per prompt"):
            image = pipeline(
                prompt,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                generator=torch.Generator(device=pipeline.device).manual_seed(args.seed)
            )[0][0]
            
            image = ToTensor()(image).to(pipeline.device, dtype=pipeline.vae.dtype).unsqueeze(0)

            re_diffused_images = []
            print("Iterating over all timesteps to create re-diffused images...")
            for t in pipeline.scheduler.timesteps:
                t = torch.tensor([int(t)], dtype=torch.long, device=pipeline.device)
                latents = pipeline.vae.encode(image).latent_dist.sample().detach()
                latents = latents * pipeline.vae.config.scaling_factor
                noise = torch.randn_like(latents)
                noisy_latents = pipeline.scheduler.add_noise(latents, noise, t)
                noisy_image = pipeline(prompt,
                                latents=noisy_latents,
                                num_inference_steps=args.num_inference_steps, 
                                guidance_scale=args.guidance_scale, 
                                generator=torch.Generator(device=pipeline.device).manual_seed(args.seed),
                                stop_t=t)[0][0]
                re_diffused_images.append(noisy_image)
            all_images.append(re_diffused_images)
    return all_images

def create_image_grid(images: List[Image.Image], rows: int, cols: int) -> Image.Image:
    w, h = images[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    for i, image in enumerate(images):
        grid.paste(image, box=(i%cols*w, i//cols*h))
    return grid

def save_images(images_list: List[List[Image.Image]], output_dir: Path, args: argparse.Namespace):
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for prompt in args.prompts:
        for run in range(args.num_images_per_prompt):
            prompt_slug = "_".join(prompt.split()[:10]) + f"_{run}"  # Use first 10 words of prompt for filename
            prompt_dir = output_dir / prompt_slug
            single_images_dir = prompt_dir / "single_images"
            single_images_dir.mkdir(parents=True, exist_ok=True)
            
            for _, images in enumerate(images_list):
                images = images[:len(images)-1]
                for i, image in enumerate(images):
                    image.save(single_images_dir / f"{prompt_slug}_{args.seed}_{i:04d}.png")
                
                save_image_grid_with_borders(images, prompt, args.seed, prompt_dir, num_cols=args.grid_cols)

def main(args: argparse.Namespace):
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
    parser = argparse.ArgumentParser(description="Generate images using various Stable Diffusion pipelines")
    parser.add_argument("--model_id", type=str, default="PalionTech/debias-diffusion-orig",
                        help="Hugging Face model ID or path to local model")
    parser.add_argument("--prompts", nargs="+", default=["A photo of a diplomat"],
                        help="List of prompts to generate images from")
    parser.add_argument("--output_dir", type=str, default="outputs/section_4.2/re_diffusion",
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