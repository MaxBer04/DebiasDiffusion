import argparse
import random
import sys
from pathlib import Path
from typing import List, Dict

import torch
from diffusers import StableDiffusionPipeline
from transformers import CLIPTextModel, CLIPTokenizer
from PIL import Image
from tqdm.auto import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from pipelines.attribute_switching_pipeline import AttributeSwitchingPipeline
from pipelines.debias_diffusion_pipeline import DebiasDiffusionPipeline
from pipelines.fair_diffusion_pipeline import SemanticStableDiffusionPipeline

def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def setup_pipeline(model: str, model_id: str, device: str) -> StableDiffusionPipeline:
    if model == "SD":
        pipeline = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    elif model == "FTF":
        pipeline = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
        text_encoder_lora_params = CLIPTextModel.from_pretrained(
            BASE_DIR / "data/FDM/text_encoder_lora_EMA_rag.pth",
            torch_dtype=torch.float16
        )
        pipeline.text_encoder.load_state_dict(text_encoder_lora_params.state_dict(), strict=False)
    elif model == "DD":
        pipeline = DebiasDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to(device)
        classifiers_base_path = BASE_DIR / "data" / "DD" / "classifiers_all" / "classifiers_qqff" / "5k"
        pipeline.set_attribute_params(
            attribute="gender",
            distribution=[0.5, 0.5],
            bias_range=(0, .5),
            classifier_path=classifiers_base_path / "gender_5k_e100_bs256_lr0.0001_tv0.8" / "best_model.pt",
            num_classes=2,
            model_type="linear",
            default_assignments=None,
            default_switch_step=None,
        )
        pipeline.set_attribute_params(
            attribute="race",
            distribution=[0.25, 0.25, 0.25, 0.25],
            bias_range=(0, .75),
            classifier_path=classifiers_base_path / "race_5k_e100_bs256_lr0.0001_tv0.8" / "best_model.pt",
            num_classes=4,
            model_type="linear",
            default_assignments=None,
            default_switch_step=None,
        )
        pipeline.set_attribute_params(
            attribute="age",
            distribution=[0.75, 0.25],
            bias_range=(0, 1.125),
            classifier_path=classifiers_base_path / "age_5k_e100_bs256_lr0.0001_tv0.8" / "best_model.pt",
            num_classes=2,
            model_type="linear",
            default_assignments=None,
            default_switch_step=None,
        )
        pipeline.set_tau_bias(19)
        pipeline.set_iota_step_range([4,19])
        pipeline.set_debiasing_options(use_debiasing=True, use_distribution_guidance=True, interpolation_method='linear')
    elif model == "AS":
        pipeline = AttributeSwitchingPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
        attribute_switch_steps = {"gender": 18, "race": 19, "age": 18}
        attribute_weights = {"gender": [1,1], "race": [1,1,1,1], "age": [3,1]}
        for attr in attribute_switch_steps.keys():
            pipeline.set_attribute_params(attr, attribute_switch_steps[attr], attribute_weights[attr])
        pipeline.set_debiasing_options(True)
    elif model == "FD":
        pipeline = SemanticStableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
        pipeline.set_momentum(scale=0.3, beta=0.6)
        editing_prompts = {
            "gender": ["male person", "female person"],
            "age": ["young person", "old person"],
            "race": ["white person", "black person", "asian person", "indian person"]
        }
        edit_warmup_steps = {"gender": [10, 10], "age": [5, 5], "race": [5, 5, 5, 5]}
        edit_guidance_scales = {"gender": [6, 6], "age": [3, 3], "race": [4, 4, 4, 4]}
        edit_thresholds = {"gender": [0.95, 0.95], "age": [0.95, 0.95], "race": [0.95, 0.95, 0.95, 0.95]}
        edit_weights = {"gender": [1, 1], "age": [3, 1], "race": [1, 1, 1, 1]}
        pipeline.set_attribute_params(editing_prompts, edit_warmup_steps, edit_guidance_scales, edit_thresholds, edit_weights)
    else:
        raise ValueError(f"Unknown model type: {model}")
    
    return pipeline.to(device)

def generate_images(pipeline, prompts: List[str], args: argparse.Namespace) -> List[Image.Image]:
    """Generate images using the provided pipeline and prompts."""
    all_images = []
    for prompt in tqdm(prompts, desc="Generating images"):
        if args.model in ["SD", "FTF", "DD", "AS"]:
            batch_prompts = [prompt] * args.num_images_per_prompt
            images = pipeline(batch_prompts, num_inference_steps=50, guidance_scale=7.5, generator=torch.Generator(device=pipeline.device).manual_seed(args.seed)).images
        elif args.model == "FD":
            num_images = args.num_images_per_prompt
            attributes = pipeline.editing_prompts.keys()
            choices = [0 for _ in attributes]
            reverse_editing_direction = [i == choice for attr in attributes for i, choice in 
                    zip(range(2 if attr != 'race' else 4), [choices.pop(0)] * (2 if attr != 'race' else 4))]
            images = pipeline(
                [prompt] * num_images,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                generator=torch.Generator(device=args.device).manual_seed(args.seed),
                reverse_editing_direction=reverse_editing_direction
            ).images
        else:
            raise ValueError(f"Unknown model type: {args.model}")
        
        all_images.extend(images)
    return all_images

def create_image_grid(images: List[Image.Image], rows: int, cols: int) -> Image.Image:
    w, h = images[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    for i, image in enumerate(images):
        grid.paste(image, box=(i%cols*w, i//cols*h))
    return grid

def save_images(images: List[Image.Image], output_dir: Path, model: str, args: argparse.Namespace):
    model_output_dir = output_dir / model
    model_output_dir.mkdir(parents=True, exist_ok=True)
    
    for prompt in args.prompts:
        prompt_slug = "_".join(prompt.split()[:10])  # Use first 10 words of prompt for filename
        prompt_dir = model_output_dir / prompt_slug
        prompt_dir.mkdir(parents=True, exist_ok=True)
        
        prompt_images = images[:args.num_images_per_prompt]
        images = images[args.num_images_per_prompt:]  # Remove processed images
        
        for i, image in enumerate(prompt_images):
            image.save(prompt_dir / f"{prompt_slug}_{args.seed}_{i:04d}.png")
        
        rows = (len(prompt_images) + args.grid_cols - 1) // args.grid_cols
        grid = create_image_grid(prompt_images, rows, args.grid_cols)
        grid.save(model_output_dir / f"{prompt_slug}_{args.seed}_grid.png")

def main(args: argparse.Namespace):
    set_seed(args.seed)
    
    for model in args.models:
        print(f"Setting up {model} pipeline...")
        pipeline = setup_pipeline(model, args.model_id, args.device)
        
        print(f"Generating images for {model}...")
        args.model = model  # Set the current model type
        images = generate_images(pipeline, args.prompts, args)
        
        print(f"Saving images for {model}...")
        save_images(images, Path(args.output_dir), model, args)
        
        print(f"Images for {model} saved to {args.output_dir}/{model}")
        
        # Clear CUDA cache to free up memory
        if args.device == "cuda":
            torch.cuda.empty_cache()

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate images using various Stable Diffusion pipelines")
    parser.add_argument("--models", nargs="+", choices=["SD", "FTF", "DD", "AS", "FD"], required=True,
                        help="Types of models to use")
    parser.add_argument("--model_id", type=str, default="PalionTech/debias-diffusion-orig",
                        help="Hugging Face model ID or path to local model")
    parser.add_argument("--prompts", nargs="+", default=["a photo of the face of a senator"],
                        help="List of prompts to generate images from")
    parser.add_argument("--output_dir", type=str, default="outputs",
                        help="Output directory for generated images")
    parser.add_argument("--num_images_per_prompt", type=int, default=64,
                        help="Number of images to generate per prompt")
    parser.add_argument("--num_inference_steps", type=int, default=50,
                        help="Number of denoising steps")
    parser.add_argument("--guidance_scale", type=float, default=7.5,
                        help="Guidance scale for classifier-free guidance")
    parser.add_argument("--grid_cols", type=int, default=8,
                        help="Number of columns in the output grid")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to run the model on")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args)