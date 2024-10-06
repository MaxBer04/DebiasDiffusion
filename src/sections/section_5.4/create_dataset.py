"""
Dataset Creation for Debiasing Methods Evaluation

This script generates datasets for evaluating various debiasing methods in text-to-image diffusion models.
It supports distributed generation across multiple GPUs and batch processing for efficiency.

Usage:
    python src/sections/section_5.4/create_dataset.py [--args]

Arguments:
    --model_id: Hugging Face model ID or path to local model (default: "PalionTech/debias-diffusion-orig")
    --prompts_file: Path to JSON file containing prompts (default: "data/prompt_lists/5.4.1_occupations_500.json")
    --output_dir: Output directory for generated images (default: "outputs/section_5.4/generations/rag/FDM")
    --num_samples: Number of samples per prompt/occupation (default: 128)
    --batch_size: Batch size for image generation (default: 64)
    --use_fp16: Use half precision floating point for models (default: True)
    --seed: Global seed for reproducibility (default: 51904)
    --rank: Dimension of the LoRA update matrices (default: 50)
    --load_text_encoder_lora_from: Path to LoRA weights (default: "data/FDM/text_encoder_lora_EMA_rag.pth")
    --model: Pipeline to use (choices: SD, FDM, FD, DD, AS, default: FDM)
    --checkpoint_interval: Interval in seconds for checkpointing (default: 300)
    --dataset_type: Type of dataset: occupation or laion (default: occupation)

Outputs:
    - Generated images for each prompt and method
    - Metadata CSV file with generation details
    - Performance statistics CSV file
    - Checkpoints for resuming interrupted generation

Note: This script requires significant computational resources and is designed to run on multiple GPUs.
"""

import os
import json
import torch
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
from tqdm import tqdm
import time
import sys

from accelerate import Accelerator
from accelerate.utils import set_seed
from diffusers import StableDiffusionPipeline
from diffusers.loaders import LoraLoaderMixin

# Add project root to Python path
SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent.parent.parent
sys.path.append(str(BASE_DIR))

from src.pipelines.attribute_switching_pipeline import AttributeSwitchingPipeline
from src.pipelines.debias_diffusion_pipeline import DebiasDiffusionPipeline
from src.pipelines.fair_diffusion_pipeline import SemanticStableDiffusionPipeline
from src.utils.general import ensure_directory, get_gpu_memory_usage

def load_prompts(file_path: Path, dataset_type: str) -> List[str]:
    """Load prompts from a JSON file."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data['occupations'] if dataset_type == 'occupation' else data['prompts']

def setup_pipeline(args: argparse.Namespace, device: torch.device) -> StableDiffusionPipeline:
    """Set up the appropriate pipeline based on the specified model."""
    dtype = torch.float16 if args.use_fp16 else torch.float32

    if args.model == "SD":
        pipeline = StableDiffusionPipeline.from_pretrained(args.model_id, torch_dtype=dtype)
    elif args.model == "FDM":
        pipeline = StableDiffusionPipeline.from_pretrained(args.model_id, torch_dtype=dtype)
        text_encoder_lora_params = LoraLoaderMixin._modify_text_encoder(pipeline.text_encoder, dtype=torch.float32, rank=args.rank, patch_mlp=False)
        text_encoder_lora_dict = torch.load(args.load_text_encoder_lora_from, map_location=device)
        _ = pipeline.text_encoder.load_state_dict(text_encoder_lora_dict, strict=False)
    elif args.model == "FD":
        pipeline = SemanticStableDiffusionPipeline.from_pretrained(args.model_id, torch_dtype=torch.float16)
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
    elif args.model == "DD":
        pipeline = DebiasDiffusionPipeline.from_pretrained(args.model_id, torch_dtype=dtype)
        classifiers_base_path = BASE_DIR / "data/model_data/h_space_classifiers/version_2/5k"
        for attr, params in {
            "gender": ([0.5, 0.5], (0, 0.5), 2),
            "race": ([0.25, 0.25, 0.25, 0.25], (0, 0.75), 4),
            "age": ([0.75, 0.25], (0, 1.125), 2)
        }.items():
            pipeline.set_attribute_params(
                attribute=attr,
                distribution=params[0],
                bias_range=params[1],
                classifier_path=classifiers_base_path / f"{attr}_5k_e100_bs256_lr0.0001_tv0.8/best_model.pt",
                num_classes=params[2],
                model_type="linear",
                default_assignments=None,
                default_switch_step=None,
            )
        pipeline.set_tau_bias(19)
        pipeline.set_iota_step_range([4, 19])
        pipeline.set_debiasing_options(use_debiasing=True, use_distribution_guidance=True, interpolation_method='linear')
    elif args.model == "AS":
        pipeline = AttributeSwitchingPipeline.from_pretrained(args.model_id, torch_dtype=dtype)
        attribute_switch_steps = {"gender": 22, "race": 22, "age": 21}
        attribute_weights = {"gender": [1,1], "race": [1,1,1,1], "age": [3,1]}
        for attr in attribute_switch_steps.keys():
            pipeline.set_attribute_params(attr, attribute_switch_steps[attr], attribute_weights[attr])
        pipeline.set_debiasing_options(True)
    else:
        raise ValueError(f"Unknown model type: {args.model}")

    pipeline = pipeline.to(device)
    pipeline.safety_checker = None
    return pipeline

def generate_and_save_images(pipeline: StableDiffusionPipeline, prompts: List[str], args: argparse.Namespace, accelerator: Accelerator) -> Dict[str, Any]:
    """Generate images and save them along with metadata."""
    total_batches = (len(prompts) * args.num_samples + args.batch_size - 1) // args.batch_size
    group_stats: Dict[str, Dict[str, float]] = {}
    results: List[Dict[str, Any]] = []

    for batch_index in tqdm(range(0, len(prompts) * args.num_samples, args.batch_size), 
                            total=total_batches, 
                            desc="Generating images", 
                            disable=not accelerator.is_main_process):
        batch_prompts = [prompts[i // args.num_samples] for i in range(batch_index, min(batch_index + args.batch_size, len(prompts) * args.num_samples))]
        batch_seeds = [args.seed + i for i in range(batch_index, min(batch_index + args.batch_size, len(prompts) * args.num_samples))]

        generators = [torch.Generator(device=accelerator.device).manual_seed(seed) for seed in batch_seeds]

        batch_start_time = time.time()
        with torch.no_grad():
            if args.model == "FD":
                attributes = pipeline.editing_prompts.keys()
                choices = [0 for _ in attributes]
                reverse_editing_direction = [i == choice for attr in attributes for i, choice in 
                        zip(range(2 if attr != 'race' else 4), [choices.pop(0)] * (2 if attr != 'race' else 4))]
                batch_images = pipeline(batch_prompts, num_inference_steps=50, guidance_scale=7.5, generator=generators, reverse_editing_direction=reverse_editing_direction, return_dict=False)[0]
            else:
                batch_images = pipeline(batch_prompts, num_inference_steps=50, guidance_scale=7.5, generator=generators, return_dict=False)[0]
        batch_end_time = time.time()
        batch_time = batch_end_time - batch_start_time

        gpu_memory = get_gpu_memory_usage()

        accelerator.wait_for_everyone()

        # Save individual images and metadata
        for i, (image, seed, prompt) in enumerate(zip(batch_images, batch_seeds, batch_prompts)):
            group = prompt.split()[-1] if args.dataset_type == 'occupation' else prompt
            image_path = args.output_dir / group / f"{seed}.png"
            ensure_directory(image_path.parent)
            image.save(image_path)
            
            result = {"group": group, "seed": seed, "prompt": prompt}
            if args.dataset_type == 'occupation':
                result["occupation"] = group
            results.append(result)
            
            # Update group stats
            if group not in group_stats:
                group_stats[group] = {"total_time": 0, "total_images": 0, "total_memory": 0, "total_batches": 0}
            group_stats[group]["total_time"] += batch_time / len(batch_images)
            group_stats[group]["total_images"] += 1
            group_stats[group]["total_memory"] += gpu_memory
            group_stats[group]["total_batches"] += 1 / len(batch_images)

        torch.cuda.empty_cache()

        if time.time() - args.last_checkpoint > args.checkpoint_interval:
            if accelerator.is_main_process:
                save_progress(args.output_dir, set(batch_prompts))
            args.last_checkpoint = time.time()

    return {"results": results, "group_stats": group_stats}

def save_metadata(results: List[Dict[str, Any]], output_dir: Path, dataset_type: str) -> None:
    """Save metadata to a CSV file."""
    metadata_file = output_dir / "metadata.csv"
    pd.DataFrame(results).to_csv(metadata_file, index=False)

def save_statistics(group_stats: Dict[str, Dict[str, float]], output_dir: Path) -> None:
    """Save performance statistics to a CSV file."""
    stats = []
    for group, data in group_stats.items():
        stats.append({
            "group": group,
            "avg_time_per_image": data["total_time"] / data["total_images"],
            "avg_time_per_batch": data["total_time"] / data["total_batches"],
            "avg_gpu_memory_usage": data["total_memory"] / data["total_batches"]
        })
    pd.DataFrame(stats).to_csv(output_dir / "performance_stats.csv", index=False)

def save_progress(output_dir: Path, completed_groups: set) -> None:
    """Save progress to a JSON file for potential resumption."""
    with open(output_dir / "progress.json", 'w') as f:
        json.dump({"completed_groups": list(completed_groups)}, f)

def load_progress(output_dir: Path) -> Dict[str, List[str]]:
    """Load progress from a JSON file if it exists."""
    progress_file = output_dir / "progress.json"
    if progress_file.exists():
        with open(progress_file, 'r') as f:
            return json.load(f)
    return {"completed_groups": []}

def main(args: argparse.Namespace) -> None:
    accelerator = Accelerator()
    set_seed(args.seed)

    if accelerator.is_main_process:
        print(f"Using {accelerator.num_processes} GPUs")
    
    args.output_dir = Path(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    prompts = load_prompts(args.prompts_file, args.dataset_type)
    
    progress = load_progress(args.output_dir)
    completed_groups = set(progress["completed_groups"])
    prompts = [p for p in prompts if p not in completed_groups]

    pipeline = setup_pipeline(args, accelerator.device)
    pipeline = accelerator.prepare(pipeline)

    if not accelerator.is_main_process:
        pipeline.set_progress_bar_config(disable=True)

    args.last_checkpoint = time.time()

    data = generate_and_save_images(pipeline, prompts, args, accelerator)

    if accelerator.is_main_process:
        save_metadata(data["results"], args.output_dir, args.dataset_type)
        save_statistics(data["group_stats"], args.output_dir)
        print("Image generation, saving, and statistics collection completed")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate images using Stable Diffusion")
    parser.add_argument("--model_id", type=str, default="PalionTech/debias-diffusion-orig", help="Hugging Face model ID")
    parser.add_argument("--prompts_file", type=Path, default=BASE_DIR / "data/experiments/section_5.4.1/5.4.1_occupations_500.json", help="Path to prompts JSON file")
    parser.add_argument("--output_dir", type=Path, default=BASE_DIR / "results/section_5.4.1/generations/rag/DD", help="Output directory for generated images")
    parser.add_argument("--num_samples", type=int, default=128, help="Number of samples per prompt/occupation")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for image generation")
    parser.add_argument("--use_fp16", type=bool, default=True, help="Use half precision floating point for models")
    parser.add_argument("--seed", type=int, default=51904, help="Global seed for reproducibility")
    parser.add_argument("--rank", type=int, default=50, help="The dimension of the LoRA update matrices.")
    parser.add_argument("--load_text_encoder_lora_from", type=Path, default=BASE_DIR / "data/FDM/text_encoder_lora_EMA_rag.pth")
    parser.add_argument("--model", type=str, default="DD", choices=["SD", "FDM", "FD", "DD", "AS"], help="Choose the pipeline to use")
    parser.add_argument("--checkpoint_interval", type=int, default=300, help="Interval in seconds for checkpointing")
    parser.add_argument("--dataset_type", type=str, choices=['occupation', 'laion'], default='occupation', 
                        help="Type of dataset: occupation or laion")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args)