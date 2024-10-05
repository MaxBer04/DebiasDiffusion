import argparse
import json
import random
import os
from pathlib import Path
import sys
import torch
from torchvision.transforms import ToPILImage, ToTensor
from torchvision.utils import make_grid
from accelerate import Accelerator
from diffusers import StableDiffusionPipeline, SemanticStableDiffusionPipeline
from diffusers.loaders import LoraLoaderMixin
from tqdm.auto import tqdm
import numpy as np
from PIL import Image
import csv
import time
import psutil
import GPUtil
import math

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(BASE_DIR))

from pipelines.attribute_switching_pipeline import AttributeSwitchingPipeline
from pipelines.debias_diffusion_pipeline import DebiasDiffusionPipeline
from pipelines.fair_diffusion_pipeline import SemanticStableDiffusionPipeline

def load_prompts(file_path, dataset_type):
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    if dataset_type == 'occupation':
        return data['occupations']
    elif dataset_type == 'laion':
        return data['prompts']
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

def generate_prompts(prompts, num_samples, dataset_type):
    all_prompts = []
    for content in prompts:
        if dataset_type == 'occupation':
            all_prompts.extend([f"A photo of the face of a {content}" for _ in range(num_samples)])
        elif dataset_type == 'laion':
            all_prompts.extend([content for _ in range(num_samples)])
    return all_prompts

def create_grid(images, batch_size):
    num_images = len(images)
    rows = int(math.sqrt(num_images))
    cols = math.ceil(num_images / rows)
    w, h = images[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    for i, image in enumerate(images):
        grid.paste(image, box=(i%cols*w, i//cols*h))
    return grid

def save_grid(grid, output_dir, group, batch_index):
    output_dir.mkdir(parents=True, exist_ok=True)
    grid.save(output_dir / f"{group}_batch_{batch_index}.png")

def save_image(image, output_dir, group, seed, max_filename_length=100):
    # Kürze den Gruppen- (bzw. Prompt-) Namen, falls er zu lang ist
    if len(group) > max_filename_length:
        shortened_group = group[:max_filename_length]
    else:
        shortened_group = group

    # Speichere das Bild im gekürzten Ordnernamen
    group_dir = output_dir / shortened_group
    group_dir.mkdir(parents=True, exist_ok=True)
    image.save(group_dir / f"{seed}.png")
    
    return shortened_group

def save_metadata(metadata, output_dir, dataset_type):
    metadata_file = output_dir / "metadata.csv"
    file_exists = metadata_file.exists()
    
    fieldnames = ["seed", "prompt", "group"] if dataset_type == 'laion' else ["occupation", "seed", "prompt", "group"]
    
    with open(metadata_file, mode='a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerows(metadata)

def save_statistics(stats, output_dir):
    stats_file = output_dir / "performance_stats.csv"
    file_exists = stats_file.exists()
    
    with open(stats_file, mode='a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["group", "avg_time_per_image", "avg_time_per_batch", "avg_gpu_memory_usage"])
        if not file_exists:
            writer.writeheader()
        writer.writerow(stats)

def get_gpu_memory_usage():
    GPUs = GPUtil.getGPUs()
    return GPUs[0].memoryUsed if GPUs else 0  # Returns memory usage of first GPU

def save_progress(output_dir, completed_groups):
    progress_file = output_dir / "progress.json"
    with open(progress_file, 'w') as f:
        json.dump({"completed_groups": list(completed_groups)}, f)

def load_progress(output_dir):
    progress_file = output_dir / "progress.json"
    if progress_file.exists():
        print("Resuming from past progress...")
        with open(progress_file, 'r') as f:
            return json.load(f)
    return {"completed_groups": []}

def main(args):
    global_seed = args.seed
    torch.manual_seed(global_seed)
    np.random.seed(global_seed)

    accelerator = Accelerator()
    device = accelerator.device

    if accelerator.is_main_process:
        print(f"Using {accelerator.num_processes} GPUs")
    
    args.output_dir.mkdir(parents=True, exist_ok=True)

    prompts = load_prompts(args.prompts_file, args.dataset_type)
    
    progress = load_progress(args.output_dir)
    completed_groups = set(progress["completed_groups"])
    prompts = [p for p in prompts if p not in completed_groups]
    
    all_prompts = generate_prompts(prompts, args.num_samples, args.dataset_type)

    dtype = torch.float16 if args.use_fp16 else torch.float32

    # Pipeline setup (unchanged)
    if args.model == "SD":
        pipeline = StableDiffusionPipeline.from_pretrained(args.model_id, torch_dtype=dtype)
        pipeline = pipeline.to(device)
    elif args.model == "FDM":
        pipeline = StableDiffusionPipeline.from_pretrained(args.model_id, torch_dtype=dtype)
        pipeline = pipeline.to(device)
        text_encoder_lora_params = LoraLoaderMixin._modify_text_encoder(pipeline.text_encoder, dtype=torch.float32, rank=args.rank, patch_mlp=False)
        text_encoder_lora_dict = torch.load(args.load_text_encoder_lora_from, map_location=device)
        _ = pipeline.text_encoder.load_state_dict(text_encoder_lora_dict, strict=False)
    elif args.model == "FD":
        pipeline = SemanticStableDiffusionPipeline.from_pretrained(args.model_id, torch_dtype=torch.float16)
        pipeline = pipeline.to(device)
        pipeline.set_momentum(scale=0.3, beta=0.6)

        editing_prompts = {
            "gender": ["male person", "female person"],
            "age": ["young person", "old person"],
            "race": ["white person", "black person", "asian person", "indian person"]
        }
        edit_warmup_steps = {"gender": [10, 10], "age": [5, 5], "race": [5, 5, 5, 5]} #
        edit_guidance_scales = {"gender": [6, 6], "age": [3, 3], "race": [4, 4, 4, 4]} #
        edit_thresholds = {"gender": [0.95, 0.95], "age": [0.95, 0.95], "race": [0.95, 0.95, 0.95, 0.95]} #
        edit_weights = {"gender": [1, 1], "age": [3, 1], "race": [1, 1, 1, 1]} #
        
        pipeline.set_attribute_params(editing_prompts, edit_warmup_steps, edit_guidance_scales, edit_thresholds, edit_weights)
    elif args.model == "DD":
        pipeline = DebiasDiffusionPipeline.from_pretrained(args.model_id, torch_dtype=dtype)
        pipeline = pipeline.to(device)
        
        classifiers_base_path = os.path.join(SCRIPT_DIR, "classifiers_all", "classifiers_qqff", "5k")
        
        pipeline.set_attribute_params(
            attribute="gender",
            distribution=[0.5, 0.5],
            bias_range=(0, 0.5),
            classifier_path=os.path.join(classifiers_base_path, "gender_5k_e100_bs256_lr0.0001_tv0.8", "best_model.pt"),
            num_classes=2,
            model_type="linear",
            default_assignments=None,
            default_switch_step=None,
        )
        
        pipeline.set_attribute_params(
            attribute="race",
            distribution=[0.25, 0.25, 0.25, 0.25],
            bias_range=(0, 0.75),
            classifier_path=os.path.join(classifiers_base_path, "race_5k_e100_bs256_lr0.0001_tv0.8", "best_model.pt"),
            num_classes=4,
            model_type="linear",
            default_assignments=None,
            default_switch_step=None,
        )
                
        pipeline.set_attribute_params(
            attribute="age",
            distribution=[0.75, 0.25],
            bias_range=(2.5, 7),
            classifier_path=os.path.join(classifiers_base_path, "age_5k_e100_bs256_lr0.0001_tv0.8", "best_model.pt"),
            num_classes=2,
            model_type="linear",
            default_assignments=None,
            default_switch_step=None,
        )
        
        pipeline.set_tau_bias(19) # 50 - 19 = 31 in thesis
        pipeline.set_iota_step_range([4,19]) 
        pipeline.set_debiasing_options(use_debiasing=True, use_distribution_guidance=True, interpolation_method='linear')
    elif args.model == "AS":
        pipeline = AttributeSwitchingPipeline.from_pretrained(args.model_id, torch_dtype=dtype)
        pipeline = pipeline.to(device)
        attribute_switch_steps = {"gender": 22, "race": 22, "age": 21} # , 
        attribute_weights = {"gender": [1,1], "race": [1,1,1,1], "age": [3,1]} # , 
        for attr in attribute_switch_steps.keys():
            pipeline.set_attribute_params(attr, attribute_switch_steps[attr], attribute_weights[attr])
        pipeline.set_debiasing_options(True)

    pipeline.safety_checker = None
    pipeline = accelerator.prepare(pipeline)

    if not accelerator.is_main_process:
        pipeline.set_progress_bar_config(disable=True)

    checkpoint_interval = args.checkpoint_interval
    last_checkpoint = time.time()

    # Generate seeds for all images
    all_seeds = [global_seed + i for i in range(len(all_prompts))]

    # Distribute prompts across GPUs
    local_prompts = all_prompts[accelerator.process_index::accelerator.num_processes]
    local_seeds = all_seeds[accelerator.process_index::accelerator.num_processes]

    total_batches = math.ceil(len(local_prompts) / args.batch_size)
    
    group_stats = {}
    
    for batch_index in tqdm(range(0, len(local_prompts), args.batch_size), 
                                total=total_batches, 
                                desc="Generating images", 
                                disable=not accelerator.is_main_process):
            batch_prompts = local_prompts[batch_index:batch_index + args.batch_size]
            batch_seeds = local_seeds[batch_index:batch_index + args.batch_size]

            generators = [torch.Generator(device=device).manual_seed(seed) for seed in batch_seeds]

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
            metadata = []
            for i, (image, seed, prompt) in enumerate(zip(batch_images, batch_seeds, batch_prompts)):
                group = prompt.split()[-1] if args.dataset_type == 'occupation' else prompt
                shortened_group = save_image(image, args.output_dir, group, seed) 
                
                if args.dataset_type == 'occupation':
                    metadata.append({"occupation": shortened_group, "seed": seed, "prompt": prompt, "group": shortened_group})
                else:
                    metadata.append({"seed": seed, "prompt": prompt, "group": shortened_group})
                
                # Update group stats
                if group not in group_stats:
                    group_stats[group] = {"total_time": 0, "total_images": 0, "total_memory": 0, "total_batches": 0}
                group_stats[group]["total_time"] += batch_time / len(batch_images)
                group_stats[group]["total_images"] += 1
                group_stats[group]["total_memory"] += gpu_memory
                group_stats[group]["total_batches"] += 1 / len(batch_images)

            # Save metadata for this batch
            save_metadata(metadata, args.output_dir, args.dataset_type)

            if accelerator.is_main_process:
                # Create and save grid for this batch
                grid = create_grid(batch_images, args.batch_size)
                save_grid(grid, args.output_dir, f"mixed_groups", batch_index // args.batch_size)

            torch.cuda.empty_cache()
            completed_groups.update(set(batch_prompts))

            if time.time() - last_checkpoint > checkpoint_interval:
                if accelerator.is_main_process:
                    print(f"Checkpointing at batch {batch_index}")
                    save_progress(args.output_dir, completed_groups)
                last_checkpoint = time.time()

    # Save per-group statistics
    for group, stats in group_stats.items():
        avg_time_per_image = stats["total_time"] / stats["total_images"]
        avg_time_per_batch = stats["total_time"] / stats["total_batches"]
        avg_gpu_memory = stats["total_memory"] / stats["total_batches"]
        
        group_stats = {
            "group": group,
            "avg_time_per_image": avg_time_per_image,
            "avg_time_per_batch": avg_time_per_batch,
            "avg_gpu_memory_usage": avg_gpu_memory
        }
        
        if accelerator.is_main_process:
            save_statistics(group_stats, args.output_dir)

    if accelerator.is_main_process:
        print("Image generation, saving, and statistics collection completed")
        
def parse_args():
    parser = argparse.ArgumentParser(description="Generate images using Stable Diffusion")
    parser.add_argument("--model_id", type=str, default="PalionTech/debias-diffusion-orig", help="Hugging Face model ID")
    parser.add_argument("--prompts_file", type=str, default=BASE_DIR / "data" / "prompt_lists" / "5.4.1_occupations_500.json", help="Path to prompts JSON file")
    parser.add_argument("--output_dir", type=str, default=BASE_DIR / "outputs" / "section_5.4" / "generations" / "rag" / "FDM", help="Output directory for generated images")
    parser.add_argument("--num_samples", type=int, default=128, help="Number of samples per prompt/occupation")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for image generation")
    parser.add_argument("--use_fp16", type=bool, default=True, help="Use half precision floating point for models")
    parser.add_argument("--seed", type=int, default=51904, help="Global seed for reproducibility")
    parser.add_argument("--rank", type=int, default=50, help="The dimension of the LoRA update matrices.")
    parser.add_argument("--load_text_encoder_lora_from", type=str, default=BASE_DIR / "data" / "FDM" / "text_encoder_lora_EMA_rag.pth")
    parser.add_argument("--model", type=str, default="FDM", choices=["SD", "FDM", "FD", "DD", "AS"], help="Choose the pipeline to use")
    parser.add_argument("--checkpoint_interval", type=int, default=300, help="Interval in seconds for checkpointing")
    parser.add_argument("--dataset_type", type=str, choices=['occupation', 'laion'], default='occupation', help="Type of dataset: occupation or laion")
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args)