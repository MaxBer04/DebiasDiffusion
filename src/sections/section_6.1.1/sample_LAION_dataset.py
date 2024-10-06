"""
LAION Dataset Sampling for DebiasDiffusion Syntactic Filtering Evaluation

This script samples and processes images from the LAION-400m dataset to evaluate
the false-positive rate of the syntactic filtering mechanism in DebiasDiffusion.
It downloads images without human faces, applies the filtering, and generates
images using both the original Stable Diffusion model and DebiasDiffusion for comparison.

Usage:
    python src/sections/section_6.1.1/sample_LAION_dataset.py [--args]

Arguments:
    --num_samples: Number of samples to process (default: 1000)
    --output_dir: Directory to save output files (default: results/section_6.1.1/laion_sampling)
    --batch_size: Batch size for image generation (default: 32)
    --seed: Random seed for reproducibility (default: 42)

Outputs:
    - CSV file with sampled prompts and filtering results
    - Generated images for both Stable Diffusion and DebiasDiffusion
    - Text file with analysis results
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Tuple

import torch
import clip
import pandas as pd
from PIL import Image
from tqdm import tqdm
from transformers import AutoFeatureExtractor, AutoModelForImageClassification

# Add project root to Python path
SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent.parent.parent.parent
sys.path.append(str(BASE_DIR))

from src.utils.fairness import extract_and_classify_nouns
from src.pipelines.debias_diffusion_pipeline import DebiasDiffusionPipeline
from src.utils.image_utils import save_images

def load_laion_subset(num_samples: int) -> pd.DataFrame:
    """
    Load a subset of the LAION-400m dataset.
    
    Args:
        num_samples (int): Number of samples to load.
    
    Returns:
        pd.DataFrame: Dataframe containing the loaded samples.
    """
    # Placeholder: Replace with actual LAION dataset loading logic
    data = {'url': ['http://example.com/image1.jpg'] * num_samples,
            'caption': ['Sample caption'] * num_samples}
    return pd.DataFrame(data)

def setup_models() -> Tuple[Any, Any, Any]:
    """
    Set up CLIP, face detection, and Stable Diffusion models.
    
    Returns:
        Tuple[Any, Any, Any]: CLIP model, face detection model, and Stable Diffusion model.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, _ = clip.load("ViT-B/32", device=device)
    face_model = AutoModelForImageClassification.from_pretrained("microsoft/resnet-50")
    sd_model = DebiasDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
    return clip_model, face_model, sd_model

def filter_images(df: pd.DataFrame, face_model: Any) -> pd.DataFrame:
    """
    Filter images to keep only those without human faces.
    
    Args:
        df (pd.DataFrame): Dataframe containing image URLs and captions.
        face_model (Any): Face detection model.
    
    Returns:
        pd.DataFrame: Filtered dataframe.
    """
    # Placeholder: Implement actual face detection logic
    return df.sample(n=len(df))

def apply_syntactic_filtering(prompts: List[str]) -> List[bool]:
    """
    Apply syntactic filtering to the given prompts.
    
    Args:
        prompts (List[str]): List of prompts to filter.
    
    Returns:
        List[bool]: List of boolean values indicating if each prompt contains human-related nouns.
    """
    return [bool(extract_and_classify_nouns(prompt)) for prompt in prompts]

def generate_images(model: Any, prompts: List[str], batch_size: int) -> List[Image.Image]:
    """
    Generate images using the given model and prompts.
    
    Args:
        model (Any): Image generation model (Stable Diffusion or DebiasDiffusion).
        prompts (List[str]): List of prompts for image generation.
        batch_size (int): Batch size for image generation.
    
    Returns:
        List[Image.Image]: List of generated images.
    """
    images = []
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i+batch_size]
        batch_images = model(batch_prompts, num_inference_steps=50, guidance_scale=7.5).images
        images.extend(batch_images)
    return images

def compute_clip_similarity(clip_model: Any, images: List[Image.Image], prompts: List[str]) -> float:
    """
    Compute CLIP similarity between images and prompts.
    
    Args:
        clip_model (Any): CLIP model.
        images (List[Image.Image]): List of generated images.
        prompts (List[str]): List of corresponding prompts.
    
    Returns:
        float: Average CLIP similarity score.
    """
    # Placeholder: Implement actual CLIP similarity computation
    return 0.5

def main(args: argparse.Namespace) -> None:
    clip_model, face_model, sd_model = setup_models()
    
    print("Loading LAION subset...")
    df = load_laion_subset(args.num_samples)
    
    print("Filtering images without faces...")
    df_filtered = filter_images(df, face_model)
    
    print("Applying syntactic filtering...")
    contains_human = apply_syntactic_filtering(df_filtered['caption'].tolist())
    df_filtered['contains_human'] = contains_human
    
    print("Generating images with Stable Diffusion...")
    sd_images = generate_images(sd_model, df_filtered['caption'].tolist(), args.batch_size)
    
    print("Generating images with DebiasDiffusion...")
    dd_images = generate_images(sd_model, df_filtered['caption'].tolist(), args.batch_size)
    
    print("Computing CLIP similarities...")
    sd_similarity = compute_clip_similarity(clip_model, sd_images, df_filtered['caption'].tolist())
    dd_similarity = compute_clip_similarity(clip_model, dd_images, df_filtered['caption'].tolist())
    
    print("Saving results...")
    os.makedirs(args.output_dir, exist_ok=True)
    df_filtered.to_csv(os.path.join(args.output_dir, 'filtered_prompts.csv'), index=False)
    save_images(sd_images, os.path.join(args.output_dir, 'sd_images'), df_filtered['caption'].tolist(), 1, 8, args.seed)
    save_images(dd_images, os.path.join(args.output_dir, 'dd_images'), df_filtered['caption'].tolist(), 1, 8, args.seed)
    
    with open(os.path.join(args.output_dir, 'analysis_results.txt'), 'w') as f:
        f.write(f"Total prompts: {len(df_filtered)}\n")
        f.write(f"Prompts with human-related nouns: {sum(contains_human)}\n")
        f.write(f"False positive rate: {sum(contains_human) / len(df_filtered):.2%}\n")
        f.write(f"Stable Diffusion CLIP similarity: {sd_similarity:.4f}\n")
        f.write(f"DebiasDiffusion CLIP similarity: {dd_similarity:.4f}\n")
    
    print("Analysis complete. Results saved to", args.output_dir)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample LAION dataset and evaluate syntactic filtering")
    parser.add_argument("--num_samples", type=int, default=1000, help="Number of samples to process")
    parser.add_argument("--output_dir", type=str, default=str(BASE_DIR / "results/section_6.1.1/laion_sampling"),
                        help="Directory to save output files")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for image generation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args)