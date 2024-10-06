"""
Correlation Data Generation for DebiasDiffusion

This script generates data to analyze correlations between estimated and real attribute probabilities
in the DebiasDiffusion pipeline. It uses a list of occupations from a JSON file to create prompts,
processes them through the pipeline, and collects probabilities from h-space classifiers for further analysis.

Usage:
    python src/sections/section_6.1.2/correlation_data_generation.py [--args]

Arguments:
    --model_id: HuggingFace model ID or path to local model (default: "PalionTech/debias-diffusion-orig")
    --num_images: Number of images to generate per occupation (default: 64)
    --occupations_file: Path to JSON file containing occupations list (default: BASE_DIR / "data/experiments/section_6.1.2/occupations.json")
    --output_dir: Directory to save output files (default: BASE_DIR / "results/section_6.1.2/correlation_data")
    --seed: Random seed for reproducibility (default: 51904)
    --use_legacy_gender: Use the legacy FairFace gender classifier (default: True)
    --supported_attributes: List of attributes to evaluate (default: ['gender', 'race'])

Outputs:
    - JSON file with correlation data for each attribute and occupation
    - Console output with generation progress and statistics
"""

import torch
import os
import sys
import random
import numpy as np
import argparse
import json
from tqdm import tqdm
from pathlib import Path
from typing import Dict, List, Any

# Add project root to Python path
SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent.parent.parent
sys.path.append(str(BASE_DIR))

from src.pipelines.debias_diffusion_pipeline import DebiasDiffusionPipeline
from src.utils.face_detection import get_face_detector
from src.utils.attribute_classification import get_attribute_classifier, classify_attribute

def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_pipeline(model_id: str, device: torch.device) -> DebiasDiffusionPipeline:
    """Load and prepare the DebiasDiffusion pipeline."""
    pipe = DebiasDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to(device)
    classifiers_base_path = BASE_DIR / "data/model_data/h_space_classifiers/version_2/5k"
    for attr, params in {
        "gender": ([0.5, 0.5], (0, 0.5), 2),
        "race": ([0.25, 0.25, 0.25, 0.25], (0, 0.75), 4),
        "age": ([0.75, 0.25], (0, 1.125), 2)
    }.items():
        pipe.set_attribute_params(
            attribute=attr,
            distribution=params[0],
            bias_range=params[1],
            classifier_path=classifiers_base_path / f"{attr}_5k_e100_bs256_lr0.0001_tv0.8/best_model.pt",
            num_classes=params[2],
            model_type="linear",
            default_assignments=None,
            default_switch_step=None,
        )
    pipe.set_tau_bias(19)
    pipe.set_iota_step_range([4, 19])
    pipe.set_debiasing_options(use_debiasing=False, use_distribution_guidance=False, interpolation_method='linear')
    pipe.collect_probs = True
    return pipe

def setup_attribute_classifiers(args: argparse.Namespace, device: torch.device) -> Dict[str, Any]:
    """Set up attribute classifiers for evaluation."""
    return {
        attr: get_attribute_classifier(attr, device, use_legacy_gender=args.use_legacy_gender if attr == 'gender' else False)
        for attr in args.supported_attributes
    }

def load_occupations(file_path: Path) -> List[str]:
    """Load occupations from a JSON file."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data['occupations']

def generate_images(pipe: DebiasDiffusionPipeline, prompts: List[str], args: argparse.Namespace) -> torch.Tensor:
    """Generate images using the DebiasDiffusion pipeline."""
    generator = torch.Generator(device=pipe.device).manual_seed(args.seed)
    return pipe(
        prompt=prompts,
        num_inference_steps=50,
        guidance_scale=7.5,
        num_images_per_prompt=1,
        generator=generator,
        return_dict=False
    )

def process_batch(pipe: DebiasDiffusionPipeline, face_detector: Any, attribute_classifiers: Dict[str, Any], 
                  occupations: List[str], args: argparse.Namespace) -> Dict[str, Any]:
    """Process a batch of occupations to generate correlation data."""
    prompts = [f"A photo of the face of a {occupation}, a person" for occupation in occupations]
    results = generate_images(pipe, prompts, args)
    images, probs_list = results[0], results[2]

    valid_indices = []
    real_probs = {attr: [] for attr in args.supported_attributes}

    for j, image in enumerate(images):
        image_tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
        image_tensor = image_tensor.unsqueeze(0).to(pipe.device)
        success, _, face_chip, _ = face_detector.detect_and_align_face(image_tensor[0])
        
        if success:
            valid_indices.append(j)
            for attr in args.supported_attributes:
                attr_probs = classify_attribute(face_chip, attribute_classifiers[attr], attr)
                real_probs[attr].append(attr_probs.copy())

    batch_results = {}
    for attr in args.supported_attributes:
        attr_probs = [probs[valid_indices].tolist() for probs in probs_list[attr]]
        batch_results[attr] = {
            'estimated_probs': attr_probs,
            'real_probs': [probs.tolist() for probs in real_probs[attr]]
        }

    return batch_results, [occupations[i] for i in valid_indices]

def main(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    pipe = load_pipeline(args.model_id, device)
    face_detector = get_face_detector(torch.cuda.current_device())
    attribute_classifiers = setup_attribute_classifiers(args, device)

    occupations = load_occupations(args.occupations_file)
    print('-'*100+'\n')
    print(f"Loaded {len(occupations)} occupations from {args.occupations_file}")
    print('\n'+'-'*100)
    
    os.makedirs(args.output_dir, exist_ok=True)

    all_results = {attr: {} for attr in args.supported_attributes}
    
    for occupation in tqdm(occupations, desc="Processing occupations"):
        occupation_results = {attr: {'estimated_probs': [], 'real_probs': []} for attr in args.supported_attributes}
        
        for _ in range(0, args.num_images, args.batch_size):
            batch_size = min(args.batch_size, args.num_images - len(occupation_results[args.supported_attributes[0]]['estimated_probs']))
            batch_occupations = [occupation] * batch_size
            batch_results, valid_occupations = process_batch(pipe, face_detector, attribute_classifiers, batch_occupations, args)
            
            for attr in args.supported_attributes:
                occupation_results[attr]['estimated_probs'].extend(batch_results[attr]['estimated_probs'])
                occupation_results[attr]['real_probs'].extend(batch_results[attr]['real_probs'])
            
            if len(occupation_results[args.supported_attributes[0]]['estimated_probs']) >= args.num_images:
                break
        
        for attr in args.supported_attributes:
            all_results[attr][occupation] = {
                'estimated_probs': occupation_results[attr]['estimated_probs'][:args.num_images],
                'real_probs': occupation_results[attr]['real_probs'][:args.num_images]
            }

    output_file = os.path.join(args.output_dir, 'correlation_data.json')
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"Results saved to {output_file}")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate correlation data for DebiasDiffusion")
    parser.add_argument("--model_id", type=str, default="PalionTech/debias-diffusion-orig", help="HuggingFace model ID or path to local model")
    parser.add_argument("--num_images", type=int, default=128, help="Number of images to generate per occupation")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for image generation")
    parser.add_argument("--occupations_file", type=Path, default=BASE_DIR / "data/experiments/section_6.1.2/6.1.2_occupations-300-400.json", help="Path to JSON file containing occupations list")
    parser.add_argument("--output_dir", type=str, default=str(BASE_DIR / "results/section_6.1.2/correlation_data"), help="Directory to save output files")
    parser.add_argument("--seed", type=int, default=51904, help="Random seed for reproducibility")
    parser.add_argument("--use_legacy_gender", action="store_true", help="Use the legacy FairFace gender classifier")
    parser.add_argument("--supported_attributes", nargs='+', default=['gender', 'race'], help="List of attributes to evaluate")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args)