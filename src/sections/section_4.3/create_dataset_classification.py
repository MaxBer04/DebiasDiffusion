"""
Dataset Creation for h-Space Classifier Training

This script generates a dataset for training h-space classifiers in the DebiasDiffusion project.
It uses a list of occupations to create prompts, generates images, and collects corresponding
h-space vectors along with attribute labels.

Usage:
    python src/sections/section_4.3/create_dataset_classification.py [--args]

Arguments:
    --model_id: HuggingFace model ID or path to local model (default: "runwayml/stable-diffusion-v1-5")
    --num_images: Number of images to generate per occupation (default: 10)
    --occupations_file: Path to JSON file containing occupations list (default: BASE_DIR / "data/experiments/section_4.3/occupations.json")
    --output_dir: Directory to save output files (default: BASE_DIR / "data/experiments/section_4.3/h_space_data")
    --batch_size: Batch size for image generation (default: 32)
    --use_classifiers: Use face detection and attribute classifiers for labeling (default: False)
    --seed: Random seed for reproducibility (default: 42)
    --attributes: List of attributes to consider (default: ['gender', 'race', 'age'])

Outputs:
    - h-space vectors and corresponding labels saved as PyTorch tensors
    - Metadata CSV file with generation details
"""

import torch
import os
import sys
import random
import json
import argparse
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict, Any, Optional, Tuple
from torchvision.transforms import ToTensor

# Add project root to Python path
SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent.parent.parent
sys.path.append(str(BASE_DIR))

from src.pipelines.debias_diffusion_pipeline import DebiasDiffusionPipeline
from src.utils.face_detection import get_face_detector
from src.utils.attribute_classification import get_attribute_classifier, classify_attribute

ATTRIBUTE_CLASSES = {
    'gender': ['male', 'female'],
    'race': ['white', 'black', 'asian', 'indian'],
    'age': ['young', 'old']
}

def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_occupations(file_path: Path) -> List[str]:
    """Load occupations from a JSON file."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data['occupations']

def setup_pipeline(model_id: str, device: torch.device) -> DebiasDiffusionPipeline:
    """Set up the DebiasDiffusion pipeline."""
    pipe = DebiasDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to(device)
    pipe.use_debiasing = False
    pipe.safety_checker = None
    return pipe

def setup_classifiers(device: torch.device) -> Dict[str, Any]:
    """Set up face detection and attribute classification models."""
    face_detector = get_face_detector(torch.cuda.current_device())
    classifiers = {
        'gender': get_attribute_classifier('gender', device),
        'race': get_attribute_classifier('race', device),
        'age': get_attribute_classifier('age', device)
    }
    return {'face_detector': face_detector, **classifiers}

def generate_prompt(occupation: str, attributes: List[str]) -> Tuple[str, Dict[str, str]]:
    """Generate a prompt with random attribute classes inserted."""
    inserted_classes = {}
    attribute_parts = []
    for attr in attributes:
        class_ = random.choice(ATTRIBUTE_CLASSES[attr])
        attribute_parts.append(class_)
        inserted_classes[attr] = class_
    prompt = f"A photo of the face of a {' '.join(attribute_parts)} {occupation}, a person"
    return prompt, inserted_classes

def process_batch(
    pipeline: DebiasDiffusionPipeline,
    prompts: List[str],
    batch_size: int,
    seed: int,
    use_classifiers: bool,
    classifiers: Optional[Dict[str, Any]] = None
) -> Tuple[torch.Tensor, List[Dict[str, Any]]]:
    """Process a batch of prompts to generate images and collect h-space vectors."""
    generator = torch.Generator(device=pipeline.device).manual_seed(seed)
    outputs = pipeline(
        prompt=prompts,
        num_inference_steps=50,
        guidance_scale=7.5,
        generator=generator,
        num_images_per_prompt=1,
        return_dict=False
    )
    
    images, h_vectors = outputs[0], outputs[1]
    
    batch_data = []
    for i, (image, h_vector) in enumerate(zip(images, h_vectors)):
        data = {'prompt': prompts[i], 'h_vector': h_vector}
        
        if use_classifiers:
            face_detected, face_chip = classifiers['face_detector'].detect_and_align_face(image)
            if face_detected:
                for attr in ['gender', 'race', 'age']:
                    probs = classify_attribute(face_chip, classifiers[attr], attr)
                    data[f'{attr}_probs'] = probs
            else:
                data['face_detected'] = False
        
        batch_data.append(data)
    
    return h_vectors, batch_data

def main(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    pipeline = setup_pipeline(args.model_id, device)
    classifiers = setup_classifiers(device) if args.use_classifiers else None

    occupations = load_occupations(args.occupations_file)
    
    all_h_vectors = []
    all_labels = []
    metadata = []

    for occupation in tqdm(occupations, desc="Processing occupations"):
        for _ in range(0, args.num_images, args.batch_size):
            batch_size = min(args.batch_size, args.num_images - len(all_h_vectors) % args.num_images)
            
            if args.use_classifiers:
                prompts = [f"A photo of the face of a {occupation}, a person" for _ in range(batch_size)]
            else:
                prompts = [generate_prompt(occupation, args.attributes)[0] for _ in range(batch_size)]
            
            h_vectors, batch_data = process_batch(pipeline, prompts, batch_size, args.seed, args.use_classifiers, classifiers)
            
            all_h_vectors.extend([data['h_vector'] for data in batch_data])
            
            for data in batch_data:
                label = {}
                if args.use_classifiers:
                    if 'face_detected' in data and not data['face_detected']:
                        continue
                    for attr in args.attributes:
                        label[attr] = data[f'{attr}_probs']
                else:
                    _, inserted_classes = generate_prompt(occupation, args.attributes)
                    
                    for attr in args.attributes:
                        one_hot = torch.zeros(len(ATTRIBUTE_CLASSES[attr]))
                        index = ATTRIBUTE_CLASSES[attr].index(inserted_classes[attr])
                        one_hot[index] = 1
                        label[attr] = one_hot
                all_labels.append(label)
            
            metadata.extend(batch_data)

    # Save h-vectors and labels
    torch.save(torch.stack(all_h_vectors), args.output_dir / "h_vectors.pt")
    torch.save(all_labels, args.output_dir / "labels.pt")

    # Save metadata
    import pandas as pd
    pd.DataFrame(metadata).to_csv(args.output_dir / "metadata.csv", index=False)

    print(f"Dataset created and saved to {args.output_dir}")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create dataset for h-space classifier training")
    parser.add_argument("--model_id", type=str, default="PalionTech/debias-diffusion-orig", help="HuggingFace model ID or path to local model")
    parser.add_argument("--num_images", type=int, default=10, help="Number of images to generate per occupation")
    parser.add_argument("--occupations_file", type=Path, default=BASE_DIR / "data/experiments/section_5.4.1/5.4.1_occupations_500.json", help="Path to JSON file containing occupations list")
    parser.add_argument("--output_dir", type=Path, default=BASE_DIR / "data/experiments/section_4.3/h_space_data", help="Directory to save output files")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for image generation")
    parser.add_argument("--use_classifiers", default=False, action="store_true", help="Use face detection and attribute classifiers for labeling")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--attributes", nargs='+', default=['gender', 'race', 'age'], help="List of attributes to consider")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args)