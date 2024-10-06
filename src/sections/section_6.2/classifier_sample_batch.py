import torch
from torchvision.transforms import ToTensor
import numpy as np
import os
import sys
import argparse
import json
import random
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

# Add custom and auxiliary paths to system path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(script_dir, '..', '..', 'custom')))
sys.path.append(os.path.abspath(os.path.join(script_dir, '..', '..', 'aux')))

from classifier import make_classifier_model
from switching_pipeline import SwitchingDiffusionPipeline
from face_detection import get_face_detector
from logger import get_logger
from plotting_utils import plot_accuracy, plot_images_with_attributes


# Utility functions
def load_classifier(classifier_path, device, num_classes):
    """
    Load a pre-trained classifier model.
    
    The classifier expects input of shape [batch_size, 1280, 8, 8] and
    outputs logits of shape [batch_size, num_classes].
    """
    classifier = make_classifier_model(
        in_channels=1280,
        image_size=8,
        out_channels=num_classes,
        prefix="eval",
    )
    state_dict = torch.load(classifier_path, map_location=device)
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    classifier.load_state_dict(new_state_dict)
    return classifier.to(device)

def load_occupations(file_path):
    """Load occupation list from a JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)['occupations']

# Analysis function
def analyze_predictions(probs_list, prompt_attributes):
    """
    Analyze classifier predictions against prompt attributes.
    
    Args:
    - probs_list: List of probability arrays, each of shape [50, num_classes]
    - prompt_attributes: List of true attribute indices
    
    Returns:
    - List of accuracies for each timestep
    """
    if not probs_list:
        print("No data to analyze in analyze_predictions function.")
        return []

    num_steps, num_images = len(probs_list[0]), len(probs_list)
    
    def get_prediction(prob_vector):
        return np.argmax(prob_vector)

    accuracies_per_step = []
    for step in range(num_steps):
        step_preds = [get_prediction(probs[step]) for probs in probs_list]
        correct_step = sum((pred == truth) for pred, truth in zip(step_preds, prompt_attributes))
        accuracies_per_step.append(correct_step / num_images * 100)

    return accuracies_per_step


def main(args):
    # Set global seeds for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    global logger

    output_dir = os.path.join(script_dir, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    logger = get_logger(output_dir, f"classifier_evaluation")

    attribute_labels = {
        'age': ['young', 'old'],
        'race': ['white', 'black', 'asian', 'indian'],
        'gender': ['male', 'female']
    }

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # Initialize the diffusion pipeline
    # The pipe takes prompts as input and returns a tuple containing:
    # - generated images: shape [batch_size, 3, 512, 512]
    # - other outputs (NSFW flags, attention maps, etc.)
    # - probability predictions: shape [num_attributes, 50, batch_size, num_classes_for_attribute]
    pipe = SwitchingDiffusionPipeline.from_pretrained(args.model_id, torch_dtype=torch.float16 if args.use_fp16 else torch.float32).to(device)
    
    # Load classifiers for each attribute
    classifiers = {attr: load_classifier(os.path.join(script_dir, getattr(args, f"{attr}_classifier_path")), 
                                         device, len(attribute_labels[attr])) for attr in args.attributes}
    pipe.classifiers = classifiers
    
    face_detector = get_face_detector(torch.cuda.current_device()) if args.detect_faces else None
    
    imgs_dir = os.path.join(output_dir, "images")
    os.makedirs(imgs_dir, exist_ok=True)

    occupations = load_occupations(os.path.join(script_dir, 'occupations.json'))

    prompt_attributes = {attr: [] for attr in args.attributes}
    probs_list = {attr: [] for attr in args.attributes}
    all_images = []

    for batch_idx in range(0, args.num_images, args.batch_size):
        batch_seed = args.seed + batch_idx
        random.seed(batch_seed)
        batch_size = min(args.batch_size, args.num_images - batch_idx)
        
        prompts = []
        batch_prompt_attributes = {attr: [] for attr in args.attributes}
        generator = torch.Generator(device.type).manual_seed(batch_seed)
        
        # Generate prompts for the batch
        for i in range(batch_size):
            random.seed(batch_seed+i)
            occupation = random.choice(occupations)
            
            prompt_parts = []
            for attr in ['age', 'race', 'gender']:
                if attr in args.attributes and args.use_attributes_in_prompt:
                    attr_value = random.choice(attribute_labels[attr])
                    batch_prompt_attributes[attr].append(attribute_labels[attr].index(attr_value))
                    prompt_parts.append(attr_value)
            
            prompt = f"A photo of the face of a {' '.join(prompt_parts)} {occupation}, a person" if args.use_attributes_in_prompt else f"A photo of the face of a {occupation}, a person"
            prompts.append(prompt)
        
        # Generate images and get classifier predictions
        outputs = pipe(prompt=prompts, generator=generator, return_dict=False)
        batch_images, probs_per_step = outputs[0], outputs[6]
        
        # Detect faces if required
        face_detected = [face_detector.detect_and_align_face(ToTensor()(img))[0] for img in batch_images] if args.detect_faces else [True] * batch_size
        
        batch_valid_images = []
        for j, is_face_detected in enumerate(face_detected):
            if is_face_detected:
                batch_valid_images.append(ToTensor()(batch_images[j]))
                for attr in args.attributes:
                    # Store probabilities for each timestep
                    attr_probs = [step_probs[j].cpu().numpy() for step_probs in probs_per_step[attr]]
                    probs_list[attr].append(attr_probs)
                if args.use_attributes_in_prompt:
                    for attr in args.attributes:
                        prompt_attributes[attr].append(batch_prompt_attributes[attr][j])
            else:
                print(f"No face detected in image {batch_idx+j}, skipping")
        
        # Create and save image grid if valid images exist
        if batch_valid_images:
            all_images.extend(batch_valid_images)
            grid_images = torch.stack(batch_valid_images)
            grid_probs = {attr: torch.stack([torch.tensor(probs[args.classifier_step]) for probs in probs_list[attr][-len(batch_valid_images):]]) for attr in args.attributes}
            
            grid_path = os.path.join(imgs_dir, f"grid_{batch_idx // args.batch_size + 1}.png")
            plot_images_with_attributes(grid_images, grid_probs, {attr: attribute_labels[attr] for attr in args.attributes}, 
                                        grid_path, nrow=int(np.sqrt(len(batch_valid_images))), 
                                        show_legend=args.show_legend, dpi=args.dpi)

    print(f"Final probs_list: {[len(probs) for attr, probs in probs_list.items()]}")
    print(f"Final prompt_attributes: {[len(attrs) for attr, attrs in prompt_attributes.items()]}")

    # Analyze predictions and plot accuracy
    if args.use_attributes_in_prompt:
        accuracies_dict = {}
        for attr in args.attributes:
            if probs_list[attr]:
                accuracies_dict[attr] = analyze_predictions(probs_list[attr], prompt_attributes[attr])
            else:
                print(f"No data for attribute {attr}. Skipping analysis.")
        
        if accuracies_dict:
            plot_accuracy(accuracies_dict, os.path.join(output_dir, "accuracy_over_time.png"))

            for attr, accuracies in accuracies_dict.items():
                logger.info(f"\n--- Final Results for {attr.capitalize()} ---")
                logger.info(f"Final accuracy: {accuracies[-1]:.2f}%")
        else:
            print("No data to analyze. Check your inputs and face detection settings.")

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate attribute classifiers")
    parser.add_argument("--attributes", nargs='+', choices=['gender', 'race', 'age'], default=['gender', 'race', 'age'],
                        help="Attributes to evaluate")
    parser.add_argument("--gender_classifier_path", default="classifiers_all/classifiers_qqff/5k/gender_5k_e100_bs256_lr0.0001_tv0.8/best_model.pt", type=str, help="Path to the gender classifier")
    parser.add_argument("--race_classifier_path", default="classifiers_all/classifiers_qqff/5k/race_5k_e100_bs256_lr0.0001_tv0.8/best_model.pt", type=str, help="Path to the race classifier")
    parser.add_argument("--age_classifier_path", default="classifiers_all/classifiers_qqff/5k/age_5k_e100_bs256_lr0.0001_tv0.8/best_model.pt", type=str, help="Path to the age classifier")
    parser.add_argument("--model_id", type=str, default="PalionTech/debias-diffusion-orig", help="Model ID for the diffusion model")
    parser.add_argument("--num_images", type=int, default=2048, help="Total number of images to generate and evaluate")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for image generation")
    parser.add_argument("--output_dir", type=str, default="results_new", help="Directory to save output files")
    parser.add_argument("--seed", type=int, default=1904, help="Random seed for image generation")
    parser.add_argument("--show_legend", default=True, action="store_true", help="Show legend in the output images")
    parser.add_argument("--dpi", type=int, default=300, help="DPI for output images")
    parser.add_argument("--detect_faces", default=True, action="store_true", help="Only use images where faces are detected")
    parser.add_argument("--classifier_step", type=int, default=25, help="Which timestep to use for classification in the grid plot")
    parser.add_argument("--use_attributes_in_prompt", default=True, action="store_true", help="Whether to include attributes in the prompt")
    parser.add_argument("--use_fp16", default=True, action="store_true", help="Whether to generated sampled images in fp16 precision or 32")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args)