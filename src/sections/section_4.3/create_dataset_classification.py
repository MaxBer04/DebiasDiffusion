import torch
import argparse
import os
import random
import sys
import numpy as np
import json
from torchvision.transforms import ToTensor, ToPILImage
from tqdm import tqdm
from accelerate import Accelerator
import time
import gc
from PIL import Image
import matplotlib.pyplot as plt
import shutil

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(script_dir, '..', '..', 'custom')))
sys.path.append(os.path.abspath(os.path.join(script_dir, '..', '..', 'aux')))

from switching_pipeline import SwitchingDiffusionPipeline
from face_detection import get_face_detector
from attribute_classification import get_attribute_classifier, classify_attribute
from script_util import (
    add_dict_to_argparser,
    classifier_and_diffusion_defaults,
)

def load_diffusion_model(args):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if args.model_type == "SD-1.5":
        pipe = SwitchingDiffusionPipeline.from_pretrained(args.model_path, torch_dtype=torch.float16).to(device)
        args.num_inference_steps = 50
    return pipe

def load_occupations(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)['occupations']

def setup_attribute_info():
    return {
        'age': {'num_classes': 2, 'attributes': ["young", "old"]},
        'gender': {'num_classes': 2, 'attributes': ["male", "female"]},
        'race': {'num_classes': 4, 'attributes': ["white", "black", "asian", "indian"]}
    }

def generate_prompt(occupation, available_attributes):
    prompt_parts = []
    inserted_attributes = {}
    
    for attr, attr_info in available_attributes.items():
        if attr_info['attributes']:
            chosen_attr = random.choice(attr_info['attributes'])
            prompt_parts.append(chosen_attr)
            inserted_attributes[attr] = chosen_attr
    
    prompt = f"A photo of the face of a {' '.join(prompt_parts)} {occupation}, a person"
    return prompt, inserted_attributes

def generate_data_batch(pipe, batch_size, seed, occupations, device, face_detector, attribute_classifiers, attribute_info, available_attributes):
    torch.manual_seed(seed)
    random.seed(seed)
    
    prompts = []
    inserted_attributes_list = []
    
    for _ in range(batch_size):
        occupation = random.choice(occupations)
        prompt, inserted_attributes = generate_prompt(occupation, available_attributes)
        prompts.append(prompt)
        inserted_attributes_list.append(inserted_attributes)
    
    generator = torch.Generator(device=pipe.device).manual_seed(seed)
    
    with torch.no_grad():
        res_debiased = pipe(
            prompt=prompts,
            negative_prompt=None,
            debias=False,
            generator=generator,
            num_inference_steps=50,
            num_images_per_prompt=1,
            return_dict=False
        )
    
    generated_images = [ToTensor()(img) for img in res_debiased[0]]
    h_debiased = torch.stack(res_debiased[5][:50])  # Shape: [50, batch_size, 1280, 8, 8]
    
    print("Checking for faces and classifying attributes...")
    valid_indices = []
    images_with_faces = []
    attribute_predictions = {attr: [] for attr in attribute_info.keys()}
    valid_occupations = []
    valid_inserted_attributes = []
    
    for i, img in enumerate(generated_images):
        face_detected, face_chip = face_detector.detect_and_align_face(img)
        if face_detected:
            valid_indices.append(i)
            images_with_faces.append(face_chip)
            for attr, classifier in attribute_classifiers.items():
                probs = classify_attribute(face_chip, classifier, attr)
                attribute_predictions[attr].append(probs)
            valid_occupations.append(occupations[i])
            valid_inserted_attributes.append(inserted_attributes_list[i])
    
    h_debiased = h_debiased[:, valid_indices]
    
    print(f"Retained images after face check: {len(valid_indices)}/{len(prompts)}")
    
    del generated_images, res_debiased
    torch.cuda.empty_cache()
    
    return h_debiased, attribute_predictions, images_with_faces, valid_occupations, valid_inserted_attributes

def create_dataset(args, pipe, occupations, device, face_detector, attribute_classifiers, attribute_info):
    batch_size = args.batch_size
    output_dir = os.path.join(script_dir, args.output_path)
    temp_dir = os.path.join(output_dir, "temp")
    os.makedirs(temp_dir, exist_ok=True)
    
    class_counts = {attr: {class_: 0 for class_ in info['attributes']} for attr, info in attribute_info.items()}
    target_counts = {attr: args.num_images // len(info['attributes']) for attr, info in attribute_info.items()}
    
    available_attributes = {attr: {'attributes': list(info['attributes'])} for attr, info in attribute_info.items()}
    
    mismatch_counts = {attr: {class_: 0 for class_ in info['attributes']} for attr, info in attribute_info.items()}
    total_processed = 0
    
    start_time = time.time()
    pbar = tqdm(total=args.num_images, desc="Creating dataset")
    
    temp_file_count = 0
    
    while any(any(count < target_counts[attr] for count in attr_counts.values()) for attr, attr_counts in class_counts.items()):
        seed = args.seed + total_processed
        h_debiased, attribute_preds, images_with_faces, batch_occupations, inserted_attributes = generate_data_batch(
            pipe, batch_size, seed, occupations, device, face_detector, attribute_classifiers, attribute_info, available_attributes
        )

        batch_dataset = []
        for j in range(h_debiased.shape[1]):
            is_valid = True
            pred_classes = {}

            for attr, preds in attribute_preds.items():
                pred = preds[j]
                pred_class = attribute_info[attr]['attributes'][np.argmax(pred)]
                pred_classes[attr] = pred_class
                
                if class_counts[attr][pred_class] >= target_counts[attr]:
                    is_valid = False
                    break
                
                inserted_value = inserted_attributes[j].get(attr)
                if inserted_value and inserted_value.lower() != pred_class.lower():
                    mismatch_counts[attr][inserted_value] += 1
            
            if is_valid:
                dataset_item = {
                    "h_debiased": h_debiased[:, j].cpu(),  # Shape: [50, 1280, 8, 8]
                    "labels": {attr: preds[j] for attr, preds in attribute_preds.items()},
                    "occupation": batch_occupations[j],
                    "inserted_attributes": inserted_attributes[j],
                    "image": images_with_faces[j]
                }
                
                batch_dataset.append(dataset_item)
                for attr, pred_class in pred_classes.items():
                    class_counts[attr][pred_class] += 1
                    if class_counts[attr][pred_class] == target_counts[attr]:
                        if pred_class in available_attributes[attr]['attributes']:
                            available_attributes[attr]['attributes'].remove(pred_class)
                        print(f"Class {pred_class} for attribute {attr} is now full!")

                total_processed += 1
                pbar.update(1)
                
                if total_processed % 100 == 0:
                    print(f"\nProcessed {total_processed} images")
                    print("Current class counts:")
                    for attr, counts in class_counts.items():
                        print(f"{attr.capitalize()}:")
                        for class_, count in counts.items():
                            print(f"  {class_}: {count}/{target_counts[attr]}")
            
        # Save batch to temporary file
        temp_file_path = os.path.join(temp_dir, f"temp_dataset_{temp_file_count}.pt")
        torch.save(batch_dataset, temp_file_path)
        temp_file_count += 1
        
        del h_debiased, attribute_preds, images_with_faces, batch_occupations, inserted_attributes, batch_dataset
        gc.collect()
        torch.cuda.empty_cache()
    
    pbar.close()
    end_time = time.time()
    total_time = end_time - start_time
    print(f"Total time taken to create dataset: {total_time:.2f} seconds")
    
    mismatch_stats = calculate_mismatch_stats(mismatch_counts, total_processed)
    
    return class_counts, mismatch_stats, temp_dir, total_processed

def calculate_mismatch_stats(mismatch_counts, total_processed):
    if isinstance(mismatch_counts, dict):
        mismatch_percentages = {
            attr: {
                class_: (count / total_processed) * 100 if total_processed > 0 else 0
                for class_, count in attr_counts.items()
            } if isinstance(attr_counts, dict) else 0
            for attr, attr_counts in mismatch_counts.items()
        }
        total_mismatch_counts = {
            attr: sum(counts.values()) if isinstance(counts, dict) else counts
            for attr, counts in mismatch_counts.items()
        }
    else:
        mismatch_percentages = 0
        total_mismatch_counts = mismatch_counts

    total_mismatch_percentages = {
        attr: (count / total_processed) * 100 if total_processed > 0 else 0
        for attr, count in total_mismatch_counts.items()
    } if isinstance(total_mismatch_counts, dict) else 0

    overall_mismatch_percentage = (sum(total_mismatch_counts.values()) / (total_processed * len(mismatch_counts))) * 100 if isinstance(total_mismatch_counts, dict) and total_processed > 0 and len(mismatch_counts) > 0 else 0
    
    return {
        "overall_mismatch_percentage": overall_mismatch_percentage,
        "total_mismatch_percentages": total_mismatch_percentages,
        "mismatch_percentages": mismatch_percentages
    }

def save_dataset(class_counts, mismatch_stats, temp_dir, num_images, output_dir):
    dataset_dir = os.path.join(output_dir, f"{num_images//1000}k")
    os.makedirs(dataset_dir, exist_ok=True)
    
    # Combine all temporary datasets
    combined_dataset = []
    for temp_file in os.listdir(temp_dir):
        if temp_file.endswith(".pt"):
            temp_data = torch.load(os.path.join(temp_dir, temp_file))
            combined_dataset.extend(temp_data)
    
    # Save combined dataset
    dataset_path = os.path.join(dataset_dir, "dataset.pt")
    torch.save(combined_dataset, dataset_path)
    print(f"Combined dataset saved to {dataset_path}")
    
    # Save stats
    stats_path = os.path.join(dataset_dir, "stats.json")
    with open(stats_path, 'w') as f:
        json.dump({"class_counts": class_counts, "mismatch_stats": mismatch_stats}, f, indent=2)
    print(f"Class counts and mismatch statistics saved to {stats_path}")
    
    # Save sample images and labels
    samples_dir = os.path.join(dataset_dir, "samples")
    os.makedirs(samples_dir, exist_ok=True)
    num_samples = min(20, len(combined_dataset))
    
    for i in range(num_samples):
        sample = random.choice(combined_dataset)
        image = sample['image']
        
        # Convert tensor to PIL Image
        image = ToPILImage()(image.squeeze())
        
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        plt.axis('off')
        plt.title(f"Sample {i+1}")
        
        # Add text information
        info_text = f"Occupation: {sample['occupation']}\n\n"
        info_text += "\n".join([f"{attr}: {probs}" for attr, probs in sample['labels'].items()])
        info_text += f"\n\nInserted attributes: {sample['inserted_attributes']}"
        
        plt.figtext(0.5, 0.02, info_text, ha="center", fontsize=10, bbox={"facecolor":"white", "alpha":0.8, "pad":5})
        
        plt.tight_layout()
        plt.savefig(os.path.join(samples_dir, f"sample_{i+1}.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save the original image
        image.save(os.path.join(samples_dir, f"sample_{i+1}_original.png"))

    print(f"Sample visualizations saved to {samples_dir}")
    
    # Clean up temporary directory
    shutil.rmtree(temp_dir)
    print(f"Temporary directory {temp_dir} removed")

def main():
    args = create_argparser().parse_args()
    
    accelerator = Accelerator(mixed_precision="fp16" if args.use_fp16 else "no")
    
    pipe = load_diffusion_model(args)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    face_detector = get_face_detector(torch.cuda.current_device())
    attribute_info = setup_attribute_info()
    attribute_classifiers = {attr: get_attribute_classifier(attr, device) for attr in attribute_info.keys()}
    occupations = load_occupations(os.path.join(script_dir, 'occupations.json'))
    
    start_time = time.time()
    class_counts, mismatch_stats, temp_dir, total_processed = create_dataset(args, pipe, occupations, device, face_detector, attribute_classifiers, attribute_info)
    end_time = time.time()
    total_time = end_time - start_time
    print(f"Total time taken for entire process: {total_time:.2f} seconds")
    
    output_dir = os.path.join(script_dir, args.output_path)
    save_dataset(class_counts, mismatch_stats, temp_dir, total_processed, output_dir)

def create_argparser():
    defaults = classifier_and_diffusion_defaults()
    defaults.update(dict(
        use_fp16=True,
        model_path="runwayml/stable-diffusion-v1-5",
        model_type="SD-1.5",
        seed=53467,
        output_path="datasets_qqff",
        num_images=2000,
        batch_size=32,
    ))
    
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser

if __name__ == "__main__":
    main()