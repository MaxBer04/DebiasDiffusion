import torch
import argparse
import os
import random
import sys
import json
from tqdm import tqdm
from accelerate import Accelerator
from torchvision.transforms import ToTensor


script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(script_dir, '..', '..', 'custom')))
sys.path.append(os.path.abspath(os.path.join(script_dir, '..', '..', 'aux')))

from switching_pipeline import SwitchingDiffusionPipeline
from face_detection import get_face_detector
from script_util import add_dict_to_argparser, classifier_and_diffusion_defaults

def load_diffusion_model(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pipe = SwitchingDiffusionPipeline.from_pretrained(args.model_path, torch_dtype=torch.float16 if args.use_fp16 else torch.float32).to(device)
    return pipe

def load_occupations(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)['occupations']

def setup_attribute_info():
    return {
        'gender': {'num_classes': 2, 'attributes': ["male", "female"]},
        'race': {'num_classes': 4, 'attributes': ["white", "black", "asian", "indian"]},
        'age': {'num_classes': 2, 'attributes': ["young", "old"]}
    }

def generate_data_batch(pipe, batch_size, seed, attribute_info, occupations, device, face_detector):
    torch.manual_seed(seed)
    random.seed(seed)
    
    prompts = []
    labels = {attr: [] for attr in attribute_info}
    batch_occupations = []
    
    for _ in range(batch_size):
        occupation = random.choice(occupations)
        attrs = {}
        for attr, info in attribute_info.items():
            attr_idx = random.randint(0, info['num_classes'] - 1)
            attrs[attr] = info['attributes'][attr_idx]
            labels[attr].append(attr_idx)
        
        prompt = f"A photo of the face of a {' '.join(attrs.values())} {occupation}, a person"
        prompts.append(prompt)
        batch_occupations.append(occupation)
    
    generator = torch.Generator(device=pipe.device).manual_seed(seed)
    
    with torch.no_grad():
        res_debiased = pipe(
            prompt=prompts,
            negative_prompt=[", ".join([a for attr_info in attribute_info.values() for a in attr_info['attributes'] if a not in [attrs[attr] for attr in attrs]])] * batch_size,
            debias=False,
            generator=generator,
            num_inference_steps=50,
            num_images_per_prompt=1,
            return_dict=False
        )
    
    h_debiased = torch.stack(res_debiased[5][:50])  # Shape: [50, batch_size, 1280, 8, 8]
    
    valid_indices = []
    for i in range(h_debiased.shape[1]):
        face_detected, _ = face_detector.detect_face(ToTensor()(res_debiased[0][i]))
        if face_detected:
            valid_indices.append(i)
    
    h_debiased = h_debiased[:, valid_indices]
    labels = {attr: [labels[attr][i] for i in valid_indices] for attr in labels}
    batch_occupations = [batch_occupations[i] for i in valid_indices]
    
    return h_debiased, labels, batch_occupations

def create_dataset(args, pipe, attribute_info, occupations, device, face_detector):
    batch_size = args.batch_size
    temp_dir = os.path.join(script_dir, args.output_path, "temp")
    os.makedirs(temp_dir, exist_ok=True)
    
    dataset = []
    total_samples = 0
    seed = args.seed
    
    with tqdm(total=args.num_images, desc="Creating dataset") as pbar:
        while total_samples < args.num_images:
            h_debiased, labels, occupations = generate_data_batch(pipe, batch_size, seed, attribute_info, occupations, device, face_detector)
            
            for j in range(h_debiased.shape[1]):
                if total_samples < args.num_images:
                    dataset.append({
                        "h_debiased": h_debiased[:, j],
                        "labels": {attr: labels[attr][j] for attr in labels},
                        "occupation": occupations[j]
                    })
                    total_samples += 1
                    pbar.update(1)
            
            if len(dataset) >= args.save_interval or total_samples == args.num_images:
                temp_path = os.path.join(temp_dir, f"dataset_{len(os.listdir(temp_dir))}.pt")
                torch.save(dataset, temp_path)
                dataset = []
            
            seed += 1
            torch.cuda.empty_cache()
    
    dataset = []
    for file_name in os.listdir(temp_dir):
        temp_path = os.path.join(temp_dir, file_name)
        dataset.extend(torch.load(temp_path))
        os.remove(temp_path)
    os.rmdir(temp_dir)
    
    return dataset

def save_dataset(dataset, output_dir, num_images):
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"dataset_{num_images/1000}k.pt")
    torch.save(dataset, output_path)
    print(f"Dataset saved to {output_path}")

def main():
    args = create_argparser().parse_args()
    
    accelerator = Accelerator(mixed_precision="fp16" if args.use_fp16 else "no")
    
    pipe = load_diffusion_model(args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    face_detector = get_face_detector(torch.cuda.current_device())
    occupations = load_occupations(os.path.join(script_dir, 'occupations.json'))
    
    attribute_info = setup_attribute_info()
    
    dataset = create_dataset(args, pipe, attribute_info, occupations, device, face_detector)
    
    output_dir = os.path.join(script_dir, args.output_path)
    save_dataset(dataset, output_dir, args.num_images)

def create_argparser():
    defaults = classifier_and_diffusion_defaults()
    defaults.update(dict(
        use_fp16=True,
        model_path="runwayml/stable-diffusion-v1-5",
        seed=9948485,
        output_path="datasets",
        num_images=5000,
        batch_size=32,
        save_interval=200,
    ))
    
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser

if __name__ == "__main__":
    main()