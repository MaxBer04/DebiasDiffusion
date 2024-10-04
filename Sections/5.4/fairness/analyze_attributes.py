import os
import json
import torch
from pathlib import Path
import numpy as np
from PIL import Image
import pandas as pd
from tqdm import tqdm
import warnings
import logging
import sys
import io
from torch.utils.data import Dataset, DataLoader
import argparse

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.append(str(SCRIPT_DIR.parent.parent / 'custom'))

# Suppress warnings and set logging level
warnings.filterwarnings("ignore")
logging.getLogger("insightface").setLevel(logging.ERROR)

# Import the new modules
from face_detection import get_face_detector
from attribute_classification import get_attribute_classifier, classify_attribute

# Classes for FairFace (for compatibility with the original code)
GENDER_CLASSES = ['Male', 'Female']
RACE_CLASSES = ['White', 'Black', 'Latino Hispanic', 'East Asian', 'Southeast Asian', 'Indian', 'Middle Eastern']
AGE_CLASSES = ['Young', 'Old']

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_classifiers(device, use_race7):
    global RACE_CLASSES
    gender_classifier = get_attribute_classifier('gender', device, use_legacy_gender=True)
    race_classifier = get_attribute_classifier('race', device, use_race7=use_race7)
    if not use_race7:
        RACE_CLASSES = ['White', 'Black', 'Asian', 'Indian']
    age_classifier = get_attribute_classifier('age', device)
    return gender_classifier, race_classifier, age_classifier

class ImageDataset(Dataset):
    def __init__(self, root_dir, dataset_type):
        self.root_dir = root_dir
        self.dataset_type = dataset_type
        self.image_paths = []
        self.groups = []
        
        metadata_file = os.path.join(root_dir, 'metadata.csv')
        self.metadata = pd.read_csv(metadata_file)
        
        for _, row in self.metadata.iterrows():
            group = row['occupation'] if 'occupation' in row else row['prompt']
            image_path = os.path.join(root_dir, group, f"{row['seed']}.png")
            if os.path.exists(image_path):
                self.image_paths.append(image_path)
                self.groups.append(group)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        group = self.groups[idx]
        image = Image.open(img_path).convert('RGB')
        return image, group

def custom_collate(batch):
    images = [item[0] for item in batch]
    groups = [item[1] for item in batch]
    return images, groups

def analyze_dataset(root_dir, batch_size=32, use_race7=False, dataset_type='occupation'):
    device = get_device()
    face_detector = get_face_detector(0 if device.type == 'cuda' else -1)
    gender_classifier, race_classifier, age_classifier = load_classifiers(device, use_race7)

    dataset = ImageDataset(root_dir, dataset_type)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=custom_collate)

    results = []

    total_images = len(dataset)
    processed_images = 0

    progress_bar = tqdm(total=total_images, desc="Analyzing images", unit="img")

    for images, groups in dataloader:
        for image, group in zip(images, groups):
            image_tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
            image_tensor = image_tensor.unsqueeze(0).to(device)

            success, _, face_chip, _ = face_detector.detect_and_align_face(image_tensor[0])
            
            if success:
                gender_probs = classify_attribute(face_chip, gender_classifier, 'gender')
                race_probs = classify_attribute(face_chip, race_classifier, 'race')
                age_probs = classify_attribute(face_chip, age_classifier, 'age')

                gender_pred = GENDER_CLASSES[gender_probs.argmax()]
                race_pred = RACE_CLASSES[race_probs.argmax()]
                
                # Mapping age probabilities to age classes
                age_pred = AGE_CLASSES[0] if age_probs[0] > age_probs[1] else AGE_CLASSES[1]

                result = {
                    'group': group,
                    'gender': gender_pred,
                    'race': race_pred,
                    'age': age_pred,
                    'gender_probs': gender_probs.tolist(),
                    'race_probs': race_probs.tolist(),
                    'age_probs': age_probs.tolist()
                }
                results.append(result)

            processed_images += 1
            progress_bar.update(1)

        progress_bar.set_postfix({"Processed": f"{processed_images}/{total_images}"})

    progress_bar.close()
    return results

def parse_args():
    parser = argparse.ArgumentParser(description="Analyze attributes in a dataset of images.")
    parser.add_argument("--dataset_dir", type=str, default = SCRIPT_DIR / "datasets/outputs_step25_bs32_NEW", 
                        help="Directory containing the dataset")
    parser.add_argument("--output_dir", type=str, default=SCRIPT_DIR / 'attribute_classifications', 
                        help="Directory to save the output CSV file")
    parser.add_argument("--output_filename", type=str, default='step25_bs32.csv', 
                        help="Name of the output CSV file")
    parser.add_argument("--batch_size", type=int, default=512,
                        help="Batch size for processing")
    parser.add_argument("--use_race7", action="store_true", 
                        help="Use 7-class race classifier instead of 4-class")
    parser.add_argument("--dataset_type", type=str, choices=['occupation', 'laion'], default='occupation',
                        help="Type of dataset: occupation (templated) or laion (non-templated)")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)

    print("Starting dataset analysis...")
    results = analyze_dataset(args.dataset_dir, batch_size=args.batch_size, use_race7=args.use_race7, dataset_type=args.dataset_type)
    
    print("Saving results...")
    df = pd.DataFrame(results)
    output_file = os.path.join(args.output_dir, args.output_filename)
    df.to_csv(output_file, index=False)
    
    print(f"Analysis complete. Results saved to {output_file}")
    print(f"Total images processed: {len(results)}")