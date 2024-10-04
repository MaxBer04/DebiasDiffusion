import os
import argparse
import pandas as pd
import shutil
from pathlib import Path
from tqdm import tqdm
import sys
from torchvision.transforms import ToTensor
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'custom')))

# Import face detection
from face_detection import get_face_detector

def load_metadata(csv_path):
    if not csv_path:
        return None
    df = pd.read_csv(csv_path)
    if 'occupation' in df.columns:
        return df, 'occupation'
    elif 'prompt' in df.columns:
        return df, 'laion'
    else:
        raise ValueError("Unrecognized CSV format. Expected 'occupation' or 'prompt' column.")

def get_image_files(input_dir, dataset_type):
    image_files = []
    for item in os.listdir(input_dir):
        item_path = os.path.join(input_dir, item)
        if os.path.isdir(item_path):
            for file in os.listdir(item_path):
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_files.append((os.path.join(item_path, file), item))
    return image_files

def get_group_name(row, dataset_type):
    if dataset_type == 'occupation':
        return f"{row['gender']}_{row['age']}_{row['race']}"
    else:  # LAION dataset
        return row['prompt']

def copy_images(image_files, metadata, output_dir, use_group_folders, dataset_type, face_detector=None):
    os.makedirs(output_dir, exist_ok=True)
    face_count = 0
    
    for image_path, group in tqdm(image_files, desc="Processing images"):
        image_name = os.path.basename(image_path)
        
        # Check for face if face detection is enabled
        if face_detector:
            image = Image.open(image_path)
            image_tensor = ToTensor()(image).unsqueeze(0)
            has_face, _, _, _ = face_detector.detect_and_align_face(image_tensor[0])
            if not has_face:
                continue
            face_count += 1
        
        if metadata is not None and use_group_folders:
            if dataset_type == 'occupation':
                attr_rows = metadata[metadata['occupation'] == group]
            else:  # LAION dataset
                attr_rows = metadata[metadata['prompt'] == group]
            
            if attr_rows.empty:
                print(f"Warning: No attribute data found for group {group}")
                continue
            
            attr_row = attr_rows.iloc[0]
            folder_name = get_group_name(attr_row, dataset_type)
            group_dir = os.path.join(output_dir, folder_name)
            os.makedirs(group_dir, exist_ok=True)
            dest_path = os.path.join(group_dir, image_name)
        else:
            dest_path = os.path.join(output_dir, image_name)
        
        shutil.copy2(image_path, dest_path)
    
    return face_count

def process_dataset(input_dir, output_dir, csv_path, use_group_folders, use_face_detection, gpu_id, dataset_type):
    if csv_path:
        print(f"Loading metadata for {input_dir}...")
        metadata, detected_dataset_type = load_metadata(csv_path)
        print(f"Detected dataset type: {detected_dataset_type}")
        dataset_type = detected_dataset_type
    else:
        print("No CSV file provided. Images will be copied without grouping.")
        metadata = None
    
    print(f"Gathering image files for {input_dir}...")
    image_files = get_image_files(input_dir, dataset_type)
    
    if not image_files:
        print(f"No images found in direct subdirectories of {input_dir}. Skipping this dataset.")
        return

    face_detector = None
    if use_face_detection:
        print("Initializing face detector...")
        face_detector = get_face_detector(gpu_id)

    print(f"Processing {len(image_files)} images from {input_dir} to {output_dir}")
    face_count = copy_images(image_files, metadata, output_dir, 
                             use_group_folders and metadata is not None, 
                             dataset_type, face_detector)
    
    if use_face_detection:
        print(f"Total images with detected faces in {input_dir}: {face_count}")
    
    print(f"Finished processing {input_dir}")

def main(args):
    input_datasets = [d for d in os.listdir(args.input_dir) if os.path.isdir(os.path.join(args.input_dir, d))]
    
    for dataset in input_datasets:
        input_dataset_path = os.path.join(args.input_dir, dataset)
        output_dataset_path = os.path.join(args.output_dir, dataset)
        csv_path = os.path.join(input_dataset_path, "metadata.csv") if os.path.exists(os.path.join(input_dataset_path, "metadata.csv")) else None
        
        print(f"\nProcessing dataset: {dataset}")
        process_dataset(input_dataset_path, output_dataset_path, csv_path, 
                        args.use_group_folders, args.use_face_detection, 
                        args.gpu_id, args.dataset_type)
    
    print("\nAll datasets processed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reorganize images based on attribute classifications with optional face detection for multiple datasets")
    parser.add_argument("--input_dir", type=str, default=SCRIPT_DIR / "datasets", help="Input directory containing multiple dataset directories")
    parser.add_argument("--output_dir", type=str, default=SCRIPT_DIR / "reorganized_datasets", help="Output directory for reorganized datasets")
    parser.add_argument("--use_group_folders", action="store_true", help="Organize images into group-specific folders")
    parser.add_argument("--use_face_detection", action="store_true", default=True, help="Use face detection to filter images")
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID for face detection")
    parser.add_argument("--dataset_type", type=str, choices=['occupation', 'laion'], default='laion', help="Type of dataset: occupation (templated) or laion (non-templated)")
    args = parser.parse_args()

    main(args)