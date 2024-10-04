import os
import json
import argparse
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import requests
from PIL import Image
from io import BytesIO
import torch
import clip
from torchvision import transforms
from transformers import AutoFeatureExtractor, AutoModelForImageClassification
import time

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_URL = "https://the-eye.eu/public/AI/cah/laion5b/"
METADATA_URL = BASE_URL + "metadata/laion2B-en/"

def load_occupations(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data['occupations']

def download_parquet_file(url, save_path, max_retries=3):
    for attempt in range(max_retries):
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            total_size = int(response.headers.get('content-length', 0))
            block_size = 8192  # 8 KB
            
            with open(save_path, 'wb') as file, tqdm(
                desc=f"Downloading {save_path.name} (Attempt {attempt + 1}/{max_retries})",
                total=total_size,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
            ) as progress_bar:
                for data in response.iter_content(block_size):
                    size = file.write(data)
                    progress_bar.update(size)
            
            # Verify the downloaded file
            try:
                pd.read_parquet(save_path)
                return True  # File is valid
            except:
                print(f"URL: {url}")
                print(f"Downloaded file is invalid. Retrying... (Attempt {attempt + 1}/{max_retries})")
                time.sleep(1)  # Wait for 1 second before retrying
        except Exception as e:
            print(f"Error downloading file: {e}. Retrying... (Attempt {attempt + 1}/{max_retries})")
            time.sleep(1)  # Wait for 1 second before retrying
    
    print(f"Failed to download {save_path.name} after {max_retries} attempts.")
    return False

def load_laion_subset(num_samples):
    print(f"Loading {num_samples} samples from LAION metadata...")
    metadata_dir = SCRIPT_DIR / "laion_metadata"
    metadata_dir.mkdir(exist_ok=True)

    df_list = []
    total_rows = 0
    
    for i in range(0,128):  # There are 128 parquet files in the metadata folder
        file_name = f"part-{i:05d}-5114fd87-297e-42b0-9d11-50f1df323dfa-c000.snappy.parquet"
        file_path = metadata_dir / file_name
        file_url = METADATA_URL + file_name
        
        if not file_path.exists() or file_path.stat().st_size == 0:
            success = download_parquet_file(file_url, file_path)
            if not success:
                continue
        
        try:
            df = pd.read_parquet(file_path)
            df_list.append(df)
            total_rows += len(df)
            
            if total_rows >= num_samples:
                break
        except Exception as e:
            print(f"Error reading {file_name}: {e}. Skipping this file.")
    
    if not df_list:
        raise ValueError("No valid parquet files could be loaded. Please check your internet connection and try again.")
    
    combined_df = pd.concat(df_list, ignore_index=True)
    return combined_df.sample(n=min(num_samples, len(combined_df)))

def setup_clip_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    return model, preprocess, device

def setup_face_detector():
    model = AutoModelForImageClassification.from_pretrained("microsoft/resnet-50")
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return preprocess, model

def filter_images_with_faces(images, preprocess, model):
    device = next(model.parameters()).device
    processed_images = torch.stack([preprocess(img) for img in images]).to(device)
    with torch.no_grad():
        outputs = model(processed_images)
    probs = outputs.logits.softmax(dim=-1)
    return [prob[0].item() > 0.5 for prob in probs]  # Threshold can be adjusted

def encode_text(clip_model, text, device):
    return clip_model.encode_text(clip.tokenize(text).to(device))

def compute_similarity(clip_model, image_features, text_features):
    return (image_features @ text_features.T).squeeze().item()

def download_and_process_image(url, preprocess):
    try:
        response = requests.get(url, timeout=10)
        img = Image.open(BytesIO(response.content)).convert("RGB")
        return preprocess(img).unsqueeze(0), img
    except:
        return None, None

def sample_images_for_occupation(laion_subset, occupation, num_samples, clip_model, clip_preprocess, device, face_preprocess, face_model):
    print(f"Sampling images for {occupation}...")
    occupation_text = f"A photo of a {occupation}"
    occupation_features = encode_text(clip_model, occupation_text, device)
    
    sampled_images = []
    pbar = tqdm(total=num_samples, desc=f"Sampling {occupation}")
    
    for _, item in laion_subset.iterrows():
        if len(sampled_images) >= num_samples:
            break
        
        image_tensor, img = download_and_process_image(item['URL'], clip_preprocess)
        if image_tensor is None or img is None:
            continue
        
        # Check if image contains a face
        has_face = filter_images_with_faces([img], face_preprocess, face_model)[0]
        if not has_face:
            continue
        
        image_features = clip_model.encode_image(image_tensor.to(device))
        similarity = compute_similarity(clip_model, image_features, occupation_features)
        
        if similarity > 0.2:  # Adjust threshold as needed
            sampled_images.append((item['URL'], similarity))
            pbar.update(1)
    
    pbar.close()
    return sorted(sampled_images, key=lambda x: x[1], reverse=True)[:num_samples]

def save_sampled_images(sampled_images, occupation, output_dir):
    occupation_dir = Path(output_dir) / occupation
    occupation_dir.mkdir(parents=True, exist_ok=True)
    
    for i, (url, _) in enumerate(sampled_images):
        try:
            response = requests.get(url, timeout=10)
            img = Image.open(BytesIO(response.content))
            img.save(occupation_dir / f"{occupation}_{i+1}.jpg")
        except:
            print(f"Failed to save image {i+1} for {occupation}")

def main(args):
    occupations = load_occupations(args.occupations_file)
    
    try:
        laion_subset = load_laion_subset(args.total_samples)
    except ValueError as e:
        print(f"Error: {e}")
        return
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
    face_preprocess, face_model = setup_face_detector()
    face_model = face_model.to(device)
    
    for occupation in occupations:
        sampled_images = sample_images_for_occupation(
            laion_subset, occupation, args.samples_per_occupation,
            clip_model, clip_preprocess, device, face_preprocess, face_model
        )
        save_sampled_images(sampled_images, occupation, args.output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sample occupation-based images from LAION")
    parser.add_argument("--occupations_file", type=str, default=SCRIPT_DIR / "occupations.json", help="Path to JSON file containing occupations")
    parser.add_argument("--samples_per_occupation", type=int, default=128, help="Number of samples per occupation")
    parser.add_argument("--total_samples", type=int, default=64000, help="Total number of samples to process from LAION")
    parser.add_argument("--output_dir", type=str, default=SCRIPT_DIR / "laion_sampled_dataset", help="Output directory for sampled images")
    args = parser.parse_args()
    main(args)