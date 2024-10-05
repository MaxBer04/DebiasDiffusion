import json
import spacy
import nltk
from nltk.corpus import wordnet as wn
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import sys
import torch
from diffusers import StableDiffusionPipeline
from facenet_pytorch import MTCNN
import numpy as np
import argparse
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(BASE_DIR))

# Initialize necessary libraries and models
nlp = spacy.load("en_core_web_sm")
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def is_human_describing_noun(word):
    """Check if a noun describes humans based on its hypernyms in WordNet."""
    synsets = wn.synsets(word, pos=wn.NOUN)
    for synset in synsets:
        for hyper in synset.closure(lambda s: s.hypernyms()):
            if 'human' in hyper.name() or 'person' in hyper.name():
                return True
    return False

def extract_nouns(text):
    """Extract nouns from the given text and return their textual representation."""
    doc = nlp(text)
    nouns = [token.text for token in doc if token.pos_ == "NOUN"]
    return nouns

def analyze_prompts(prompts, dataset_type):
    """Analyze the list of prompts and return the results."""
    total = len(prompts)
    human_related = 0
    not_recognized = []
    recognized_correctly = []  # Liste für korrekt erkannte Prompts
    
    for prompt in tqdm(prompts, desc="Analyzing prompts"):
        if dataset_type == 'occupation':
            prompt = f"A photo of a {prompt}"
        nouns = extract_nouns(prompt)
        recognized = False
        for noun in nouns:
            if is_human_describing_noun(noun):
                print(f"PROMPT: {prompt} | NOUN: {noun}")
                human_related += 1
                recognized = True
                recognized_correctly.append(prompt)  # Prompts korrekt erkannt
                break
        if not recognized:
            not_recognized.append(prompt)  # Prompts nicht erkannt
    
    return human_related, total, not_recognized, recognized_correctly

def plot_nlp_results(human_related, total):
    """Create a plot of NLP analysis results and save it as PNG and SVG."""
    labels = ['Human-related', 'Not human-related']
    sizes = [human_related, total - human_related]
    colors = ['#ff9999', '#66b3ff']
    
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    ax.axis('equal')
    
    plt.title('NLP Analysis of Prompts', fontsize=16)
    plt.suptitle(f'Total prompts analyzed: {total}', fontsize=12)
    
    output_dir = SCRIPT_DIR / 'outputs'
    os.makedirs(output_dir, exist_ok=True)
    
    plt.savefig(os.path.join(output_dir, 'nlp_analysis.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'nlp_analysis.svg'), format='svg', bbox_inches='tight')
    plt.close()

def load_diffusion_model(model_id):
    """Load and return the Stable Diffusion model."""
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe = pipe.to(device)
    return pipe

def load_face_detection_model():
    """Load and return the MTCNN face detection model."""
    return MTCNN(keep_all=True, device=device)

def generate_images(pipe, prompt, num_images=50, batch_size=10):
    """Generate images using the Stable Diffusion model."""
    images = []
    total_batches = (num_images + batch_size - 1) // batch_size
    
    for _ in tqdm(range(total_batches), desc=f"Generating images for '{prompt}'", leave=False):
        current_batch_size = min(batch_size, num_images - len(images))
        batch = pipe([prompt] * current_batch_size, num_inference_steps=50)
        images.extend(batch.images)
        
        if len(images) >= num_images:
            break
    
    return images[:num_images]

def detect_faces(face_detector, images):
    """Detect faces in the generated images."""
    face_count = 0
    for image in tqdm(images, desc="Detecting faces", leave=False):
        image_np = np.array(image)
        faces, _ = face_detector.detect(image_np)
        if faces is not None:
            face_count += 1
    return face_count

def process_prompts(prompts, diffusion_model, face_detector, num_images=50, batch_size=10):
    """Process prompts to generate images and detect faces."""
    results = {}
    total_prompts = len(prompts)
    
    for i, prompt in enumerate(prompts, 1):
        print(f"\nProcessing prompt {i}/{total_prompts}: {prompt}")
        images = generate_images(diffusion_model, prompt, num_images, batch_size)
        face_count = detect_faces(face_detector, images)
        percentage = (face_count / len(images)) * 100
        results[prompt] = percentage
        print(f"Face detection result for '{prompt}': {percentage:.2f}%")
    
    return results

def plot_face_detection_results(results):
    """Plot face detection results."""
    prompts = list(results.keys())
    percentages = list(results.values())

    plt.figure(figsize=(15, 10))
    plt.bar(range(len(prompts)), percentages)
    plt.ylabel('Percentage of Images with Faces')
    plt.title('Face Detection Results for Prompts')
    plt.xticks(range(len(prompts)), prompts, rotation=90)
    plt.tight_layout()

    output_dir = BASE_DIR / 'outputs' / "NLTK_results"
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'face_detection_results.png'))
    plt.close()

def save_results(nlp_results, face_detection_results):
    """Save NLP and face detection results as JSON files."""
    output_dir = SCRIPT_DIR / 'outputs'
    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(output_dir, 'nlp_results.json'), 'w') as f:
        json.dump(nlp_results, f, indent=2)
    
    with open(os.path.join(output_dir, 'face_detection_results.json'), 'w') as f:
        json.dump(face_detection_results, f, indent=2)

def main(args):
    print("Starting the process...")
    
    # Load prompts
    with open(args.input_file, 'r') as f:
        data = json.load(f)
    prompts = data['occupations'] if args.dataset_type == 'occupation' else data['prompts']
    print(f"Loaded {len(prompts)} prompts.")

    # NLP Analysis
    print("\nPerforming NLP analysis...")
    human_related, total, not_recognized, recognized_correctly = analyze_prompts(prompts, args.dataset_type)
    
    nlp_results = {
        "total_prompts": total,
        "human_related": human_related,
        "not_recognized": not_recognized,
        "recognized_correctly": recognized_correctly,
        "percentage_human_related": (human_related / total) * 100
    }
    
    print(f"NLP Analysis Results:")
    print(f"Total prompts: {total}")
    print(f"Human-related prompts: {human_related}")
    print(f"Percentage human-related: {nlp_results['percentage_human_related']:.2f}%")
    print(f"Not recognized prompts: {len(not_recognized)}")
    print(f"Correctly recognized prompts: {len(recognized_correctly)}")

    plot_nlp_results(human_related, total)
    print("NLP analysis plot saved in the 'outputs' folder.")

    # Face Detection Analysis
    if args.run_face_detection:
        print("\nLoading diffusion model...")
        diffusion_model = load_diffusion_model(args.model_id)
        print("Diffusion model loaded.")

        print("Loading face detection model...")
        face_detector = load_face_detection_model()
        print("Face detection model loaded.")

        print("\nProcessing not recognized prompts for face detection...")
        face_detection_results = process_prompts(not_recognized, diffusion_model, face_detector, 
                                                 num_images=args.num_images, batch_size=args.batch_size)

        plot_face_detection_results(face_detection_results)
        print("Face detection results plotted and saved as 'face_detection_results.png' in the 'outputs' folder.")
    else:
        face_detection_results = {}

    # Save all results
    save_results(nlp_results, face_detection_results)
    print("\nAll results have been saved in the 'outputs' folder.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze prompts using NLP and optionally perform face detection.")
    parser.add_argument("--model-id", type=str, default="PalionTech/debias-diffusion-orig", help="HuggingFace model id")
    parser.add_argument("--input_file", type=str, default="/root/DebiasDiffusion/data/prompt_lists/6.1.1_LAION_400_NO_PERSON.json", help="Path to the JSON file containing prompts.")
    parser.add_argument("--dataset_type", type=str, choices=['occupation', 'laion'], default="laion", 
                        help="Type of dataset: 'occupation' or 'laion'.")
    parser.add_argument("--run_face_detection", action="store_true", default=True,
                        help="Run face detection on prompts not recognized as human-related.")
    parser.add_argument("--num_images", type=int, default=32, 
                        help="Number of images to generate per prompt for face detection.")
    parser.add_argument("--batch_size", type=int, default=64, 
                        help="Batch size for image generation.")
    
    args = parser.parse_args()
    main(args)
