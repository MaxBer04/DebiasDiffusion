import torch
import torchvision
from diffusers import StableDiffusionPipeline
import os
import sys
import random
import numpy as np
import argparse
import json
from tqdm import tqdm
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.append(str(SCRIPT_DIR.parent.parent / 'custom'))
sys.path.append(str(SCRIPT_DIR.parent.parent / 'aux'))

from debias_diffusion_pipeline import DebiasDiffusionPipeline
from face_detection import get_face_detector
from attribute_classification import get_attribute_classifier, classify_attribute

def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    model_id = args.model_id
    pipe = DebiasDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe = pipe.to("cuda")

    classifiers_base_path = os.path.join(SCRIPT_DIR, "classifiers_all", "classifiers_qqff", "5k")

    for attr in args.supported_attributes:
        if attr == 'gender':
            bias_range = (0, 0.5)
        elif attr == 'race':
            bias_range = (0,0.75)
        elif attr == 'age':
            bias_range = (0, 1.125)
        
        pipe.set_attribute_params(
            attribute=attr,
            distribution=[0.5, 0.5] if attr != 'race' else [0.25, 0.25, 0.25, 0.25],
            bias_range=bias_range,
            classifier_path=os.path.join(classifiers_base_path, f"{attr}_5k_e100_bs256_lr0.0001_tv0.8", "best_model.pt"),
            num_classes=2 if attr != 'race' else 4,
            model_type="linear",
            default_assignments=None,
            default_switch_step=None,
        )

    #pipe.set_tau_bias(24) 
    #pipe.set_iota_step_range([4,19]) 
    pipe.set_debiasing_options(use_debiasing=False, use_distribution_guidance=False, interpolation_method='linear')
    pipe.collect_probs = True

    all_results = {attr: {} for attr in args.supported_attributes}
    
    face_detector = get_face_detector(0)
    attribute_classifiers = {
        attr: get_attribute_classifier(attr, 'cuda:0', use_legacy_gender=args.use_legacy_gender if attr == 'gender' else False)
        for attr in args.supported_attributes
    }

    os.makedirs(args.output_dir, exist_ok=True)

    # Main loop with progress bar for occupations
    for occupation in tqdm(args.occupations, desc="Processing occupations"):
        prompt = f"A photo of the face of a {occupation}, a person"
        
        generator = torch.Generator(device="cuda").manual_seed(args.seed)
        
        results = pipe(
            prompt=[prompt] * args.num_images,
            num_inference_steps=50,
            guidance_scale=7.5,
            num_images_per_prompt=1,
            generator=generator,
            return_dict=False,
        )
        images = results[0]
        probs_list = results[2]  # This is now a dictionary with attributes as keys

        valid_images = []
        real_probs = {attr: [] for attr in args.supported_attributes}
        valid_indices = []

        for j, image in enumerate(images):
            image_tensor = torchvision.transforms.ToTensor()(image).unsqueeze(0).to('cuda:0')
            success, bbox, face_chip, aligned_face_chip = face_detector.detect_and_align_face(image_tensor[0])
            
            if success:
                valid_images.append(image_tensor[0])
                valid_indices.append(j)
                for attr in args.supported_attributes:
                    attr_probs = classify_attribute(face_chip, attribute_classifiers[attr], attr)
                    real_probs[attr].append(attr_probs.copy())

        valid_images = torch.stack(valid_images)
        for attr in args.supported_attributes:
            real_probs[attr] = np.stack(real_probs[attr])

        num_valid_images = len(valid_images)
        print(f"Total valid images (with detected faces) for {occupation}: {num_valid_images}")

        # Filter probs for valid images
        for attr in args.supported_attributes:
            attr_probs = [probs[valid_indices].tolist() for probs in probs_list[attr]]
            all_results[attr][occupation] = {
                'estimated_probs': attr_probs,
                'real_probs': real_probs[attr].tolist()
            }

    # Save all results in a easily consumable format
    with open(os.path.join(args.output_dir, 'all_results.json'), 'w') as f:
        json.dump(tensor_to_list(all_results), f)

def tensor_to_list(obj):
    if isinstance(obj, torch.Tensor):
        return obj.cpu().numpy().tolist()
    elif isinstance(obj, dict):
        return {k: tensor_to_list(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [tensor_to_list(v) for v in obj]
    else:
        return obj

def parse_args():
    occupations = ["broadcast technician", "lawyer", "terrazzo finisher", "jurist", "postal service mail sorter", "healthcare social worker", "head cook", "chemical equipment tender", "extraction worker", "barber", "healthcare practitioner", "chemical plant and system operator", "social science research assistant", "architectural and civil drafter", "biologist", "copy marker", "structural metal fabricator", "human resources manager", "boxer", "wedding planner", "social work teacher", "fishing and hunting worker", "transportation worker", "library science teacher", "earth driller", "social and community service manager", "analyst", "ship engineer", "proprietor", "sales worker", "adjunct professor", "stationary engineer", "craft artist", "industrialist", "chief executive", "courier", "conciliator", "undersecretary", "film editor", "religious worker", "power tool repairer", "air traffic controller", "locksmiths and safe repairer", "makeup artist", "claims adjuster", "telecommunications line installer", "psychiatrist", "cement mason", "management analyst", "human resources specialist", "landlord", "lifeguard", "health education specialist", "broadcast announcer", "purchasing manager", "mechanical drafter", "clerk", "furniture finisher", "business teacher", "understudy", "baker", "coil finisher", "audiologist", "meeting, convention, and event planner", "anthropology teacher", "parking enforcement worker", "pharmacy aide", "event planner", "medical assistant", "counseling psychologist", "psychology teacher", "calibration technician", "booth cashier", "mail superintendent", "command and control center specialist", "economics professor", "orthotist", "bicycle repairer", "dentist", "bookkeeper", "coin machine servicer", "social scientist", "service unit operator", "magistrate judge", "outdoor power equipment mechanic", "policy processing clerk", "environmental scientist", "farm equipment mechanic", "negotiator", "building cleaning worker", "news analyst", "computer hardware engineer", "author", "home health aide", "flagger", "paperhanger", "religion teacher", "podiatrist", "parking attendant", "chef"]
    parser = argparse.ArgumentParser(description="Generate biased images and save data")
    parser.add_argument("--model_id", type=str, default="PalionTech/debias-diffusion-orig", help="Model ID for the diffusion model")
    parser.add_argument("--num_images", type=int, default=64, help="Number of images to generate per occupation")
    parser.add_argument("--occupations", nargs='+', default=occupations, help="List of occupations to generate images for")
    parser.add_argument("--output_dir", type=str, default=SCRIPT_DIR / "output_correlations_NEW", help="Directory to save output files")
    parser.add_argument("--seed", type=int, default=51904, help="Random seed for reproducibility")
    parser.add_argument("--use_legacy_gender", default=True, action="store_true", help="Use the legacy FairFace gender classifier")
    parser.add_argument("--supported_attributes", default=['gender', 'race'], nargs='+', help="The list of attributes to evaluate") #'age', 
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args)