import torch
import argparse
import os
import sys
import json
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(script_dir, '..', '..', 'custom')))
sys.path.append(os.path.abspath(os.path.join(script_dir, '..', '..', 'aux')))

from classifier import make_classifier_model
from script_util import add_dict_to_argparser, classifier_and_diffusion_defaults

def load_classifier(classifier_path, device, num_classes, model_type, use_fp16=True):
    classifier = make_classifier_model(
        in_channels=1280,
        image_size=8,
        out_channels=num_classes,
        model_type=model_type
    )
    state_dict = torch.load(classifier_path, map_location=device)
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    classifier.load_state_dict(new_state_dict)
    
    if use_fp16:
        classifier = classifier.half()
    else:
        classifier = classifier.float()
    
    return classifier.to(device)

def load_dataset(dataset_path):
    return torch.load(dataset_path)

def mask_h_vectors(h_debiased, timestep):
    """
    Mask h_debiased vectors for timesteps after the current one.
    Fill the rest with zeros (neutral value).
    """
    masked = h_debiased.clone()
    masked[:, timestep+1:] = 0
    return masked

def evaluate_classifiers(classifiers, dataset, device, batch_size, use_fp16):
    results = {name: [] for name in classifiers}
    
    num_batches = len(dataset) // batch_size
    
    for batch_idx in tqdm(range(num_batches), desc="Evaluating classifiers"):
        start_idx = batch_idx * batch_size
        end_idx = start_idx + batch_size
        batch_data = dataset[start_idx:end_idx]
        
        h_debiased = torch.stack([data['h_debiased'] for data in batch_data]).to(device)
        if use_fp16:
            h_debiased = h_debiased.half()
        labels = {attr: torch.tensor([data['labels'][attr] for data in batch_data], device=device) 
                  for attr in batch_data[0]['labels']}
        
        for name, (classifier, attr, model_type) in classifiers.items():
            if model_type == "single_layer":
                predictions = []
                for t in range(50):
                    masked_h = mask_h_vectors(h_debiased, t)
                    logits = classifier(masked_h)  # Shape: [batch_size, 50, num_classes]
                    preds = torch.argmax(logits[:, t], dim=-1)
                    correct = (preds == labels[attr]).float()
                    predictions.append(correct.mean().item())
                results[name].append(predictions)
            else:  # "multi_layer" or "resnet18"
                predictions = []
                for t in range(50):
                    masked_h = mask_h_vectors(h_debiased, t)
                    logits = classifier(masked_h[:, t], torch.full((batch_size,), t, device=device))
                    preds = torch.argmax(logits, dim=1)
                    correct = (preds == labels[attr]).float()
                    predictions.append(correct.mean().item())
                results[name].append(predictions)
    
    return {name: np.mean(preds, axis=0) for name, preds in results.items()}

def plot_results(results, output_path):
    plt.figure(figsize=(12, 8))
    sns.set_style("whitegrid")

    color_palette = plt.cm.Set2.colors + plt.cm.Set3.colors + plt.cm.tab20.colors

    attributes = set(attr.split('_')[0] for attr in results.keys())
    colors = plt.cm.get_cmap('tab10')(np.linspace(0, 1, len(attributes)))
    color_dict = dict(zip(attributes, colors))

    linestyles = [
        'solid', 'dotted', 'dashed', 'dashdot', 
        (0, (1, 1)), (0, (5, 5)), (0, (3, 5, 1, 5)), (0, (3, 1, 1, 1)),
        (0, (1, 10)), (0, (5, 10)), (0, (3, 10, 1, 10)), (0, (3, 5, 1, 5, 1, 5)),
        (0, (3, 1, 1, 1, 1, 1)), (0, (5, 1)), (0, (1, 1, 5, 5)), (0, (5, 5, 1, 5))
    ]

    grouped_results = {}
    for name, accuracies in results.items():
        attr = name.split('_')[0]
        if attr not in grouped_results:
            grouped_results[attr] = []
        grouped_results[attr].append((name, accuracies))

    for attr, group in grouped_results.items():
        color = color_dict[attr]
        for i, (name, accuracies) in enumerate(group):
            linestyle = linestyles[i % len(linestyles)]
            plt.plot(range(50), accuracies * 100, label=f"{attr.capitalize()} - {name}",
                     color=color, linestyle=linestyle)

    plt.xlabel("Timestep", fontsize=14)
    plt.ylabel("Accuracy (%)", fontsize=14)
    plt.ylim(0, 100)
    plt.title("Classifier Accuracy over Diffusion Timesteps", fontsize=16)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    args = create_argparser().parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    args.dataset_path = os.path.join(script_dir, args.dataset_path)
    
    dataset = load_dataset(args.dataset_path)
    
    classifiers = {}
    for classifier_info in args.classifiers:
        name, path, attr, num_classes, model_type = classifier_info.split(',')
        path = os.path.join(script_dir, path)
        classifiers[name] = (load_classifier(path, device, int(num_classes), model_type, args.use_fp16), attr, model_type)
    
    results = evaluate_classifiers(classifiers, dataset, device, args.batch_size, args.use_fp16)
    
    output_dir = os.path.join(script_dir, args.output_path)
    os.makedirs(output_dir, exist_ok=True)
    plot_results(results, os.path.join(output_dir, "classifier_accuracy.png"))
    plot_results(results, os.path.join(output_dir, "classifier_accuracy.svg"))

def create_argparser():
    attribute1 = "race"
    attribute2 = "age"
    attribute3 = "gender"
    k=5
    defaults = classifier_and_diffusion_defaults()
    defaults.update(dict(
        dataset_path="evaluation_datasets/dataset_5k.pt",
        output_path=f"results/qqff/{k}k",
        batch_size=256,
        use_fp16=True,
        classifiers=[
            f"{attribute1}_{k}k_qqff_v2,classifiers_all/classifiers_qqff_v2/{k}k/{attribute1}_{k}k_e100_bs256_lr0.0001_tv0.8_v2/best_model.pt,{attribute1},4,linear",
            f"{attribute1}_{k}k_qqff,classifiers_all/classifiers_qqff/{k}k/{attribute1}_{k}k_e100_bs256_lr0.0001_tv0.8/best_model.pt,{attribute1},4,linear",
            f"{attribute1}_{k}k_qq,classifiers_all/classifiers_qq/{k}k/{attribute1}_{k}k_e60_bs256_lr0.0001_tv0.8/best_model.pt,{attribute1},4,linear",
            
            f"{attribute2}_{k}k_qqff_v2,classifiers_all/classifiers_qqff_v2/{k}k/{attribute2}_{k}k_e100_bs256_lr0.0001_tv0.8_v2/best_model.pt,{attribute2},2,linear",
            f"{attribute2}_{k}k_qqff,classifiers_all/classifiers_qqff/{k}k/{attribute2}_{k}k_e100_bs256_lr0.0001_tv0.8/best_model.pt,{attribute2},2,linear",
            f"{attribute2}_{k}k_qq,classifiers_all/classifiers_qq/{k}k/{attribute2}_{k}k_e60_bs256_lr0.0001_tv0.8/best_model.pt,{attribute2},2,linear",
            
            f"{attribute3}_{k}k_qqff_v2,classifiers_all/classifiers_qqff_v2/{k}k/{attribute3}_{k}k_e100_bs256_lr0.0001_tv0.8_v2/best_model.pt,{attribute3},2,linear",
            f"{attribute3}_{k}k_qqff,classifiers_all/classifiers_qqff/{k}k/{attribute3}_{k}k_e100_bs256_lr0.0001_tv0.8/best_model.pt,{attribute3},2,linear",
            f"{attribute3}_{k}k_qq,classifiers_all/classifiers_qq/{k}k/{attribute3}_{k}k_e60_bs256_lr0.0001_tv0.8/best_model.pt,{attribute3},2,linear",
        ]
    ))
    
    # All qq only
    """ f"{attribute1}_01k,classifiers_all/.1k/{attribute1}_01_e80/best_model.pt,{attribute1},4,multi_layer",
    f"{attribute1}_05k,classifiers_all/.5k/{attribute1}_05_e60/best_model.pt,{attribute1},4,multi_layer",
    f"{attribute1}_1k,classifiers_all/1k/{attribute1}_1k_e60_bs500_lr0.0001/best_model.pt,{attribute1},4,multi_layer",
    f"{attribute1}_2k,classifiers_all/2k/{attribute1}_2k_e60_bs500_lr0.0001/best_model.pt,{attribute1},4,multi_layer",
    f"{attribute1}_5k,classifiers_all/5k/{attribute1}_5k_e60_bs256_lr0.0001_tv0.8/best_model.pt,{attribute1},4,multi_layer",
    
    f"{attribute2}_01k,classifiers_all/.1k/{attribute2}_01k_e40_bs100_lr0.0001_tv0.8/best_model.pt,{attribute2},2,multi_layer",
    f"{attribute2}_05k,classifiers_all/.5k/{attribute2}_05k_e40_bs500_lr0.0001_tv0.8/best_model.pt,{attribute2},2,multi_layer",
    f"{attribute2}_1k,classifiers_all/1k/{attribute2}_1k_e40_bs512_lr0.0001_tv0.8/best_model.pt,{attribute2},2,multi_layer",
    f"{attribute2}_2k,classifiers_all/2k/{attribute2}_2k_e40_bs512_lr0.0001_tv0.8/best_model.pt,{attribute2},2,multi_layer",
    f"{attribute2}_5k,classifiers_all/5k/{attribute2}_5k_e40_bs256_lr0.0001_tv0.8/best_model.pt,{attribute2},2,multi_layer",
    
    f"{attribute3}_01k,classifiers_all/.1k/{attribute3}_01k_e60_bs100_lr0.0001_tv0.8/best_model.pt,{attribute3},2,multi_layer",
    f"{attribute3}_05k,classifiers_all/.5k/{attribute3}_05k_e60_bs256_lr0.0001_tv0.8/best_model.pt,{attribute3},2,multi_layer",
    f"{attribute3}_1k,classifiers_all/1k/{attribute3}_1k_e60_bs512_lr0.0001_tv0.8/best_model.pt,{attribute3},2,multi_layer",
    f"{attribute3}_2k,classifiers_all/2k/{attribute3}_2k_e30_bs512_lr0.0001_tv0.6/best_model.pt,{attribute3},2,multi_layer",
    f"{attribute3}_5k,classifiers_all/5k/{attribute3}_5k_e30_bs256_lr0.0001_tv0.6/best_model.pt,{attribute3},2,multi_layer", """
    
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser

if __name__ == "__main__":
    main()