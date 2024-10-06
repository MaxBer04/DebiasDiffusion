"""
Attribute Switching Experiment for Diffusion Models

This script implements the attribute switching experiment described in Section 5.1.3 of the thesis.
It computes score differences for different attribute classes across timesteps to determine
optimal switching points for attribute manipulation in the diffusion process.

Usage:
    python src/sections/section_5.1.3/attribute_switching.py [--args]

Arguments:
    --model_id: Hugging Face model ID or path to local model (default: "PalionTech/debias-diffusion-orig")
    --num_runs: Number of runs for each experiment (default: 10)
    --num_inference_steps: Number of inference steps (default: 50)
    --num_images_per_prompt: Number of images per prompt (default: 5)
    --output_dir: Output directory for generated plots (default: "results/section_5.1.3")
    --seed: Random seed for reproducibility (default: 4000)
    --device: Device to run the model on (default: "cuda" if available, else "cpu")
    --show_legend: Show legend on the scores plot (default: False)
    --mark_min_values: Mark minimum values on the scores plot (default: True)

Outputs:
    - Plots showing the differences in score predictions for different attributes
    - Plots showing the normalized differences and optimal switching points
"""

import argparse
import torch
import os
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent.parent.parent
sys.path.append(str(BASE_DIR))

from src.pipelines.testing_diffusion_pipeline import TestingDiffusionPipeline

def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def setup_pipeline(model_id: str, device: str) -> TestingDiffusionPipeline:
    """Set up the diffusion pipeline."""
    pipeline = TestingDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to(device)
    return pipeline

def compute_score_differences(pipe: TestingDiffusionPipeline, args: argparse.Namespace) -> tuple:
    """Compute score differences for attribute pairs."""
    differences = []
    differences_with_beta = []
    values_list_scores = []

    for run in tqdm(range(args.num_runs), desc="Processing run:"):
        seed = args.seed + run
        
        generator = torch.Generator(pipe.device).manual_seed(seed)
        scores_1 = pipe(["A color photo of the face of a male"]*args.num_images_per_prompt, 
                        generator=generator, 
                        num_inference_steps=args.num_inference_steps, 
                        num_images_per_prompt=1, 
                        return_dict=False)[2]
        
        generator = torch.Generator(pipe.device).manual_seed(seed)
        scores_2 = pipe(["A color photo of the face of a female"]*args.num_images_per_prompt, 
                        generator=generator, 
                        num_inference_steps=args.num_inference_steps, 
                        num_images_per_prompt=1, 
                        return_dict=False)[2]
        
        scores_differences = [s1 - s2 for s1, s2 in zip(scores_1[:-1], scores_2[:-1])]
        differences.append(torch.stack(scores_differences))
        
        score_differences_list = []
        timesteps = pipe.scheduler.timesteps[:-1]
        for i, t in enumerate(timesteps):
            beta_t = pipe.scheduler.betas[t]
            score_diff = beta_t * scores_differences[i]
            score_differences_list.append(score_diff)

        score_differences_tensor = torch.stack(score_differences_list)
        differences_with_beta.append(score_differences_tensor)
            
        diff_taus_mean_scores = []
        ts = range(len(timesteps))
        for i in ts:
            to_tau = torch.sum(score_differences_tensor[:len(timesteps) - i], dim=0)
            from_tau = torch.sum(score_differences_tensor[len(timesteps) - i:], dim=0)
            norm = torch.norm(from_tau - to_tau, dim=0)
            mean = torch.mean(norm)
            diff_taus_mean_scores.append(mean.cpu())
        max_temp = max(diff_taus_mean_scores)
        diff_taus_mean_scores = [tensor / max_temp for tensor in diff_taus_mean_scores]
        values_list_scores.append(diff_taus_mean_scores)
    
    D_ts = [torch.norm(t, dim=1).mean(dim=(1, 2, 3)).cpu().numpy() for t in differences]
    D_ts_2 = [torch.norm(t, dim=1).mean(dim=(1, 2, 3)).cpu().numpy() for t in differences_with_beta]

    return D_ts, D_ts_2, values_list_scores

def plot_results(D_ts: list, D_ts_2: list, values_list_scores: list, output_dir: Path, args: argparse.Namespace) -> None:
    """Plot the results of the attribute switching experiment."""
    ts = range(1, 51)
    
    plt.figure(figsize=(6, 4))
    for D_t in D_ts:
        plt.plot(ts, D_t[::-1], color='#939393', alpha=0.7, linewidth=1)
    plt.xlabel(r'$t$')
    plt.ylabel(r'$\hat{D}_t$')
    plt.gca().invert_xaxis()
    plt.tight_layout()
    plt.savefig(output_dir / 'D_t.png')
    plt.savefig(output_dir / 'D_t.svg')
    plt.close()

    plt.figure(figsize=(6, 4))
    for D_t in D_ts_2:
        plt.plot(ts, D_t[::-1], color='#939393', alpha=0.7, linewidth=1)
    plt.xlabel(r'$t$')
    plt.ylabel(r'$\beta_t\hat{D}_t$')
    plt.gca().invert_xaxis()
    plt.tight_layout()
    plt.savefig(output_dir / 'D_t_2.png')
    plt.savefig(output_dir / 'D_t_2.svg')
    plt.close()

    plt.figure(figsize=(6, 4))
    min_indices = []
    for i, values in enumerate(values_list_scores):
        line, = plt.plot(ts, values, color='#939393', alpha=0.7, linewidth=1)
        if args.show_legend:
            line.set_label(f'Run {i+1}')
        if args.mark_min_values:
            min_index = np.argmin(values)
            min_indices.append(min_index)
            plt.plot(ts[min_index], values[min_index], 'o', color='#4A4A4A', markersize=2)
    
    if args.mark_min_values:
        mean_min_index = int(np.round(np.mean(min_indices)))
        plt.axvline(x=ts[mean_min_index], color='red', alpha=0.8, linestyle='--', linewidth=1)

    plt.xlabel(r'$\tau$')
    plt.ylabel(r'$\tilde{D}_{\tau} / \tilde{D}_{\tau}^{\text{max}}$')
    plt.gca().invert_xaxis()
    if args.show_legend:
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_dir / 'scores.png')
    plt.savefig(output_dir / 'scores.svg')
    plt.close()

def main(args: argparse.Namespace) -> None:
    """Main function to run the attribute switching experiment."""
    set_seed(args.seed)
    
    print(f"Setting up pipeline...")
    pipeline = setup_pipeline(args.model_id, args.device)
    
    print(f"Computing score differences...")
    D_ts, D_ts_2, values_list_scores = compute_score_differences(pipeline, args)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Plotting results...")
    plot_results(D_ts, D_ts_2, values_list_scores, Path(args.output_dir), args)
    
    print(f"Results saved to {args.output_dir}")

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Attribute switching experiment for diffusion models")
    parser.add_argument("--model_id", type=str, default="PalionTech/debias-diffusion-orig",
                        help="Hugging Face model ID or path to local model")
    parser.add_argument("--num_runs", type=int, default=10,
                        help="Number of runs for each experiment")
    parser.add_argument("--num_inference_steps", type=int, default=50,
                        help="Number of inference steps")
    parser.add_argument("--num_images_per_prompt", type=int, default=5,
                        help="Number of images per prompt")
    parser.add_argument("--output_dir", type=str, default="results/section_5.1.3",
                        help="Output directory for generated plots")
    parser.add_argument("--seed", type=int, default=4000,
                        help="Random seed for reproducibility")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to run the model on")
    parser.add_argument("--show_legend", action="store_true", default=False,
                        help="Show legend on the scores plot")
    parser.add_argument("--mark_min_values", action="store_true", default=True,
                        help="Mark minimum values on the scores plot")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args)