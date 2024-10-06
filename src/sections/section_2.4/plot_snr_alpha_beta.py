"""
Plot Signal-to-Noise Ratio (SNR) and Variance Schedules for Diffusion Models

This script generates and saves plots of the variance schedule (beta), signal schedule (alpha),
and the corresponding Signal-to-Noise Ratio (SNR) for diffusion models. These plots are crucial
for understanding the noise injection process in diffusion models, as described in Section 2.4
of the associated thesis.

Usage:
    python src/sections/section_2.4/plot_snr_alpha_beta.py

Outputs:
    - betas_alphas_plot.png/.svg: Plot of beta and alpha schedules
    - snr.png/.svg: Plot of the Signal-to-Noise Ratio

The script uses a cosine variance schedule as defined in Dhariwal and Nichol (2021).
"""

import sys
import os
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from diffusers.schedulers.scheduling_pndm import betas_for_alpha_bar

# Add project root to Python path
SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent.parent.parent
sys.path.append(str(BASE_DIR))

# Configure matplotlib for high-quality plots
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 18,
    "axes.labelsize": 20,
    "axes.titlesize": 22,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 16,
    "figure.titlesize": 24
})

def compute_schedules(num_inference_steps: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute beta, alpha, and SNR schedules for the diffusion process.

    Args:
        num_inference_steps (int): Number of inference steps in the diffusion process.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: Beta schedule, alpha schedule, and SNR.
    """
    betas = betas_for_alpha_bar(num_inference_steps).numpy()
    alphas = np.cumprod(1 - betas, axis=0)
    snr = alphas / (1 - alphas)
    return betas, alphas, snr

def plot_betas_alphas(betas: np.ndarray, alphas: np.ndarray, save_path: Path) -> None:
    """
    Plot and save the beta and alpha schedules.

    Args:
        betas (np.ndarray): Beta schedule.
        alphas (np.ndarray): Alpha schedule.
        save_path (Path): Directory to save the plot.
    """
    fig, ax = plt.subplots(figsize=(6, 4), dpi=300)
    ax.plot(range(1, len(betas)+1), betas, label='Betas', marker='o', markersize=4)
    ax.plot(range(1, len(alphas)+1), alphas, label='Alphas', marker='x', markersize=4)
    ax.set_xlabel('Timestep t')
    ax.set_ylabel('')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.invert_xaxis()
    plt.tight_layout()
    
    plt.savefig(save_path / 'betas_alphas_plot.png', bbox_inches='tight')
    plt.savefig(save_path / 'betas_alphas_plot.svg', bbox_inches='tight')
    plt.close()

def plot_snr(snr: np.ndarray, save_path: Path) -> None:
    """
    Plot and save the Signal-to-Noise Ratio (SNR).

    Args:
        snr (np.ndarray): Signal-to-Noise Ratio.
        save_path (Path): Directory to save the plot.
    """
    fig, ax = plt.subplots(figsize=(6, 4), dpi=300)
    ax.plot(range(1, len(snr)+1), snr, label='SNR', marker='*', color='black', markersize=4)

    log_snr_1, log_snr_2 = 0, -2
    ax.axhline(y=10**log_snr_2, color='red', linestyle='--', alpha=0.7)
    ax.axhline(y=10**log_snr_1, color='green', linestyle='--', alpha=0.7)

    gap = 1
    x_min, x_max = 1 - gap, len(snr) + gap
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(.5*min(snr), 1.5*max(snr))
    
    ax.fill_between([x_min, x_max], 10**log_snr_1, ax.get_ylim()[1], color='green', alpha=0.1)
    ax.fill_between([x_min, x_max], 10**log_snr_2, 10**log_snr_1, color='red', alpha=0.1)
    ax.fill_between([x_min, x_max], ax.get_ylim()[0], 10**log_snr_2, color='blue', alpha=0.1)

    ax.set_xlabel('Timestep t')
    ax.set_ylabel('log(SNR)')
    ax.legend(['SNR', r'$\tau_1$', r'$\tau_2$'])
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.invert_xaxis()
    ax.set_yscale('log')

    plt.tight_layout()
    plt.savefig(save_path / 'snr.png', bbox_inches='tight')
    plt.savefig(save_path / 'snr.svg', bbox_inches='tight')
    plt.close()

def plot_snr_alpha_beta(imgs_out_dir: Path, num_inference_steps: int = 50) -> None:
    """
    Generate and save plots for beta, alpha schedules and SNR.

    Args:
        imgs_out_dir (Path): Directory to save the output images.
        num_inference_steps (int): Number of inference steps. Defaults to 50.
    """
    imgs_out_dir.mkdir(parents=True, exist_ok=True)
    
    betas, alphas, snr = compute_schedules(num_inference_steps)
    plot_betas_alphas(betas, alphas, imgs_out_dir)
    plot_snr(snr, imgs_out_dir)

if __name__ == "__main__":
    output_dir = BASE_DIR / "results" / "section_2.4" / "SNR"
    plot_snr_alpha_beta(output_dir)
    print(f"Plots saved to {output_dir}")