import sys
import os
import matplotlib.pyplot as plt
import numpy as np
from diffusers.schedulers.scheduling_pndm import betas_for_alpha_bar
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = Path(__file__).resolve().parent.parent.parent

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

def plot_snr_alpha_beta(imgs_out_dir, num_inference_steps):
    betas = betas_for_alpha_bar(num_inference_steps).numpy()
    alphas = np.cumprod(1 - betas, axis=0)
    snr = alphas / (1 - alphas)

    # Plot für Alphas und Betas
    fig, ax = plt.subplots(figsize=(6, 4), dpi=300)
    ax.plot(range(1, len(betas)+1), betas, label='Betas', marker='o', markersize=4)
    ax.plot(range(1, len(alphas)+1), alphas, label='Alphas', marker='x', markersize=4)
    ax.set_xlabel('Timestep t')
    ax.set_ylabel('')
    ax.set_title('') # Alphas and Betas over Timesteps
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.invert_xaxis()
    plt.tight_layout()
    
    plt.savefig(os.path.join(imgs_out_dir, 'betas_alphas_plot.png'), bbox_inches='tight')
    plt.savefig(os.path.join(imgs_out_dir, 'betas_alphas_plot.svg'), bbox_inches='tight')
    plt.close()
    
    # Plot für SNR
    fig, ax = plt.subplots(figsize=(6, 4), dpi=300)
    ax.plot(range(1, len(snr)+1), snr, label='SNR', marker='*', color='black', markersize=4)

    log_snr_1 = 0
    log_snr_2 = -2

    ax.axhline(y=10**log_snr_2, color='red', linestyle='--', alpha=0.7)
    ax.axhline(y=10**log_snr_1, color='green', linestyle='--', alpha=0.7)

    gap = 1
    x_min = 1 - gap
    x_max = len(snr) + gap
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(.5*min(snr), 1.5*max(snr))
    
    ax.fill_between([x_min, x_max], 10**log_snr_1, ax.get_ylim()[1], color='green', alpha=0.1)
    ax.fill_between([x_min, x_max], 10**log_snr_2, 10**log_snr_1, color='red', alpha=0.1)
    ax.fill_between([x_min, x_max], ax.get_ylim()[0], 10**log_snr_2, color='blue', alpha=0.1)

    ax.set_xlabel('Timestep t')
    ax.set_ylabel('') #(log-)SNR
    ax.set_title('') #Signal-to-Noise Ratio over Timesteps
    ax.legend(['SNR', r'$\tau_1$', r'$\tau_2$'])
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.invert_xaxis()
    ax.set_yscale('log')

    plt.tight_layout()

    plt.savefig(os.path.join(imgs_out_dir, 'snr.png'), bbox_inches='tight')
    plt.savefig(os.path.join(imgs_out_dir, 'snr.svg'), bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    imgs_out_dir = BASE_DIR / "outputs" / "section_2.4" / "SNR"
    os.makedirs(imgs_out_dir, exist_ok=True)
    plot_snr_alpha_beta(imgs_out_dir, num_inference_steps=50)