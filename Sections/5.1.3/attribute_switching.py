import torch
import os
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from diffusers import StableDiffusionPipeline

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(BASE_DIR))

from pipelines.testing_diffusion_pipeline import TestingDiffusionPipeline

def load_model(model_id):
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    pipeline = TestingDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipeline = pipeline.to(device)
    return pipeline

def create_and_save_plot(tau_values, values_list, type, label, title, filename, output_dir, show_legend=False, save_svg=False, is_normalized=False, mark_min_values=True):
    plt.figure(figsize=(6, 4.5)) 
    plt.rcParams.update({'font.size': 16, 'axes.labelsize': 18, 'axes.titlesize': 20, 'xtick.labelsize': 14, 'ytick.labelsize': 14})

    min_indices = []
    for i, values in enumerate(values_list):
        values = values[::-1]  # Werte umkehren
        line, = plt.plot(tau_values, values, color='#939393', alpha=0.7, linewidth=1) 
        
        if mark_min_values:
            min_index = np.argmin(values)
            min_indices.append(min_index)
            plt.plot(tau_values[min_index], values[min_index], 'o', color='#4A4A4A', markersize=2)
        
        if show_legend:
            line.set_label(f'Run {i+1}')
    
    if mark_min_values:
        mean_min_index = np.round(np.mean(min_indices))
        plt.axvline(x=mean_min_index, color='red', alpha=0.8, linestyle='--', linewidth=1)
    
    if type == 0:
        plt.xlabel(r'$\tau$')
    else:
        plt.xlabel(r'$t$')
    
    if type==1:
        plt.ylabel(r'$\hat{D}_t$')
    elif type==2:
        plt.ylabel(r'$\beta_t\hat{D}_t$')
    elif type==0:
        plt.ylabel(r'$\tilde{D}_{\tau} / \tilde{D}_{\tau}^{\text{max}}$')
    

    # if is_normalized:
    #     plt.ylabel(r'$\frac{' + label + r'}{\max(' + label + r')}$', fontsize=16)
    # else:
    #     plt.ylabel(r'$' + label + r'$', fontsize=16)
    
    plt.title(title, fontsize=18)
    if show_legend:
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)
    
    # X-Achsen-Ticks anpassen
    plt.gca().set_xticks(tau_values[::-1][::10])  
    plt.gca().set_xticklabels(tau_values[::-1][::10])
    
    plt.gca().set_xlim(max(tau_values), min(tau_values))
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Als PNG speichern
    plt.savefig(output_dir / f'{filename}.png', format='png', dpi=300, bbox_inches='tight')
    
    # Als SVG speichern, falls gewünscht
    if save_svg:
        plt.savefig(output_dir / f'{filename}.svg', format='svg', bbox_inches='tight')
    
    plt.close()

def main():
    args = {
        'model_id': "PalionTech/debias-diffusion-orig",
        'num_runs': 10,
        'num_inference_steps': 50,
        'output_dir': BASE_DIR / "outputs" / "section_5.1.3",
        'seed': 4000,
        'save_svg': True,  # Set this to True to save SVGs
        'show_legend': False
    }

    args['output_dir'].mkdir(parents=True, exist_ok=True)

    pipe = load_model(args['model_id'])
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')


    values_list_scores = []
    differences = []
    differences_with_beta = []

    for run in tqdm(range(args['num_runs']), desc="Processing run:"):
        seed = args['seed'] + run
        
        generator = torch.Generator(device.type).manual_seed(seed)
        scores_1 = pipe(["A color photo of the face of a male"]*5, generator=generator, num_inference_steps=args['num_inference_steps'], num_images_per_prompt=1, return_dict=False)[2] 
        
        generator = torch.Generator(device.type).manual_seed(seed)
        scores_2 = pipe(["A color photo of the face of a female"]*5, generator=generator, num_inference_steps=args['num_inference_steps'], num_images_per_prompt=1, return_dict=False)[2]
        
        scores_differences = [s1 - s2 for s1, s2 in zip(scores_1, scores_2)]
        differences.append(torch.stack(scores_differences))
        
        score_differences_list = []
        timesteps = pipe.scheduler.timesteps
        for i, t in enumerate(timesteps):
            beta_t = pipe.scheduler.betas[t]
            alpha_bar_t = pipe.scheduler.alphas_cumprod[t]
            score_diff = beta_t * scores_differences[i] # (1/torch.sqrt(1-alpha_bar_t)) * 
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
    
    create_and_save_plot(ts, values_list_scores, 0, r'\left\|\sum_{t \leq \tau} D_t - \sum_{t \geq \tau} D_t\right\|', '', 'scores', args['output_dir'], args['show_legend'], args['save_svg'], is_normalized=True)
    create_and_save_plot(ts, D_ts, 1, r'\epsilon_{\theta}(z_t, t, c_1) - \epsilon_{\theta}(z_t, t, c_2)', '', 'D_t', args['output_dir'], args['show_legend'], args['save_svg'], mark_min_values=False)
    create_and_save_plot(ts, D_ts_2, 2, r'\beta_t \left(\epsilon_{\theta}(z_t, t, c_1) - \epsilon_{\theta}(z_t, t, c_2)\right)', '', 'D_t_2', args['output_dir'], args['show_legend'], args['save_svg'], mark_min_values=False)

if __name__ == "__main__":
    main()