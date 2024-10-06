# Section 2.4: Diffusion Models

This section contains scripts to generate plots illustrating key concepts of diffusion models, as discussed in Section 2.4 of the thesis.

## Contents

- `plot_snr_alpha_beta.py`: Generates plots for the variance schedule (beta), signal schedule (alpha), and Signal-to-Noise Ratio (SNR).

## Usage

To generate the plots:

```bash
python src/sections/section_2.4/plot_snr_alpha_beta.py
```

## Output
The script generates the following plots in `results/section_2.4/SNR/`:

- `betas_alphas_plot.png/.svg`: Visualization of beta and alpha schedules
- `snr.png/.svg`: Visualization of the Signal-to-Noise Ratio

These plots are crucial for understanding the noise injection process in diffusion models and correspond to the theoretical discussion in the thesis.

## Note

The generated plots use a cosine variance schedule as defined in Dhariwal and Nichol (2021). Adjust the num_inference_steps parameter in the script if needed.