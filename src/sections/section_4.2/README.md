# Section 4.2: Image Predictions and Re-Diffusion

This directory contains scripts for visualizing the image generation process of diffusion models and conducting re-diffusion experiments, as discussed in Section 4.2 of the thesis.

## Contents

- `image_predictions.py`: Generates and saves image predictions at various timesteps during the diffusion process.
- `re_diffusion.py`: Implements the re-diffusion experiment to demonstrate the stability of the generative process.

## Usage

### Image Predictions

To generate image predictions:

```bash
python src/sections/section_4.2/image_predictions.py [--args]
```

### Re-Diffusion Experiment

To run the re-diffusion experiment:
```
python src/sections/section_4.3/re_diffusion.py [--args]
```

Use `--help` with each script to see all available arguments.

## Output

- Image Predictions: Results are saved in `results/section_4.2/image_predictions/`
- Re-Diffusion: Results are saved in `results/section_4.3/re_diffusion/`

Both scripts generate individual images for each step and image grids showing the entire process.

## Note

These scripts are mildly computationally intensive and but require GPU resources to run in considerable time.