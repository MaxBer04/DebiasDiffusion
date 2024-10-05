# Exploratory

This section contains scripts to accompany the comprehensive evaluation provided in the thesis.

## Scripts

- `generate_comparison_images.py`: Generate comparison images using various text-to-image diffusion models.

### Using generate_comparison_images.py

This script generates images from a set of prompts using different text-to-image diffusion models for comparison purposes.

#### Prerequisites

Ensure you have installed all required dependencies as specified in the project's main README.md file.

#### Usage

Run the script from the project root directory:

```bash
python src/sections/section_5.4/generate_comparison_images.py --models SD FDM DD AS FD \
                                                              --prompts "a photo of a doctor" "a portrait of a CEO" \
                                                              --num_images_per_prompt 4
```

#### Arguments

- `--models`: Types of models to use (SD, FDM, DD, AS, FD)
- `--model_id`: Hugging Face model ID or path to local model (default: "PalionTech/debias-diffusion-orig")
- `--prompts`: List of prompts to generate images from
- `--output_dir`: Output directory for generated images (default: PROJECT_ROOT/results/section_5.4/comparison_images)
- `--num_images_per_prompt`: Number of images to generate per prompt (default: 4)
- `--num_inference_steps`: Number of denoising steps (default: 50)
- `--guidance_scale`: Guidance scale for classifier-free guidance (default: 7.5)
- `--grid_cols`: Number of columns in the output grid (default: 2)
- `--seed`: Random seed for reproducibility (default: 42)
- `--device`: Device to run the model on (default: "cuda" if available, else "cpu")

#### Output

The script will save generated images in the specified output directory, organized by model type and prompt. It will also create image grids for easy comparison.

#### Results

The results of these experiments will be saved in the `results/section_5.4/` directory. Please refer to this directory for output images and any generated metrics or plots.