# DebiasDiffusion: Mitigating Biases in Text-to-Image Diffusion Models

This repository contains the implementation and evaluation scripts for the DebiasDiffusion method, a novel approach to mitigate biases in text-to-image diffusion models. The work was conducted as part of a master's thesis project.

## Project Structure

- `data/`: Contains raw and processed data used in the experiments.
- `src/`: Source code for the DebiasDiffusion method and related utilities.
- `tools/`: Utility scripts for data management and experiment execution.
- `docs/`: Additional documentation.
- `results/`: Output directory for experimental results.

## Quick Start

### Installation

1. Clone the repository.
2. Create and activate the conda environment:
  ```
  conda env create -f environment.yaml
  conda activate DebiasDiffusion
  ```
3. Install spaCy model:
```
python -m spacy download en_core_web_sm
```
For detailed installation instructions, see [this more elaborate guide](docs/installation.md).

### Testing the different pipelines

To test the different pipelines (see Section 5.1 in the thesis), we refer to [this starting guide](exploratory/README.md) and specifically the `plot_example_images.py` script.

### Downloading Data

To download the necessary data for experiments and models:
```
python tools/public_data_downloader.py all --output_dir ./data
```

For model weights only:
```
python tools/public_data_downloader.py model_data --output_dir ./data
```

For detailed data download instructions and how to download data for single experiments, see [docs/data_download.md](docs/data_download.md).

## Running Experiments

Detailed instructions for running specific experiments can be found in the README files within each section folder under `src/sections/`.

## Results

The `results/` directory will contain the outputs from the various scripts provided. For a detailed analysis of these results, please refer to the accompanying thesis.

## Acknowledgements

This work is based on the Stable Diffusion model by Stability AI and uses the HuggingFace Diffusers library. We acknowledge their contributions to the field of text-to-image generation.

For further assistance, please contact the author at [maber133@uni-duesseldorf.de](maber133@uni-duesseldorf.de).