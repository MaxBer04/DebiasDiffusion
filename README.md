# DebiasDiffusion: Mitigating Biases in Text-to-Image Diffusion Models

This repository contains the implementation and evaluation scripts for the DebiasDiffusion method, a novel approach to mitigate biases in text-to-image diffusion models. The work was conducted as part of a master's thesis project.

## Project Structure

- `data/`: Contains raw and processed data used in the experiments.
- `src/`: Source code for the DebiasDiffusion method and related utilities.
- `scripts/`: Utility scripts for data management and experiment execution.
- `tests/`: Unit tests for the codebase.
- `docs/`: Additional documentation.
- `results/`: Output directory for experimental results.
- `configs/`: Configuration files for experiments.

## Installation

### Prerequisites

- A machine with a CUDA-compatible GPU
  - Minimum 24GB VRAM for inference
  - Recommended 48GB VRAM for training scripts
- Ubuntu 20.04 or later (for Linux users)
- CUDA 12.4 or compatible version
- Python 3.11.2

### Quick Install

1. Clone the repository
2. Install Miniconda (if not already installed)
3. Create and activate the conda environment
4. Install additional dependencies
5. Install PyTorch with CUDA support


### Step-by-step Installation

1. Clone this repository:
  ```
  git clone https://github.com/yourusername/DebiasDiffusion.git
  cd DebiasDiffusion
  ```

2. Install Miniconda (if not already installed):
  ```
  mkdir -p ~/miniconda3
  wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
  bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
  rm -rf ~/miniconda3/miniconda.sh
  ~/miniconda3/bin/conda init bash
  ```
    Close and reopen your terminal after this step.

3. Create and activate the conda environment:
  ```
  conda env create -f environment.yaml
  conda activate DebiasDiffusion
  ```

4. Install additional dependencies:
  ```
  sudo apt-get update && sudo apt-get install -y g++ build-essential texlive-latex-extra dvipng libgl1-mesa-glx
  conda install -c conda-forge dlib
  pip install -r requirements.txt
  python -m spacy download en_core_web_sm
  ```

5. Install PyTorch with CUDA support:
  ```
  conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
  ```

Note: If you encounter any issues with package conflicts, try installing them one by one or consult the official documentation for each package.

## Usage

This project provides scripts to reproduce the results from the complete thesis. Readers may either download pre-processed data or generate new data for the experiments.

### Downloading Pre-processed Data

To download the pre-processed data used in the experiments:
```
python scripts/s3_utils.py download --dataset_type all
```

### Running Experiments

Detailed instructions for running specific experiments can be found in the README files within each section folder under `src/sections/`.

## Results

The `results/` directory will contain the outputs from the various scripts provided. For a detailed analysis of these results, please refer to the accompanying thesis.

## Acknowledgements

This work is based on the Stable Diffusion model by Stability AI and uses the HuggingFace Diffusers library. We acknowledge their contributions to the field of text-to-image generation.