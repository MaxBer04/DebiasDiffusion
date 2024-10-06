# Detailed Installation Guide

This guide provides step-by-step instructions for setting up the DebiasDiffusion project environment.

## Prerequisites

- A machine with a CUDA-compatible GPU, following values are recommendations, smaller setup work as well:
  - Minimum 24GB VRAM for inference
  - Recommended 48GB VRAM for training scripts
- Ubuntu 20.04 or later (for Linux users)
- CUDA 12.4 or compatible version
- Python 3.11.2

## Step-by-step Installation

1. Clone the repository:
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

3. (Optional on some systems) Install additional components as well as CUDA:
```
apt-get update && apt-get install -y g++ build-essential texlive-latex-extra dvipng libgl1-mesa-glx cmake
```

4. Create and activate the conda environment (this may take a couple minutes to install the pip dependencies):
  ```
  conda env create -f environment.yaml
  conda activate DebiasDiffusion
  ```

5. Install spaCy model:
```
python -m spacy download en_core_web_sm
```

6. (Optional) Install more modern version of PyTorch (not supported on all systems but may be faster or even necessary on newer systems):
  ```
  conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
  ```

Note: 
- If you encounter any issues with package conflicts, try installing them one by one or consult the official documentation for each package.
- In case you run into the following error during classification of attributes, execute step 6:
```
Applied providers: ['CPUExecutionProvider'], with options: {'CPUExecutionProvider': {}}
model ignore: /root/.insightface/models/buffalo_l/genderage.onnx genderage
<timestamp> [E:onnxruntime:Default, provider_bridge_ort.cc:1992 TryGetProviderInfo_CUDA] /onnxruntime_src/onnxruntime/core/session/provider_bridge_ort.cc:1637 onnxruntime::Provider& onnxruntime::ProviderLibrary::Get() [ONNXRuntimeError] : 1 : FAIL : Failed to load library libonnxruntime_providers_cuda.so with error: libcublasLt.so.12: cannot open shared object file: No such file or directory
```

## Verifying the Installation

To verify that everything is installed correctly, you can run:
```
python -c "import torch; print(torch.cuda.is_available())"
```

This should print `True` if CUDA is available and properly configured.

## Troubleshooting

If you encounter any issues during the installation process, please check the following:

1. Ensure your NVIDIA drivers are up to date and compatible with CUDA 12.4.
2. Make sure you have sufficient disk space for the conda environment and datasets (~500GB in total).
3. If you're having issues with specific packages, try installing them individually and check for any error messages.

For further assistance, please contact the author under [maber133@uni-duesseldorf.de](maber133@uni-duesseldorf.de).