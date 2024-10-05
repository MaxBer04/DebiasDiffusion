# Data Directory

This directory contains all the data used in the DebiasDiffusion project, which focuses on mitigating biases in text-to-image diffusion models. The data is organized into two main categories:

## model_data/

This directory contains essential data required by the various models and components of DebiasDiffusion.

### Structure:
- `h_space_classifiers/`: Trained h-space classifier models used by Debias Diffusion for bias detection and mitigation.
- `fdm_weights/`: Weights for the Finetuned Diffusion Model (FDM), used in comparative experiments.
- `other_model_data/`: Additional model-related data, such as pre-trained embeddings or auxiliary classifiers.

## experiments/

This directory stores datasets created or used in the experiments described in the thesis. Each subdirectory corresponds to a specific section of the thesis.

### Structure:
- `section_4.2/`: Data for experiments on prioritizing pretext tasks in diffusion models.
- `section_5.1.3/`: Datasets for attribute switching experiments.
- `section_5.4/`: Large-scale datasets for comprehensive fairness and quality evaluations.
- `section_6.1/`: Data for ablation studies on Debias Diffusion components.

### Usage:
- When running experiments, scripts should read input data from these directories.
- New datasets generated during experiments should be saved here in the appropriate section folder.

## Data Management

To handle the large datasets used in this project:

1. Use the `tools/s3_utils.py` script for downloading pre-processed data or uploading generated data.
2. To download all datasets:
  ```
  python tools/s3_utils.py download --dataset_type all
  ```
3. Download inidividual datasets per section:

## Important Notes

- Large datasets are not included in the repository due to size constraints.
- Ensure you have sufficient storage space before downloading or generating datasets. 
  The main dataset, containing all data necessary for the thesis, amounts to roughly ~500GB.
- Some experiments may require specific GPU capabilities, particularly for image generation tasks.

For detailed instructions on running specific experiments and handling their data, refer to the README files in the corresponding `src/sections/` directories.