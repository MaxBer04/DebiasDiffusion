# Results Directory

This directory contains the outputs and results from various experiments. The results are organized by thesis sections to maintain a clear connection between the experiments and the written work.

## Structure

- `section_4.2/`: Results related to prioritizing pretext tasks in diffusion models
- `section_5.1.3/`: Results from attribute switching experiments
- `section_5.4/`: Comprehensive evaluation results, including fairness metrics and image quality assessments
- `section_6.1/`: Ablation studies results for Debias Diffusion and h-space classifiers

## Contents

Each section directory may contain:

1. Plot files (PNG and SVG formats)
2. Evaluation metadata (CSV or JSON files)
3. Summary statistics

## Usage Guidelines

- When running experiments, ensure that results are saved in the appropriate section folder.
- For plots, save both PNG and SVG formats for flexibility in thesis presentation.
- Metadata files should be clearly named and include a timestamp or version number.

## Reproducibility

To reproduce these results:

1. Ensure you have the necessary data in the `data/experiments/` directory.
2. Run the corresponding scripts from the `src/sections/` directory.
3. For detailed instructions on running specific experiments, refer to the README in each `src/sections/` subdirectory.

Note: Some results may require significant computational resources, particularly for large-scale image generation experiments.

## Analyzing Results

- Use the plots and metadata in this directory to compare the performance of different debiasing methods.
- Pay special attention to the fairness metrics (e.g., Fairness Discrepancy) and image quality measures (e.g., FID scores).
- For a comprehensive analysis of these results, refer to the corresponding sections in the thesis document.

For any questions about the results or reproduction of experiments, please refer to the main project README or contact the project maintainer at [maber133@uni-duesseldorf.de](maber133@uni-duesseldorf.de).