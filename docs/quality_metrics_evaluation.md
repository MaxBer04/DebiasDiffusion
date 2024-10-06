# Quality Metrics Evaluation Guide

This guide provides detailed instructions for evaluating image quality using FD-infinity and KD metrics in the DebiasDiffusion project.

## Prerequisites

- Completed generation of datasets using DebiasDiffusion methods
- Python 3.10 or later
- Conda package manager

## Setup

1. Create and activate a new conda environment:
  ```
  conda create --name dgm-eval pip python==3.10
  conda activate dgm-eval
  ```

2. Clone and install the dgm-eval library:
```
git clone https://github.com/layer6ai-labs/dgm-eval.git
cd dgm-eval
pip install -e .
```

## Prepare Datasets

1. Restructure generated datasets:
  ```
  python src/sections/section_5.4/quality/sample_custom_dataset.py \
  --input_dir BASE_DIR/data/experiments/section_5.4.1/5.4.1_datasets \ 
  --output_dir BASE_DIR/data/experiments/section_5.4.1/restructured_datasets \
  --dataset_type "occupation"
  ```

2. Create FairFace reference datasets:
  ```
  python src/sections/section_5.4/quality/sample_fairface_dataset.py \
  --output_dir BASE_DIR/data/experiments/section_5.4.1/fairface_datasets/DD_rag \
  --num_samples 9839
  ```

Note: Run this command for each dataset, adjusting the output directory and number of samples accordingly.

## Evaluate Metrics

Run the following command for each dataset pair:
  ```
  python -m dgm_eval path/to/fairface_dataset path/to/generated_dataset \
  --device cuda \
  --model dinov2 \
  --metrics fd-infinity kd
  ```

Example for Debias Diffusion (DD) with gender, race, and age debiasing:
```
python -m dgm_eval \
BASE_DIR/data/experiments/section_5.4.1/fairface_datasets/DD_rag \
BASE_DIR/data/experiments/section_5.4.1/restructured_datasets/DD_rag \ 
--device cuda \
--model dinov2 \
--metrics fd-infinity kd
```

## Tips

- The number of images in a generated dataset can be found in the corresponding `metadata.csv` file.
- Use the `--gender False`, `--race False`, or `--age False` flags when sampling FairFace datasets to exclude specific attributes.
- To specify a target distribution, use the `--target_dist` flag with a path to an attribute classification CSV file.

## Interpreting Results

The dgm-eval library will output FD-infinity and KD scores in the console. Lower scores indicate better quality and similarity to the reference dataset.

## Returning to DebiasDiffusion Environment

After completing the evaluation, return to the main project environment:
```
conda deactivate
conda activate DebiasDiffusion
```

For any issues or questions, please refer to the [dgm-eval documentation](https://github.com/layer6ai-labs/dgm-eval) or contact the author of this thesis under [maber133@uni-duesseldorf.de](maber133@uni-duesseldorf.de).