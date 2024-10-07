# Section 6.2: h-Space Classifier Evaluation

This directory contains scripts for evaluating the performance of h-space classifiers used in the DebiasDiffusion project. These classifiers play a crucial role in estimating attribute probabilities during the image generation process.

## Contents

- `create_dataset.py`: Script for generating datasets to train and evaluate h-space classifiers
- `evaluate_dataset.py`: Script for evaluating the performance of trained h-space classifiers

## Quick Start

### Creating a Dataset

To create a dataset for h-space classifier evaluation:

```bash
python create_dataset.py --num_images 5000 \
                         --occupations_file <BASE_DIR>/data/experiments/section_6.2/occupations.json \
                         --output_dir <BASE_DIR>/data/experiments/section_6.2/h_space_data
```

### Evaluating h-Space Classifiers

To evaluate the performance of h-space classifiers:
```bash
python evaluate_dataset.py --dataset_path <BASE_DIR>/data/experiments/section_6.2/h_space_data/dataset_5k.pt \
                           --output_path <BASE_DIR>/results/section_6.2/h_space_evaluation
```

## Data

The evaluation uses datasets created from a list of occupations, generating images and collecting corresponding h-vectors. Three types of datasets can be created:

1. Self-labeled
2. Classifier-labeled
3. Classifier-labeled with one-hot encoding

For more details on these dataset types, refer to Section 6.2 of the associated thesis.

## Results

The evaluation results, including:

- Accuracy plots for each attribute (gender, race, age) across all 50 timesteps
- Comparison plots between different training methods
- CSV files with detailed evaluation metrics

will be saved in the specified output directory (default: `<BASE_DIR>/results/section_6.2/h_space_evaluation`).

## Note

This evaluation is crucial for understanding the performance of h-space classifiers at different stages of the image generation process. It provides insights into the effectiveness of various training methods and dataset sizes.

For more detailed information about this evaluation and its significance, please refer to Section 6.2 of the associated thesis.

For further assistance, please contact the author at [maber133@uni-duesseldorf.de](maber133@uni-duesseldorf.de).