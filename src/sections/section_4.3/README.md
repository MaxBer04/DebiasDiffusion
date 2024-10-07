# Section 4.3: H-Space Classifier Training

This directory contains scripts for training h-space classifiers used in the DebiasDiffusion project. These classifiers are crucial for estimating attribute probabilities during the image generation process.

## Contents

- `classifier_train_dataset_classifications.py`: Script for training h-space classifiers on various datasets

## Quick Start

To train h-space classifiers:

```bash
python classifier_train_dataset_classifications.py --dataset_path <BASE_DIR>/data/experiments/section_4.3/h_space_data/dataset_5k.pt \
                                                   --output_path <BASE_DIR>/results/section_4.3/h_space_classifiers \
                                                   --attributes gender race age \
                                                   --dataset_sizes 5k \
                                                   --methods self_labeled cls_labeled cls_labeled_oh
```

## Training Process

The script supports training on three types of datasets:
1. Self-labeled
2. Classifier-labeled
3. Classifier-labeled with one-hot encoding

You can specify which attributes to train (gender, race, age), dataset sizes, and training methods using the appropriate command-line arguments.

## Adjusting Training Parameters

You can adjust various training parameters:

- `--batch_size`: Batch size for training (default: 256)
- `--epochs`: Number of training epochs (default: 100)
- `--lr`: Learning rate (default: 1e-4)
- `--save_interval`: Interval for saving model checkpoints (default: 3)

## Using Weights & Biases (wandb)

This script uses wandb for experiment tracking and visualization. To use wandb:

1. Install wandb:
   ```
   pip install wandb
   ```

2. Log in to your wandb account:
   ```
   wandb login
   ```

3. Set the wandb project and run name in the script arguments:
   ```bash
   python classifier_train_dataset_classifications.py --wandb_project "h_space_classifiers" --wandb_name "experiment_1"
   ```

4. View your results on the wandb dashboard:
   - Go to https://wandb.ai/
   - Navigate to your project
   - You'll see real-time updates of your training progress, including loss curves, accuracy plots, and other metrics

## Results

The training script will save:

- Trained model checkpoints
- Best model weights
- Training logs
- Accuracy plots
- Confusion matrices
- Classification reports

in the specified output directory.

## Note

Training h-space classifiers is a crucial step in improving the performance of the DebiasDiffusion method. Experiment with different dataset sizes, training methods, and hyperparameters to find the optimal configuration for your use case.

For more detailed information about the h-space classifiers and their role in the DebiasDiffusion method, please refer to Section 4.3 of the associated thesis.

For further assistance, please contact the author at [maber133@uni-duesseldorf.de](maber133@uni-duesseldorf.de).
