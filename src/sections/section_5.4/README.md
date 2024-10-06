# Section 5.4: Evaluation Scripts

This directory contains scripts for evaluating the DebiasDiffusion method and other debiasing approaches. The evaluation covers fairness, semantics preservation, and image quality metrics.

## Contents

- `fairness/`: Scripts for fairness evaluation
- `semantics_preservation/`: Scripts for semantics preservation evaluation
- `quality/`: Scripts for image quality evaluation
- `NLTK/`: Scripts for syntactic filtering ablation

## Quick Start

### Fairness Evaluation

1. Classify attributes:
  ```
  python fairness/analyze_attributes.py --dataset_dir path/to/dataset
  ```

2. Compute FD metric or create plots:
  ```
  python fairness/analyze_FD.py --input_dir path/to/attribute_classifications
  python fairness/create_areaplots.py --data_dir path/to/attribute_classifications
  ```

### Semantics Preservation Evaluation

1. Analyze CLIP scores:
  ```
  python semantics_preservation/analyze_CLIP.py --datasets_dir path/to/datasets
  ```

2. Analyze LPIPS scores:
  ```
  python semantics_preservation/analyze_LPIPS.py --datasets_dir path/to/datasets
  ```

### Syntactic Filtering Ablation

Run the NLTK ablation script:
  ```
  python NLTK/full_nltk_ablation.py --input_file path/to/prompts.json
  ```

### Quality Metrics Evaluation

For detailed instructions on evaluating image quality using FD-infinity and KD metrics, please refer to the [Quality Metrics Evaluation Guide](../../.././docs/quality_metrics_evaluation.md).

## Notes

- Ensure you have the required dependencies installed (see main project README).
- Adjust paths and arguments as needed for your specific setup.

For further assistance, please contact the author under [maber133@uni-duesseldorf.de](maber133@uni-duesseldorf.de).