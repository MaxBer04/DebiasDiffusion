# Section 6.1.1: Syntactic Filtering Evaluation

This directory contains scripts and data for evaluating the false positive rate of the syntactic filtering mechanism used in the DebiasDiffusion project. The evaluation focuses on the Natural Language Toolkit (NLTK) performance in correctly identifying non-human-related nouns in prompts.

## Contents

- `sample_LAION_dataset.py`: Script for sampling and processing LAION-400m dataset
- `full_nltk_ablation.py`: Main script for conducting the NLTK evaluation (located in `src/sections/section_5.4/`)

## Data

The evaluation uses a list of prompts sampled from the LAION-400m dataset, specifically chosen to not contain human faces. 
For example for the first experiment in section 5.4, this data is located by default at `data/experiments/section_6.1.1/6.1.1_LAION_400_NO_PERSON.json`

## Usage

1. Ensure you have the required data file in place.

2. Run the NLTK evaluation:

```bash
python src/sections/section_5.4/full_nltk_ablation.py --input_file <BASE_DIR>/data/experiments/section_6.1.1/6.1.1_LAION_400_NO_PERSON.json
```

This script will:

- Apply syntactic filtering to the prompts
- Generate images using both Stable Diffusion and DebiasDiffusion
- Compute CLIP similarities
- Save results and analysis


3. For the analysis of generated faces, refer to the scripts in the `section_5.4` directory. These scripts provide comprehensive evaluation of fairness, semantics preservation, and image quality. See the README in `src/sections/section_5.4/` for detailed instructions on running these evaluations.

## Results

The evaluation results, including:

- Filtered prompts
- Generated images
- Analysis results

will be saved in the specified output directory (default: `results/section_6.1.1/`).

## Note

This evaluation is crucial for understanding the effectiveness and limitations of the syntactic filtering mechanism in DebiasDiffusion. The false positive rate obtained here provides insights into how often the system incorrectly identifies non-human-related prompts as requiring debiasing.
For more detailed information about this evaluation and its significance, please refer to `Section 6.1.1` of the associated thesis.