# Section 6.1.2: Batch Size and Tau Bias Analysis

This directory contains scripts for analyzing the impact of batch size and tau_bias on the DebiasDiffusion model's performance, as discussed in Section 6.1.2 of the associated thesis.

## Contents

- `correlation_data_generation.py`: Generates correlation data between estimated and real biases
- `correlation_plots.py`: Creates plots visualizing the correlation data
- `plot_bs_tau.py`: Analyzes and plots the relationship between semantic preservation (LPIPS) and fairness (FD)

## Usage

### Generating Correlation Data

```bash
python correlation_data_generation.py --num_images 128 \
                                      --occupations_file <BASE_DIR>/data/experiments/section_6.1.2/6.1.2_occupations-300-400.json \
                                      --output_dir <BASE_DIR>/results/section_6.1.2/correlation_data
```

### Creating Correlation Plots

```
python correlation_plots.py --input_file <BASE_DIR>/results/section_6.1.2/correlation_data/correlation_data.json \
                            --output_dir <BASE_DIR>/results/section_6.1.2/correlation_plots
```

### Analyzing Batch Size and Tau Bias

```
python plot_bs_tau.py --output_dir <BASE_DIR>/results/section_6.1.2/batch_size_tau_analysis
```

## Data

The scripts use data from the following locations:

- Original datasets: `<BASE_DIR>/data/experiments/section_6.1.2/6.1.2_datasets`
- Pre-computed results: `<BASE_DIR>/data/experiments/section_6.1.2`

To generate new results or use different datasets:

1. Use scripts in `src/sections/section_5.4/` to evaluate LPIPS and FD metrics on new datasets
2. Update the `lpips` and `bias` dictionaries in `plot_bs_tau.py` with the new values

## Results

The analysis results, including:

- Correlation data (JSON)
- Correlation plots (PNG, SVG)
- Batch size and tau_bias analysis plots (PNG, SVG)

will be saved in the specified output directories (default: `<BASE_DIR>/results/section_6.1.2/`).

## Note

For more detailed information about this analysis and its significance, please refer to Section 6.1.2 of the associated thesis.

For further assistance or questions, please contact the author at [maber133@uni-duesseldorf.de](maber133@uni-duesseldorf.de)