# Section 5.1.3: Attribute Switching

This section contains the implementation of the attribute switching experiment described in Section 5.1.3 of the thesis. The experiment demonstrates how changing text conditioning at specific timesteps during the image generation process can alter specific attributes of the generated images.

## Contents

- `attribute_switching.py`: Implements the attribute switching experiment for diffusion models.

## Usage

To run the attribute switching experiment:

```bash
python src/sections/section_5.1.3/attribute_switching.py [--args]
```

### Key arguments:

- `--prompts`: List of prompts to generate images from
- `--num_images_per_prompt`: Number of images to generate per prompt (default: 32)
- `--tau_gender`, `--tau_age`, `--tau_race`: Timesteps to switch respective attributes

Use `--help` to see all available arguments.

## Output

Results are saved in `results/section_5.1.3/attribute_switching/`:

- Individual images for each prompt and attribute combination
- Image grids showing the attribute switching results for each prompt

## Note

This experiment provides visual evidence of the model's ability to alter specific attributes during the generation process. It demonstrates the flexibility of diffusion models in controlled image generation, as discussed in the thesis.