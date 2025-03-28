# U-Net

Implementation of the [U-Net model](https://arxiv.org/abs/1505.04597).

## Model Details

For benchmarking on the Well, we used the following parameters.

| Parameters          | Values |
|---------------------|--------|
| Spatial Filter Size | 3      |
| Initial Dimension   | 48     |
| Block per Stage     | 1      |
| Up/Down Blocks      | 4      |
| Bottleneck Blocks   | 1      |

## Trained Model Versions

Below is the list of checkpoints available for the training of U-Net on different datasets of the Well.

| Dataset | Learning Rate | Epochs | VRMSE |
|---------|---------------|--------|-------|
| [acoustic_scattering (maze)](https://huggingface.co/polymathic-ai/UNET-acoustic_scattering) | 1E-2 | 26 | 0.0395 |
| [active_matter](https://huggingface.co/polymathic-ai/UNET-active_matter) | 5E-3 | 239 | 0.2609 |
| [convective_envelope_rsg](https://huggingface.co/polymathic-ai/UNET-convective_envelope_rsg) | 5E-4 | 19 | 0.0701 |
| [gray_scott_reaction_diffusion](https://huggingface.co/polymathic-ai/UNET-gray_scott_reaction_diffusion) | 1E-2 | 44 | 0.5870 |
| [helmholtz_staircase](https://huggingface.co/polymathic-ai/UNET-helmholtz_staircase) | 1E-3 | 120 | 0.01655 |
| [MHD_64](https://huggingface.co/polymathic-ai/UNET-MHD_64) | 5E-4 | 165 | 0.1988 |
| [planetswe](https://huggingface.co/polymathic-ai/UNET-planetswe) | 1E-2 | 49 | 0.3498 |
| [post_neutron_star_merger](https://huggingface.co/polymathic-ai/UNET-post_neutron_star_merger) | - | - | – |
| [rayleigh_benard](https://huggingface.co/polymathic-ai/UNET-rayleigh_benard) | 1E-4 | 29 | 0.8448 |
| [rayleigh_taylor_instability](https://huggingface.co/polymathic-ai/UNET-rayleigh_taylor_instability) | 5E-4 | 193 | 0.6140 |
| [shear_flow](https://huggingface.co/polymathic-ai/UNET-shear_flow) | 5E-4 | 29 | 0.836 |
| [supernova_explosion_64](https://huggingface.co/polymathic-ai/UNET-supernova_explosion_64) | 5E-4 | 46 | 0.3242 |
| [turbulence_gravity_cooling](https://huggingface.co/polymathic-ai/UNET-turbulence_gravity_cooling) | 1E-3 | 14 | 0.3152 |
| [turbulent_radiative_layer_2D](https://huggingface.co/polymathic-ai/UNET-turbulent_radiative_layer_2D) | 5E-3 | 500 | 0.2394 |
| [viscoelastic_instability](https://huggingface.co/polymathic-ai/UNET-viscoelastic_instability) | 5E-4 | 198 | 0.3147 |

## Loading the model from Hugging Face

To load the U-Net model trained on a dataset of the Well, use the following while assigning `datasetname` to the actual name of the dataset.

```python
from the_well.benchmark.models import UNetClassic

model = UNetClassic.from_pretrained(f"polymathic-ai/UNET-{datasetname}")
```
