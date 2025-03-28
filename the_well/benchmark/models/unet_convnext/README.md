# CNextU-Net

Implementation of the [U-Net model](https://arxiv.org/abs/1505.04597) using [ConvNext blocks](https://arxiv.org/abs/2201.03545).

## Model Details

For benchmarking on the Well, we used the following parameters.

| Parameters          | Values |
|---------------------|--------|
| Spatial Filter Size | 7      |
| Initial Dimension   | 42     |
| Block per Stage     | 2      |
| Up/Down Blocks      | 4      |
| Bottleneck Blocks   | 1      |


## Trained Model Versions

Below is the list of checkpoints available for the training of CNextU-Net on different datasets of the Well.

| Dataset | Learning Rate | Epoch | VRMSE |
|---------|---------------|-------|-------|
| [acoustic_scattering_maze](https://huggingface.co/polymathic-ai/CNextU-Net-acoustic_scattering) | 1E-3 | 10 | 0.0196 |
| [active_matter](https://huggingface.co/polymathic-ai/CNextU-Net-active_matter) | 5E-3 | 156 | 0.0953 |
| [convective_envelope_rsg](https://huggingface.co/polymathic-ai/CNextU-Net-convective_envelope_rsg) | 1E-4 | 5 | 0.0663 |
| [gray_scott_reaction_diffusion](https://huggingface.co/polymathic-ai/CNextU-Net-gray_scott_reaction_diffusion) | 1E-4 | 15 | 0.3596 |
| [helmholtz_staircase](https://huggingface.co/polymathic-ai/CNextU-Net-helmholtz_staircase) | 5E-4 | 47 | 0.00146 |
| [MHD_64](https://huggingface.co/polymathic-ai/CNextU-Net-MHD_64) | 5E-3 | 59 | 0.1487 |
| [planetswe](https://huggingface.co/polymathic-ai/CNextU-Net-planetswe) | 1E-2 | 18 | 0.3268 |
| [post_neutron_star_merger](https://huggingface.co/polymathic-ai/CNextU-Net-post_neutron_star_merger) | - | - | - |
| [rayleigh_benard](https://huggingface.co/polymathic-ai/CNextU-Net-rayleigh_benard) | 5E-4 | 12 | 0.4807 |
| [rayleigh_taylor_instability](https://huggingface.co/polymathic-ai/CNextU-Net-rayleigh_taylor_instability) | 5E-3 | 56 | 0.3771 |
| [shear_flow](https://huggingface.co/polymathic-ai/CNextU-Net-shear_flow) | 5E-4 | 9 | 0.3972 |
| [supernova_explosion_64](https://huggingface.co/polymathic-ai/CNextU-Net-supernova_explosion_64) | 5E-4 | 13 | 0.2801 |
| [turbulence_gravity_cooling](https://huggingface.co/polymathic-ai/CNextU-Net-turbulence_gravity_cooling) | 1E-3 | 3 | 0.2093 |
| [turbulent_radiative_layer_2D](https://huggingface.co/polymathic-ai/CNextU-Net-turbulent_radiative_layer_2D) | 5E-3 | 495 | 0.1247 |
| [viscoelastic_instability](https://huggingface.co/polymathic-ai/CNextU-Net-viscoelastic_instability) | 5E-4 | 114 | 0.1966 |

## Loading the model from Hugging Face

To load the CNextU-Net model trained on a dataset of the Well, use the following while assigning `datasetname` to the actual name of the dataset.

```python
from the_well.benchmark.models import UNetConvNext

model = UNetConvNext.from_pretrained(f"polymathic-ai/CNextU-Net-{datasetname}")
```
