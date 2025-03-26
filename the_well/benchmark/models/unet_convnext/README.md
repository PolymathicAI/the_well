---
arxiv: 2201.03545
datasets:
- polymathi-ai/acoustic_scattering_discontinuous
- polymathi-ai/acoustic_scattering_inclusions
- polymathi-ai/acoustic_scattering_maze
- polymathi-ai/active_matter
- polymathi-ai/convective_envelope_rsg
- polymathi-ai/gray_scott_reaction_diffusion
- polymathi-ai/helmholtz_staircase
- polymathi-ai/MHD_64
- polymathi-ai/planetswe
- polymathi-ai/post_neutron_star_merger
- polymathi-ai/rayleigh_benard
- polymathi-ai/rayleigh_taylor_instability
- polymathi-ai/shear_flow
- polymathi-ai/supernova_explosion_64
- polymathi-ai/turbulence_gravity_cooling
- polymathi-ai/turbulent_radiative_layer_2D
- polymathi-ai/viscoelastic_instability
tags:
- physics

---

# Benchmarking Models on the Well

[The Well](https://github.com/PolymathicAI/the_well) is a 15TB dataset collection of physics simulations. This model is part of the models that have been benchmarked on the Well.


The models have been trained for a fixed time of 12 hours or up to 500 epochs, whichever happens first. The training was performed on a NVIDIA H100 96GB GPU.
In the time dimension, the context length was set to 4. The batch size was set to maximize the memory usage. We experiment with 5 different learning rates for each model on each dataset.
We use the model performing best on the validation set to report test set results.

The reported results are here to provide a simple baseline. **They should not be considered as state-of-the-art**. We hope that the community will build upon these results to develop better architectures for PDE surrogate modeling.

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

To load the CNextU-Net model trained on a dataset of the Well, use the following while replacing `<datasetname>` by the actual name of the dataset.

```python
from the_well.benchmark.models import UNetConvNext

model = UNetConvNext.from_pretrained("polymathic-ai/CNextU-Net-<datasetname>")
```
