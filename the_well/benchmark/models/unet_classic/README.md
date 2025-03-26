---
arxiv: 1505.04597
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
