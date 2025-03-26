---
arxiv: 2310.00120
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

# Tensorized Fourier Neural Operator

Implementation of the [Tensorized Fourier Neural Operator](https://arxiv.org/abs/2310.00120) provided by [`neuraloperator v0.3.0`](https://neuraloperator.github.io/dev/index.html).

## Model Details

For benchmarking on the Well, we used the following parameters.

| Parameters | Values |
|------------|--------|
| Modes      | 16     |
| Blocks     | 4      |
| Hidden Size| 128    |

## Trained Model Versions

Below is the list of checkpoints available for the training of TFNO on different datasets of the Well.

| Dataset | Learning Rate | Epoch | VRMSE |
|---------|----------------|-------|-------|
| [acoustic_scattering_maze](https://huggingface.co/polymathic-ai/TFNO-acoustic_scattering) | 1E-3 | 27 | 0.5034 |
| [active_matter](https://huggingface.co/polymathic-ai/TFNO-active_matter) | 1E-3 | 243 | 0.3342 |
| [convective_envelope_rsg](https://huggingface.co/polymathic-ai/TFNO-convective_envelope_rsg) | 1E-3 | 13 | 0.0195 |
| [gray_scott_reaction_diffusion](https://huggingface.co/polymathic-ai/TFNO-gray_scott_reaction_diffusion) | 5E-3 | 45 | 0.1784 |
| [helmholtz_staircase](https://huggingface.co/polymathic-ai/TFNO-helmholtz_staircase) | 5E-4 | 131 | 0.00031 |
| [MHD_64](https://huggingface.co/polymathic-ai/TFNO-MHD_64) | 1E-3 | 155 | 0.3347 |
| [planetswe](https://huggingface.co/polymathic-ai/TFNO-planetswe) | 5E-4 | 49 | 0.1061 |
| [post_neutron_star_merger](https://huggingface.co/polymathic-ai/TFNO-post_neutron_star_merger) | 5E-4 | 99 | 0.4064 |
| [rayleigh_benard](https://huggingface.co/polymathic-ai/TFNO-rayleigh_benard) | 1E-4 | 31 | 0.8568 |
| [rayleigh_taylor_instability](https://huggingface.co/polymathic-ai/TFNO-rayleigh_taylor_instability) | 1E-4 | 175 | 0.2251 |
| [shear_flow](https://huggingface.co/polymathic-ai/TFNO-shear_flow) | 1E-3 | 24 | 0.3626 |
| [supernova_explosion_64](https://huggingface.co/polymathic-ai/TFNO-supernova_explosion_64) | 1E-4 | 35 | 0.3645 |
| [turbulence_gravity_cooling](https://huggingface.co/polymathic-ai/TFNO-turbulence_gravity_cooling) | 5E-4 | 10 | 0.2789 |
| [turbulent_radiative_layer_2D](https://huggingface.co/polymathic-ai/TFNO-turbulent_radiative_layer_2D) | 1E-3 | 500 | 0.4938 |
| [viscoelastic_instability](https://huggingface.co/polymathic-ai/TFNO-viscoelastic_instability) | 5E-3 | 199 | 0.7021 |

## Loading the model from Hugging Face

To load the TFNO model trained on a dataset of the Well, use the following while replacing `<datasetname>` by the actual name of the dataset.

```python
from the_well.benchmark.models import TFNO

model = TFNO.from_pretrained("polymathic-ai/TFNO-<datasetname>")
```
