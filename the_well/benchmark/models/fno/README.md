---
arxiv: 2010.08895
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

# Fourier Neural Operator

Implementation of the [Fourier Neural Operator](https://arxiv.org/abs/2010.08895) provided by [`neuraloperator v0.3.0`](https://neuraloperator.github.io/dev/index.html).

## Model Details

For benchmarking on the Well, we used the following parameters.

| Parameters  | Values |
|-------------|--------|
| Modes       | 16     |
| Blocks      | 4      |
| Hidden Size | 128    |


## Trained Model Versions

Below is the list of checkpoints available for the training of FNO on different datasets of the Well.

| Dataset                                | Best Learning Rate | Epochs | VRMSE  |
|----------------------------------------|--------------------|--------|--------|
| [acoustic_scattering_maze](https://huggingface.co/polymathic-ai/FNO-acoustic_scattering_maze)             | 1E-3               | 27     | 0.5033 |
| [active_matter](https://huggingface.co/polymathic-ai/FNO-active_matter)                                   | 5E-3               | 239    | 0.3157 |
| [convective_envelope_rsg](https://huggingface.co/polymathic-ai/FNO-convective_envelope_rsg)               | 1E-4               | 14     | 0.0224 |
| [gray_scott_reaction_diffusion](https://huggingface.co/polymathic-ai/FNO-gray_scott_reaction_diffusion)   | 1E-3               | 46     | 0.2044 |
| [helmholtz_staircase](https://huggingface.co/polymathic-ai/FNO-helmholtz_staircase)                       | 5E-4               | 132    | 0.00160|
| [MHD_64](https://huggingface.co/polymathic-ai/FNO-MHD_64)                                                 | 5E-3               | 170    | 0.3352 |
| [planetswe](https://huggingface.co/polymathic-ai/FNO-planetswe)                                           | 5E-4               | 49     | 0.0855 |
| [post_neutron_star_merger](https://huggingface.co/polymathic-ai/FNO-post_neutron_star_merger)             | 5E-4               | 104    | 0.4144 |
| [rayleigh_benard](https://huggingface.co/polymathic-ai/FNO-rayleigh_benard)                               | 1E-4               | 32     | 0.6049 |
| [rayleigh_taylor_instability](https://huggingface.co/polymathic-ai/FNO-rayleigh_taylor_instability)       | 5E-3               | 177    | 0.4013 |
| [shear_flow](https://huggingface.co/polymathic-ai/FNO-shear_flow)                                         | 1E-3               | 24     | 0.4450 |
| [supernova_explosion_64](https://huggingface.co/polymathic-ai/FNO-supernova_explosion_64)                 | 1E-4               | 40     | 0.3804 |
| [turbulence_gravity_cooling](https://huggingface.co/polymathic-ai/FNO-turbulence_gravity_cooling)         | 1E-4               | 13     | 0.2381 |
| [turbulent_radiative_layer_2D](https://huggingface.co/polymathic-ai/FNO-turbulent_radiative_layer_2D)     | 5E-3               | 500    | 0.4906 |
| [viscoelastic_instability](https://huggingface.co/polymathic-ai/FNO-viscoelastic_instability)             | 5E-3               | 205    | 0.7195 |

## Loading the model from Hugging Face

To load the FNO model trained on a dataset of the Well, use the following while assigning `datasetname` to the actual name of the dataset.

```python
from the_well.benchmark.models import FNO

model = FNO.from_pretrained(f"polymathic-ai/FNO-{datasetname}")
```
