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
