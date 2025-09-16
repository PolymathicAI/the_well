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
| [acoustic_scattering_maze](https://huggingface.co/polymathic-ai/TFNO-acoustic_scattering_maze) | 1E-3 | 27 | 0.5034 |
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
