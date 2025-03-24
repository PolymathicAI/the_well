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

# Fourier Neural Operator

Implementation of the [Tensorized Fourier Neural Operator](https://arxiv.org/abs/2310.00120) provided by [`neuraloperator v0.3.0`](https://neuraloperator.github.io/dev/index.html).

## Model Details

For benchmarking on the Well, we used the following parameters.

| Parameters | Values |
|------------|--------|
| Modes      | 16     |
| Blocks     | 4      |
| Hidden Size| 128    |

## Loading the model from Hugging Face
