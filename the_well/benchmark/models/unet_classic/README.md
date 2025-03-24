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

## Loading the model from Hugging Face
