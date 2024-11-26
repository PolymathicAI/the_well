# Benchmarks

To showcase the dataset and the associated benchmarking library, we provide a set of simple baselines time-boxed to 12 hours on a single NVIDIA H100 to demonstrate the effectiveness of naive approaches on these challenging problems and motivate the development of more sophisticated approaches. These baselines are trained on the forward problem - predicting the next snapshot of a given simulation from a short history of 4 time-steps. The models used here are the Fourier Neural Operator, Tucker-Factorized FNO, U-net and a modernized U-net using ConvNext blocks. The neural operator models are implemented using the <a href="https://neuraloperator.github.io/dev/index.html"> neuraloperator </a> library.

We emphasize that these settings are not selected to explore peak performance of modern machine learning, but rather that they reflect reasonable compute budgets and off-the-shelf choices that might be selected by a domain scientist exploring machine learning for their problems. Therefore we focus on popular models using settings that are either defaults or commonly tuned.

### Test results


| Dataset                          | FNO     | TFNO    | U-net   | CNextU-net       |
|----------------------------------|---------|---------|---------|------------------|
| `acoustic_scattering_maze`       | 0.5062  | 0.5057  | 0.0351  | **0.0153**       |
| `active_matter`                  | 0.3691  | 0.3598  | 0.2489  | **0.1034**       |
| `convective_envelope_rsg`        | **0.0269** | 0.0283  | 0.0555  | 0.0799        |
| `euler_multi_quadrants_periodicBC` | 0.4081  | 0.4163  | 0.1834  | **0.1531**     |
| `gray_scott_reaction_diffusion`  | **0.1365** | 0.3633  | 0.2252  | 0.1761        |
| `helmholtz_staircase`            | **0.00046** | 0.00346 | 0.01931 | 0.02758      |
| `MHD_64`                         | 0.3605  | 0.3561  | 0.1798  | **0.1633**       |
| `planetswe`                      | 0.1727  | **0.0853** | 0.3620 | 0.3724         |
| `post_neutron_star_merger`       | 0.3866  | **0.3793** | -     | -               |
| `rayleigh_benard`                | 0.8395  | **0.6566** | 1.4860 | 0.6699         |
| `rayleigh_taylor_instability` (At = 0.25) | >10     | >10     | >10     | >10     |
| `shear_flow`                     | 0.1567  | **0.1348** | 0.5910 | 0.2037         |
| `supernova_explosion_64`         | 0.3783  | 0.3785  | **0.3063** | 0.3181        |
| `turbulence_gravity_cooling`     | 0.2429  | 0.2673  | 0.6753  | **0.2096**       |
| `turbulent_radiative_layer_2D`   | 0.5001  | 0.5016  | 0.2418  | **0.1956**       |
| `turbulent_radiative_layer_3D`   | 0.5278  | 0.5187  | 0.3728  | **0.3667**       |
| `viscoelastic_instability`       | 0.7212  | 0.7102  | 0.4185  | **0.2499**       |


*Table 1: Model Performance Comparison - VRMSE metrics on test sets (lower is better) for models performing best on the validation set (results below). Best results are shown in bold. VRMSE is scaled such that predicting the mean value of the target field results in a score of 1. <strong>Test set results for models performing best on the validation set.</strong>*


### Validation results

| Dataset                              | FNO     | TFNO    | U-net   | CNextU-net       |
|--------------------------------------|---------|---------|---------|------------------|
| `acoustic_scattering_maze`           | 0.5033  | 0.5034  | 0.0395  | **0.0196**       |
| `active_matter`                      | 0.3157  | 0.3342  | 0.2609  | **0.0953**       |
| `convective_envelope_rsg`            | 0.0224  | **0.0195** | 0.0701  | 0.0663        |
| `euler_multi_quadrants_periodicBC`   | 0.3993  | 0.4110  | 0.2046  | **0.1228**       |
| `gray_scott_reaction_diffusion`      | 0.2044  | **0.1784** | 0.5870  | 0.3596        |
| `helmholtz_staircase`                | 0.00160 | **0.00031** | 0.01655 | 0.00146      |
| `MHD_64`                             | 0.3352  | 0.3347  | 0.1988  | **0.1487**       |
| `planetswe`                          | **0.0855** | 0.1061  | 0.3498  | 0.3268        |
| `post_neutron_star_merger`           | 0.4144  | **0.4064** | -       | -             |
| `rayleigh_benard`                    | 0.6049  | 0.8568  | 0.8448  | **0.4807**       |
| `rayleigh_taylor_instability` (At = 0.25) | 0.4013  | **0.2251** | 0.6140  | 0.3771   |
| `shear_flow`                         | 0.2963  | **0.2087** | 0.5799  | 0.3258        |
| `supernova_explosion_64`             | 0.3804  | 0.3645  | 0.3242  | **0.2801**       |
| `turbulence_gravity_cooling`         | 0.2381  | 0.2789  | 0.3152  | **0.2093**       |
| `turbulent_radiative_layer_2D`       | 0.4906  | 0.4938  | 0.2394  | **0.1247**       |
| `turbulent_radiative_layer_3D`       | 0.5199  | 0.5174  | 0.3635  | **0.3562**       |
| `viscoelastic_instability`           | 0.7195  | 0.7021  | 0.3147  | **0.1966**       |


*Table 2: Dataset and model comparison in VRMSE metric on the validation sets, best result in bold. VRMSE is scaled such that predicting the mean value of the target field results in a score of 1.*

### Rollout loss (6:12)

| Dataset                              | FNO $\phantom{T}$ (6:12) | TFNO (6:12) | U-net (6:12) | CNextU-net (6:12) |
|--------------------------------------|------------|-------------|--------------|--------------------|
| `acoustic_scattering_maze`           | 1.06       | 1.13        | **0.56**     | 0.78               |
| `active_matter`                      | $>$10      | 7.52        | 2.53         | **2.11**           |
| `convective_envelope_rsg`            | **0.28**   | 0.32        | 0.76         | 1.15               |
| `euler_multi_quadrants_periodicBC`   | 1.13       | 1.23        | **1.02**     | 4.98               |
| `gray_scott_reaction_diffusion`      | 0.89       | 1.54        | 0.57         | **0.29**           |
| `helmholtz_staircase`                | **0.002**  | 0.011       | 0.057        | 0.110              |
| `MHD_64`                             | **1.24**   | 1.25        | 1.65         | 1.30               |
| `planetswe`                          | 0.81       | **0.29**    | 1.18         | 0.42               |
| `post_neutron_star_merger`           | 0.76       | **0.70**    | ---          | ---                |
| `rayleigh_benard`                    | $>$10      | $>$10       | $>$10        | $>$10              |
| `rayleigh_taylor_instability`        | $>$10      | **6.72**    | $>$10        | $>$10              |
| `shear_flow`                         | 1.62       | 1.63        | 1.22         | **0.32**           |
| `supernova_explosion_64`             | 2.41       | 1.86        | **0.94**     | 1.12               |
| `turbulence_gravity_cooling`         | 3.55       | 4.49        | 7.14         | **1.30**           |
| `turbulent_radiative_layer_2D`       | 1.79       | 6.01        | 0.66         | **0.54**           |
| `turbulent_radiative_layer_3D`       | 0.81       | $>$10       | 0.95         | **0.77**           |
| `viscoelastic_instability`           | 4.11       | 0.93        | 0.89         | **0.52**           |

### Rollout loss (13:30)

| Dataset                              | FNO (13:30) | TFNO (13:30) | U-net (13:30) | CNextU-net (13:30) |
|--------------------------------------|-------------|--------------|---------------|---------------------|
| `acoustic_scattering_maze`           | 1.72        | 1.23         | <u>0.92</u>   | 1.13               |
| `active_matter`                      | $>$10       | 4.72         | <u>2.62</u>   | 2.71               |
| `convective_envelope_rsg`            | <u>0.47</u> | 0.65         | 2.16          | 1.59               |
| `euler_multi_quadrants_periodicBC`   | <u>1.37</u> | 1.52         | 1.63          | $>$10              |
| `gray_scott_reaction_diffusion`      | $>$10       | $>$10        | $>$10         | <u>7.62</u>        |
| `helmholtz_staircase`                | <u>0.003</u>| 0.019        | 0.097         | 0.194              |
| `MHD_64`                             | <u>1.61</u> | 1.81         | 4.66          | 2.23               |
| `planetswe`                          | 2.96        | 0.55         | 1.92          | <u>0.52</u>        |
| `post_neutron_star_merger`           | 1.05        | <u>1.05</u>  | ---           | ---                |
| `rayleigh_benard`                    | $>$10       | $>$10        | $>$10         | $>$10              |
| `rayleigh_taylor_instability`        | $>$10       | $>$10        | <u>2.84</u>   | 7.43               |
| `shear_flow`                         | $>$10       | $>$10        | $>$10         | 1.91               |
| `supernova_explosion_64`             | $>$10       | $>$10        | <u>1.69</u>   | 4.55               |
| `turbulence_gravity_cooling`         | 5.63        | 6.95         | 4.15          | <u>2.09</u>        |
| `turbulent_radiative_layer_2D`       | 3.54        | $>$10        | 1.04          | <u>1.01</u>        |
| `turbulent_radiative_layer_3D`       | 0.94        | $>$10        | 1.09          | <u>0.86</u>        |
| `viscoelastic_instability`           | ---         | ---          | ---           | ---                |


*Table: Time-Averaged Losses by Window - VRMSE metrics on test sets (lower is better), averaged over time windows (6:12) and (13:30). Best results are shown in bold for (6:12) and underlined for (13:30). VRMSE is scaled such that predicting the mean value of the target field results in a score of 1.*
