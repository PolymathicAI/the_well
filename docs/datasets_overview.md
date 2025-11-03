# Overview of the Dataset Collection

The Well is composed of 16 datasets totaling 15TB of data with individual datasets ranging from 6.9GB to 5.1TB.
The data is provided on uniform grids and sampled at constant time intervals.
Data and associated metadata are stored in self-documenting `HDF5` and metadata `dataset_name.yaml` files. All datasets use a shared data specification described in [api.md#data-format].
These files include all available state variables or spatially varying coefficients associated with a given set of dynamics in `numpy` arrays of shape `(n_traj, n_steps, coord1, coord2, (coord3))` in single precision `fp32`.
We distinguish between scalar (`t0_fields`), vector (`t1_fields`), and tensor-valued fields (`t2_fields`) due to their different transformation properties.
Each file is randomly split into training/testing/validation sets with a respective split of 0.8/0.1/0.1 * `n_traj`.
Details of individual datasets are given in the following table:

| Dataset                       | CS             | Resolution (pixels)         | n_steps | n_traj |
|-------------------------------|----------------|-----------------------------|---------|--------|
| `acoustic_scattering`         | Cartesian 2D   | 256 × 256                   | 100     | 8,000  |
| `active_matter`               | Cartesian 2D   | 256 × 256                   | 81      | 360    |
| `convective_envelope_rsg`     | Spherical      | 256 × 128 × 256             | 100     | 29     |
| `euler_multi_quadrants`       | Cartesian 2D   | 512 × 512                   | 100     | 10,000 |
| `gray_scott_reaction_diffusion`| Cartesian 2D   | 128 × 128                   | 1,001   | 1,200  |
| `helmholtz_staircase`         | Cartesian 2D   | 1,024 × 256                 | 50      | 512    |
| `MHD`                         | Cartesian 3D   | 64³ and 256³                | 100     | 100    |
| `planetswe`                   | Angular        | 256 × 512                   | 1,008   | 120    |
| `post_neutron_star_merger`    | Log-Spherical  | 192 × 128 × 66              | 181     | 8      |
| `rayleigh_benard`             | Cartesian 2D   | 512 × 128                   | 200     | 1,750  |
| `rayleigh_taylor_instability` | Cartesian 3D   | 128 × 128 × 128             | 120     | 45     |
| `shear_flow`                  | Cartesian 2D   | 128 × 256                   | 200     | 1,120  |
| `supernova_explosion`         | Cartesian 3D   | 64³ and 128³                | 59      | 1,000  |
| `turbulence_gravity_cooling`  | Cartesian 3D   | 64 × 64 × 64                | 50      | 2,700  |
| `turbulent_radiative_layer_2D`| Cartesian 2D   | 128 × 384                   | 101     | 90     |
| `turbulent_radiative_layer_3D`| Cartesian 3D   | 128 × 128 × 256             | 101     | 90     |
| `viscoelastic_instability`    | Cartesian 2D   | 512 × 512                   | variable| 260    |


**Table:** *Dataset description, coordinate system (CS), resolution of snapshots, n_steps (number of time-steps per trajectory), and n_traj (total number of trajectories in the dataset).*

| Dataset                    | Size (GB) | Run time (h) | Hardware        | Software                        |
|----------------------------|-----------|--------------|-----------------|---------------------------------|
| `acoustic_discontinuous`   | 157       | 0.25        | 64 C            | Clawpack                    |
| `acoustic_inclusions`      | 283       | 0.25        | 64 C            | Clawpack                     |
| `acoustic_maze`            | 311       | 0.33        | 64 C            | Clawpack                    |
| `active_matter`            | 51.3      | 0.33        | A100 GPU        | Python                          |
| `convective_envelope_rsg`  | 570       | 1460        | 80 C            | Athena++                    |
| `euler`                    | 5170      | 80*         | 160 C*          | ClawPack                   |
| `helmholtz_staircase`      | 52        | 0.11        | 64 C            | Python                          |
| `MHD_256`                  | 4580      | 48          | 64 C            | Fortran MPI                     |
| `MHD_64`                   | 72        | --          | --              | --                              |
| `gray_scott_reaction_diffusion`        | 154       | 33*         | 40 C            | Matlab                          |
| `planetswe`                | 186       | 0.75        | 64 C            | Dedalus                    |
| `post_neutron_star_merger` | 110       | 505*        | 300 C*          | νbhlight                     |
| `rayleigh_benard`          | 358       | 60*         | 768 C*          | Dedalus                      |
| `rayleigh_taylor_instability` | 256   | 65*         | 128 C*          | TurMix3D                     |
| `shear_flow`               | 115       | 5*          | 448 C*          | Dedalus                     |
| `supernova_explosion_128`  | 754       | 4*          | 1040 C*         | ASURA-FDPS                   |
| `supernova_explosion_64`   | 268       | 4*          | 1040 C*         | ASURA-FDPS                   |
| `turbulence_gravity_cooling` | 829    | 577*        | 1040 C*         | ASURA-FDPS                   |
| `turbulent_radiative_layer_2D` | 6.9  | 2*          | 48 C            | Athena++                    |
| `turbulent_radiative_layer_3D` | 745  | 271*        | 128 C           | Athena++                     |
| `viscoelastic_instability` | 66        | 34*         | 64 C            | Dedalus                      |

**Table:** *Information about the different dataset generation. In the running time and hardware columns, * denotes a total for all the runs. Otherwise, these figures are given for running one simulation only. For hardware, C denotes the number of Cores. Computation was performed on nodes equipped with either 2 48-core AMD Genoa or 2 32-core Intel Icelake.*
