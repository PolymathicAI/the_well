# Magnetohydrodynamics (MHD) compressible turbulence

**NOTE:** This dataset is available in two different resolutions $256^3$ for `MHD_256` and $64^3$ for `MHD_64`. The data was first generated at $256^3$ and then downsampled to $64^3$ after anti-aliasing with an ideal low-pass filter. The data is available in both resolutions.

**One line description of the data:** This is an MHD fluid flows in the compressible limit (subsonic, supersonic, sub-Alfvenic, super-Alfvenic).

**Longer description of the data:** An essential component of the solar wind, galaxy formation, and of interstellar medium (ISM) dynamics is magnetohydrodynamic (MHD) turbulence. This dataset consists of isothermal MHD simulations without self-gravity (such as found in the diffuse ISM) initially generated with resolution $256^3$ and then downsampled to $64^3$ after anti-aliasing with an ideal low-pass filter. This dataset is the downsampled version.

**Associated paper**: [Paper](https://iopscience.iop.org/article/10.3847/1538-4357/abc484/pdf)

**Domain expert**: [Blakesley Burkhart](https://www.bburkhart.com/), CCA, Flatiron Institute & Rutgers University.

**Code or software used to generate the data**: Fortran + MPI.

**Equation**:

$$
\begin{align*}
\frac{\partial \rho}{\partial t} + \nabla \cdot (\rho \mathbf{v}) &= 0 \\
\frac{\partial \rho \mathbf{v}}{\partial t} + \nabla \cdot (\rho \mathbf{v} \mathbf{v} - \mathbf{B} \mathbf{B}) + \nabla p &= 0 \\
\frac{\partial \mathbf{B}}{\partial t} - \nabla \times (\mathbf{v} \times \mathbf{B}) &= 0
\end{align*}
$$

where $\rho$ is the density, $\mathbf{v}$ is the velocity, $\mathbf{B}$ is the magnetic field, $\mathbf{I}$ the identity matrix and $p$ is the gas pressure.

![Gif](https://users.flatironinstitute.org/~polymathic/data/the_well/datasets/MHD_64/gif/density_unnormalized.gif)

| Dataset    | FNO | TFNO  | Unet | CNextU-net
|:-:|:-:|:-:|:-:|:-:|
| `MHD_64`  | 0.3605 | 3561 |0.1798|$\mathbf{0.1633}$|

Table: VRMSE metrics on test sets (lower is better). Best results are shown in bold. VRMSE is scaled such that predicting the mean value of the target field results in a score of 1.

## About the data

**Dimension of discretized data:** 100 timesteps of 64 $\times$ 64 $\times$ 64 cubes.

**Fields available in the data:** Density (scalar field), velocity (vector field), magnetic field (vector field).

**Number of trajectories:** 10 Initial conditions x 10 combination of parameters = 100 trajectories.

**Estimated size of the ensemble of all simulations:** 71.6 GB.

**Grid type:** uniform grid, cartesian coordinates.

**Initial conditions:** uniform IC.

**Boundary conditions:** periodic boundary conditions.

**Data are stored separated by ($\Delta t$):** 0.01 (arbitrary units).

**Total time range ($t\_{min}$ to $t\_{max}$):** $t\_{min} = 0$, $t\_{max} = 1$.

**Spatial domain size ($L_x$, $L_y$, $L_z$):** dimensionless so 64 pixels.

**Set of coefficients or non-dimensional parameters evaluated:** all combinations of $\mathcal{M}_s=${0.5, 0.7, 1.5, 2.0 7.0} and $\mathcal{M}_A =${0.7, 2.0}.

**Approximate time and hardware used to generate the data:** Downsampled from `MHD_256` after applying ideal low-pass filter.

## What is interesting and challenging about the data:

**What phenomena of physical interest are catpured in the data:** MHD fluid flows in the compressible limit (sub and super sonic, sub and super Alfvenic).

**How to evaluate a new simulator operating in this space:** Check metrics such as Power spectrum, two-points correlation function.

Please cite the associated paper if you use this data in your research:

```
@article{burkhart2020catalogue,
  title={The catalogue for astrophysical turbulence simulations (cats)},
  author={Burkhart, B and Appel, SM and Bialy, S and Cho, J and Christensen, AJ and Collins, D and Federrath, Christoph and Fielding, DB and Finkbeiner, D and Hill, AS and others},
  journal={The Astrophysical Journal},
  volume={905},
  number={1},
  pages={14},
  year={2020},
  publisher={IOP Publishing}
}
```
