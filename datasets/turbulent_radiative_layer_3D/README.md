# Turbulent Radiative Mixing Layers - 3D

**One line description of the data:** In many astrophysical systems, hot gas moves relative to cold gas, which leads to mixing. Mixing populates intermediate temperature gas that is highly reactive — in this case it is rapidly cooling.

**Longer description of the data:** In this simulation, there is cold, dense gas on the bottom and hot dilute gas on the top. They are moving relative to each other at highly subsonic velocities. This set up is unstable to the Kelvin Helmholtz instability, which is seeded with small scale noise that is varied between the simulations. The hot gas and cold gas are both in thermal equilibrium in the sense that the heating and cooling are exactly balanced. However, once mixing occurs as a result of the turbulence induced by the Kelvin Helmholtz instability, the intermediate temperatures become populated. This intermediate temperature gas is not in thermal equilibrium, and cooling beats heating. This leads to a net mass flux from the hot phase to the cold phase. This process occurs in the interstellar medium, and in the Circum-Galactic medium when cold clouds move through the ambient, hot medium. By understanding how the total cooling and mass transfer scale with the cooling rate, we are able to constrain how this process controls the overall phase structure, energetics and dynamics of the gas in and around galaxies.

**Associated paper**: [Paper](https://iopscience.iop.org/article/10.3847/2041-8213/ab8d2c/pdf).

**Domain expert**: [Drummond Fielding](https://dfielding14.github.io/), CCA, Flatiron Institute & Cornell University.

**Code or software used to generate the data**: [Athena++](https://www.athena-astro.app/).

**Equation**:

$$
\begin{align*}
\frac{ \partial \rho}{\partial t} + \nabla \cdot \left( \rho \vec{v} \right) &= 0 \\
\frac{ \partial \rho \vec{v} }{\partial t} + \nabla \cdot \left( \rho \vec{v}\vec{v} + P \right) &= 0 \\
\frac{ \partial E }{\partial t} + \nabla \cdot \left( (E + P) \vec{v} \right) &= - \frac{E}{t_{\rm cool}} \\
E = P / (\gamma -1) \, \, \gamma &= 5/3
\end{align*}
$$

with $\rho$ the density, $\vec{v}$ the 3D velocity, $P$ the pressure, $E$ the total energy, and $t_{\rm cool}$ the cooling time.

![Gif](https://users.flatironinstitute.org/~polymathic/data/the_well/datasets/turbulent_radiative_layer_3D/gif/density_normalized.gif)

| Dataset    | FNO | TFNO  | Unet | CNextU-net
|:-:|:-:|:-:|:-:|:-:|
| `turbulent_radiative_layer_3D`  | 0.5278 |0.5187| 0.3728 |$\mathbf{0.3667}$|

Table: VRMSE metrics on test sets (lower is better). Best results are shown in bold. VRMSE is scaled such that predicting the mean value of the target field results in a score of 1.

## About the data

**Dimension of discretized data:** 101 timesteps of 256 $\times$ 128 $\times$ 128 cubes.

**Fields available in the data:** Density (scalar field), pressure (scalar field), velocity (vector field).

**Number of trajectories:** 90 trajectories (10 different seeds for each of the 9 $t_{cool}$ variations).

**Estimated size of the ensemble of all simulations:** 744.6 GB.

**Grid type:** uniform, cartesian coordinates.

**Initial conditions:** Analytic, described in the [paper](https://ui.adsabs.harvard.edu/abs/2020ApJ...894L..24F/abstract).

**Boundary conditions:** periodic for the 128x128 directions ($x,y$), and zero-gradient for the 256 direction ($z$).

**Simulation time-step:** varies with $t_{cool}$. Smallest $t_{cool}$ is $1.32.10^{-2}$, largest $t_{cool}$ is $1.74.10^{-2}$. This is not in seconds, as this is a dimensionless simulation time. To convert, the code time is $L_{box}/cs_{hot}$, where $L_{box}$= 1 parsec and cs_{hot}=100km/s.

**Data are stored separated by ($\Delta t$):** data is separated by intervals of simulation time of 2.661722.

**Total time range ($t_{min}$ to $t_{max}$):** $t_{min} = 0$, $t_{max} = 266.172178$.

**Spatial domain size ($L_x$, $L_y$, $L_z$):** $x,y\in[-0.5,0.5]$, $z\in[-1,1]$.

**Set of coefficients or non-dimensional parameters evaluated:** $t_{cool} = \{0.03, 0.06, 0.1, 0.18, 0.32, 0.56, 1.00, 1.78, 3.16\}$.

**Approximate time to generate the data:** $34\,560$ CPU hours for all simulations.

**Hardware used to generate the data:** each simulation was generated on a 128 core "Rome" node.

## What is interesting and challenging about the data:

**What phenomena of physical interest are catpured in the data:** Capte the mass flux from hot to cold phase. Capture turbulent velocities. Capture the amount of mass per temperature bin ($T = \frac{P}{\rho}$).

**How to evaluate a new simulator operating in this space:** Check whether the above physical phenomena are captured by the algorithm.

Please cite the associated paper if you use this data in your research:

```
@article{fielding2020multiphase,
  title={Multiphase gas and the fractal nature of radiative turbulent mixing layers},
  author={Fielding, Drummond B and Ostriker, Eve C and Bryan, Greg L and Jermyn, Adam S},
  journal={The Astrophysical Journal Letters},
  volume={894},
  number={2},
  pages={L24},
  year={2020},
  publisher={IOP Publishing}
}
```
