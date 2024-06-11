# Name of the simulation

**One line description of the data:** 3D radiation hydrodynamic simulations of convective envelopes of red supergiant stars.

**Longer description of the data:** Massive stars evolve into red supergiants, which have large radii and luminosities, and low-density, turbulent, convective envelopes. These simulations model the (inherently 3D) convective properties and gives insight into the progenitors of supernovae explosions.

**Associated paper**: [Paper](https://iopscience.iop.org/article/10.3847/1538-4357/ac5ab3)

**Domain scientist**: [Yan-Fei Jiang](https://jiangyanfei1986.wixsite.com/yanfei-homepage) (CCA, Flatiron Institute), [Jared Goldberg](https://jaredagoldberg.wordpress.com/) (CCA, Flatiron Institute), [Jeff Shen](https://jshen.net) (Princeton University).

**Code or software used to generate the data**: [Athena++](https://www.athena-astro.app/)

**Equations**
```math
\begin{align*}
\frac{\partial\rho}{\partial t}+\mathbf{\nabla}\cdot(\rho\mathbf{v})&=0\\
\frac{\partial(\rho\mathbf{v})}{\partial t}+\mathbf{\nabla}\cdot({\rho\mathbf{v}\mathbf{v}+{{\sf P_{\rm gas}}}}) &=-\mathbf{G}_r-\rho\mathbf{\nabla}\Phi \\
\frac{\partial{E}}{\partial t}+\mathbf{\nabla}\cdot\left[(E+ P_{\rm gas})\mathbf{v}\right] &= -c G^0_r -\rho\mathbf{v}\cdot\mathbf{\nabla}\Phi\\
\frac{\partial I}{\partial t}+c\mathbf{n}\cdot\mathbf{\nabla} I &= S(I,\mathbf{n})
\end{align*}
```
where 
- $\rho$ = gas density
- $\mathbf{v}$ = flow velocity
- ${\sf P_{\rm gas}}$ = gas pressure tensor
- $P_{\rm gas}$ = gas pressure scalar
- $E$ = total gas energy density
    - $E = E_g + \rho v^2 / 2$, where $E_g = 3 P_{\rm gas} / 2$ = gas internal energy density
- $G^0_r$ and $\mathbf{G}_r$ = time-like and space-like components of the radiation four-force
- $I$ = frequency integrated intensity, which is a function of time, spatial coordinate, and photon propagation direction $\mathbf{n}$
- $\mathbf{n}$ = photon propagation direction

![Gif](https://users.flatironinstitute.org/~polymathic/data/the_well/datasets/convective_envelope_rsg/gif/density_unnormalized.gif)

| Dataset    | FNO | TFNO  | Unet | CNextU-net
|:-:|:-:|:-:|:-:|:-:|
| convective_envelope_rsg  | 1.08  | $\mathbf{1.06}$ |1.15|1.07|

Preliminary benchmarking, in VRMSE.

# About the data

**Dimension of discretized data:** $100$ time-steps of $256\times 128 \times 256$ images per trajectory.

**Fields available in the data:** energy (scalar field), density (scalar field), pressure (Scalar field), velocity (vector field).

**Number of trajectories:** 29 (they are cuts of one long trajectory, long trajectory available on demand).

**Estimated size of the ensemble of all simulations:** 854 GB.

**Grid type:** spherical coordinates, uniform in $(\log r, \theta,\phi)$.  Simulations are done for a portion of a sphere (not the whole sphere), so the simulation volume is like a spherical cake slice.

**Initial and boundary conditions:** The temperature at the inner boundary (IB) is first set to equal that of the appropriate radius coordinate in the MESA (1D) model ($400\~R_\odot$ and $300\~R_\odot$) and the density selected to approximately recover the initial total mass of the star in the simulation ($15.4\~M_\odot$ and $14\~M_\odot$). 
Between $300\~R_\odot$ and $400\~R_\odot$, the initial profile is constructed with the radiative luminosity to be $10^5\~L_\odot$, and this is kept fixed in the IB.


**Simulation time-step:** 198 days.

**Data are stored separated by ($\Delta t$):** units here are sort of arbitrary, $\Delta t= 8$.

**Total time range ($t_{min}$ to $t_{max}$):**  0, 806 (arbitrary).

**Spatial domain size:** $R$ from $300-6700~{\rm R_\odot}$, θ from $π/4−3π/4$ and $\phi$ from $0−π$, with $δr/r ≈ 0.01$.

**Set of coefficients or non-dimensional parameters evaluated:**

| Simulation | radius of inner boundary $R_{IB}/R_\odot$ | radius of outer boundary $R_{OB}/R_\odot$ | heat source | resolution (r × θ × $\phi$) | duration | core mass $mc/M\odot$ | final mass $M_{\rm final}/M_\odot$ |
|--|--|--|--|--|--|--|--|
| Whole simulation (to obtain the 29 trajectories) | 300 | 6700 | fixed L | 256 × 128 × 256 | 5766 days | 10.79 | 12.9 |

**Approximate time to generate the data:** 2 months on 80 nodes for each run.

**Hardware used to generate the data:** 80x NASA Pleiades Skylake CPU nodes.

# What is interesting and challenging about the data:

**What phenomena of physical interest are captured in the data:** turbulence and convection (inherently 3D processes), variability.

**How to evaluate a new simulator operating in this space:** can it predict behaviour of simulation in convective steady-state, given only perhaps a few snapshots at the beginning of the simulation?

**Caveats:** complicated geometry, size of a slice in R varies with R (think of this as a slice of cake, where the parts of the slice closer to the outside have more area/volume than the inner parts), simulation reaches convective steady-state at some point and no longer "evolves".
