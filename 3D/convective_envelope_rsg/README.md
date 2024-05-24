# Name of the simulation

**One line description of the data:** 3D radiation hydrodynamic simulations of convective envelopes of red supergiant stars

**Longer description of the data:** Massive stars evolve into red supergiants, which have large radii and luminosities, and low-density, turbulent, convective envelopes. These simulations model the (inherently 3D) convective properties and gives insight into the progenitors of supernovae explosions.

**Associated paper**: [Numerical Simulations of Convective Three-dimensional Red Supergiant Envelopes](https://iopscience.iop.org/article/10.3847/1538-4357/ac5ab3)

**Domain scientist**: Yan-Fei Jiang (CCA), Jared Goldberg (CCA), [Jeff Shen (Princeton)](https://jshen.net)

**Code or software used to generate the data**: [Athena++](https://www.athena-astro.app/)

**Equations**
$$
\frac{\partial\rho}{\partial t}+\bm{\nabla}\cdot(\rho\bm{v})=0\\
\frac{\partial(\rho\bm{v})}{\partial t}+\bm{\nabla}\cdot({\rho\bm{v}\bm{v}+{{\sf P_{\rm gas}}}}) =-\mathbf{G}_r-\rho\bm{\nabla}\Phi \\
\frac{\partial{E}}{\partial t}+\bm{\nabla}\cdot\left[(E+ P_{\rm gas})\bm{v}\right] = -c G^0_r -\rho\bm{v}\cdot\bm{\nabla}\Phi\\
\frac{\partial I}{\partial t}+c\bm{n}\cdot\bm{\nabla} I = S(I,\bm{n})
$$
where 
- $\rho$ = gas density
- $\bm{v}$ = flow velocity
- ${\sf P_{\rm gas}}$ = gas pressure tensor
- $P_{\rm gas}$ = gas pressure scalar
- $E$ = total gas energy density
    - $E = E_g + \rho v^2 / 2$, where $E_g = 3 P_{\rm gas} / 2$ = gas internal energy density
- $G^0_r$ and $\mathbf{G}_r$ = time-like and space-like components of the radiation four-force
- $I$ = frequency integrated intensity, which is a function of time, spatial coordinate, and photon propagation direction $\bm{n}$
- $\bm{n}$ = photon propagation direction


See [sim.mp4](sim.mp4) for a visualization of one of the simulations.

# About the data

Dimension of discretized data: Coordinates are in `time x r x theta x phi`.
- Simulation 1: `878 x 384 x 128 x 256`. 
- Simulation 2: `2926 x 256 x 128 x 256`. 


Fields available in the data: `energy`, `rho`, `pressure`, `vx`, `vy`, `vz` (note that `vx`, `vy`, `vz` correspond to velocities in the `r`, `theta`, and `phi` directions)

Number of trajectories: 2

Estimated size of the ensemble of all simulations: 854 GB

Grid type: spherical coordinates, uniform in `log R`, `theta`, `phi`. The simulation is done for a portion of a sphere (not the whole sphere), so the simulation volume is like a spherical cake slice.

Initial and boundary conditions: The temperature at the inner boundary (IB) is first set to equal that of the appropriate radius coordinate in the MESA (1D) model ($400~R_\odot$ and $300~R_\odot$) and the density selected to approximately recover the initial total mass of the star in the simulation ($15.4~M_\odot$ and $14~M_\odot$). 

- Simulation 1: To perturb from the radiatively stable initial conditions and supply the convective luminosity, we increase the temperature at the IB by 10% compared with the initial condition (a “hot plate”), while density is fixed and velocity is reflective at the inner boundary. This boundary condition produces a radiative layer near the bottom with the desired luminosity, which causes the envelope away from the bottom boundary to be convective. 
- Simulation 2: Between $300~R_\odot$ and $400~R_\odot$, the initial profile is constructed with the radiative luminosity to be $10^5~L_\odot$, and this is kept fixed in the IB.


Simulation time-step: Simulation 1: 5865 days, Simulation 2: 5766 days

Data are stored separated by ($\Delta t$): units here are sort of arbitrary, but simulation 1 has dt=4, simulation 2 has dt=8

Total time range ($t_{min}$ to $t_{max}$): Simulation 1: 20000-23508, simulation 2: 2-23402 (again, sort of arbitrary)

Spatial domain size:
- Simulation 1: $R$ from $400-22,400~{\rm R_\odot}$, θ from $π/4−3π/4$ and $\phi$ from $0−π$, with $δr/r ≈ 0.01$
- Simulation 2: $R$ from $300-6700~{\rm R_\odot}$, θ from $π/4−3π/4$ and $\phi$ from $0−π$, with $δr/r ≈ 0.01$

Set of coefficients or non-dimensional parameters evaluated:

| Simulation | radius of inner boundary $R_{IB}/R_\odot$ | radius of outer boundary $R_{OB}/R_\odot$ | heat source | resolution (r × θ × $\phi$) | duration | core mass $mc/M\odot$ | final mass $M_{\rm final}/M_\odot$ |
|--|--|--|--|--|--|--|--|
| 1 | 400 | 22,400 | hot plate | 384 × 128 × 256 | 5865 days | 12.8 | 16.4 |
| 2 | 300 | 6700 | fixed L | 256 × 128 × 256 | 5766 days | 10.79 | 12.9 |

Approximate time to generate the data: 2 months on 80 nodes for each run

Hardware used to generate the data and precision used for generating the data: 80x NASA Pleiades Skylake CPU nodes

# What is interesting and challenging about the data:

What phenomena of physical interest are captured in the data: turbulence and convection (inherently 3D processes), variability

How to evaluate a new simulator operating in this space: can it predict behaviour of simulation in convective steady-state, given only perhaps a few snapshots at the beginning of the simulation?

Caveats: complicated geometry, size of a slice in R varies with R (think of this as a slice of cake, where the parts of the slice closer to the outside have more area/volume than the inner parts), simulation reaches convective steady-state at some point and no longer "evolves"
