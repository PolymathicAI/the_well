# Post neutron star merger

**One line description of the data:** Simulations of the aftermath of a neutron star merger.

**Longer description of the data:** The simulations presented here are axisymmetrized snapshots of full three-dimensional general relativistic neutrino radiation magnetohydrodynamics. The plasma physics is treated with finite volumes with constrained transport for the magnetic field on a curvilinear grid. The system is closed by a tabulated nuclear equation of state assuming nuclear statistical equilibrium (NSE). The radiation field is treated via Monte Carlo transport, which is a particle method. The particles are not included in this dataset, however their effects are visible as source terms on the fluid.

**Associated paper**: The simulations included here are from a series of papers: [Paper 1](https://iopscience.iop.org/article/10.3847/1538-4365/ab09fc/pdf), [Paper 2](https://link.aps.org/accepted/10.1103/PhysRevD.100.023008), [Paper 3](https://arxiv.org/abs/1912.03378), [Paper 4](https://arxiv.org/abs/2212.10691), [Paper 5](https://arxiv.org/abs/2311.05796).

**Domain expert**: [Jonah Miller](https://www.thephysicsmill.com/), Los Alamos National Laboratory.

**Code or software used to generate the data**: Open source software [nublight](https://github.com/lanl/nubhlight).

**Equation**: See equations 1-5 and 16 of Miller, Ryan, Dolence (2019).

The fluid sector consists of the following system of equations.

$$
\begin{align*}
  \partial_t \left(\sqrt{g}\rho_0 u^t\right) + \partial_i\left(\sqrt{g}\rho_0u^i\right) &= 0 \\
  \partial_t\left[\sqrt{g} \left(T^t_{\ \nu} + \rho_0u^t \delta^t_\nu\right)\right] + \partial_i\left[\sqrt{g}\left(T^i_{\ \nu} + \rho_0 u^i \delta^t_\nu\right)\right] &= \sqrt{g} \left(T^\kappa_{\ \lambda} \Gamma^\lambda_{\nu\kappa} + G_\nu\right)\,\,\,\, \forall \nu = 0,1,\ldots,4
\end{align*}
$$

$$
\begin{align*}
  \partial_t \left(\sqrt{g} B^i\right) + \partial_j \left[\sqrt{g}\left(b^ju^i - b^i u^j\right)\right] &= 0 \\
  \partial_t\left(\sqrt{g}\rho_0 Y_e u^t\right) + \partial_i\left(\sqrt{g}\rho_0Y_eu^i\right) &= \sqrt{g} G_{\text{ye}}
\end{align*}
$$

The standard radiative transfer equation is

$$
\frac{D}{d\lambda}\left(\frac{h^3\mathcal{I}_{\nu,f}}{\varepsilon^3}\right) = \left(\frac{h^2\eta _{\nu,f}}{\varepsilon^2}\right) - \left(\frac{\varepsilon \chi _{\nu,f}}{h}\right) \left(\frac{h^3\mathcal{I} _{\nu,f}}{\varepsilon^3}\right)
$$

<p align="center"> <img src="https://users.flatironinstitute.org/~polymathic/data/the_well/datasets/post_neutron_star_merger/gif/Ye_good_normalized.gif" width="50%"></p>




| Dataset    | FNO | TFNO  | Unet | CNextU-net
|:-:|:-:|:-:|:-:|:-:|
| `post_neutron_star_merger`  | 0.3866 | $\mathbf{0.3793}$ | - |-|

Table: VRMSE metrics on test sets (lower is better). Best results are shown in bold. VRMSE is scaled such that predicting the mean value of the target field results in a score of 1. Unet and CNextU-net results are not available as these architectures require all the dimensions of the data to be multiples of 2.


## About the data

**Dimension of discretized data:** 181 time-steps of 192 $\times$ 128 $\times$ 66 snapshots.

**Fields available in the data:** fluid density (scalar field), fluid internal energy (scalar field), electron fraction (scalar field), temperate (scalar field), entropy (scalar field), velocity (vector field), magnetic field (vector field), contravariant tensor metric of space-time (tensor field, no time-dependency). A description of fields available in an output file can be found [here](https://github.com/lanl/nubhlight/wiki).

**Number of trajectories:** 8 full simulations.

**Size of the ensemble of all simulations:** 110.1 GB.

**Grid type**: Uniform grid, log-spherical coordinates.

**Initial conditions:** Constant entropy torus in hydrostatic equilibrium orbiting a black hole. Black hole mass and spin, as well as torus mass, spin, electron fraction, and entropy vary.

**Boundary conditions:** open.

**Simulation time-step:** approximately 0.01 in code units. Physical time varies; roughly 147 nanoseconds for fiducial model.

**Data are stored separated by ($\Delta t$):** 50 in code units. Physical time varies; roughly 0.6 milliseconds for fiducial model.

**Total time range ($t_{min}$ to $t_{max}$):** 10000 in code units. Physical time varies; roughly 127 milliseocnds for fudicial model

**Spatial domain size:** Spherical coordinates. Radius roughly 2 to 1000 in code units. Physical values vary. Outer boundary is at roughly 4000 for fiducial model. Polar angle 0 to pi. Azimuthal angle 0 to 2*pi. Note that the coordinates are curvilinear. In Cartesian space, spacing is logarithmic in radius and there is a focusing of grid lines near the equator.

**Set of coefficients or non-dimensional parameters evaluated:** Black hole spin parameter a, ranges 0 to 1. Initial mass and angular momentum of torus. In dimensionless units, evaluated as inner radius Rin and radius of maximum pressure Rmax. Torus initial electron fraction Ye and entropy kb. Black hole mass in solar masses.

**Approximate time to generate the data:** Roughly 3 weeks per simulation on 300 cores.

**Hardware used to generate the data and precision used for generating the data:** Data generated at double precision on several different supercomputers. All calculations were CPU calculations parallelized with a hybrid MPI + OpenMP strategy. 1 MPI rank per socket. Oldest calculations performed on the Los Alamos Badger cluster, now decommissioned. Intel Xeon E5-2695v5 2.1 GHz. 12 cores per socket, 24 core cores per node. Simulations run on 33 nodes. Some newer simulations run on Los Alamos Capulin cluster, now decomissioned. ARM ThunderX2 nodes. 56 cores per node. Simulation run on 33 nodes.

## Simulation Index

| Scenario | Shorthand name | Description                                                         |
|----------|----------------|---------------------------------------------------------------------|
| 0        | collapsar_hi   | Disk resulting from collapse of massive rapidly rotating star.       |
| 1        | torus_b10      | Disk inspired by 2017 observation of a neutron star merger. Highest magnetic field strength. |
| 2        | torus_b30      | Disk inspired by 2017 observation of a neutron star merger. Intermediate magnetic field strength. |
| 3        | torus_gw170817 | Disk inspired by 2017 observation of a neutron star merger. Weakest magnetic field strength. |
| 4        | torus_MBH_10   | Disk from black hole-neutron star merger. 10 solar mass black hole.  |
| 5        | torus_MBH_2p31 | Disk from black hole-neutron star merger. 2.31 solar mass black hole.|
| 6        | torus_MBH_2p67 | Disk from black hole-neutron star merger. 2.76 solar mass black hole.|
| 7        | torus_MBH_2p69 | Disk from black hole-neutron star merger. 2.79 solar mass black hole.|
| 8        | torus_MBH_6    | Disk from black hole-neutron star merger. 6 solar mass black hole.   |


## General relativistic quantities
The core quantity that describes the curvature of spacetime and its
impact on a simulation is `['t0_fields']['gcon']` of the HDF5 file. From this, other quantities can be computed.

## To reproduce
The values in `simulation_parameters.json` are sufficient to reproduce a
simulation using [nubhlight](https://github.com/lanl/nubhlight) using
the `torus_cbc` problem generator, with one exception. You must
provide tabulated equation of state and opacity data. We use the SFHo
equation of state provided on the
[stellar collapse website](https://stellarcollapse.org/).
Tabulated neutrino opacities were originally computed for the Fornax
code and are not public. However adequate open source substitutes may
be generated by the [nulib](http://www.nulib.org/) library.

## Explanation of simulation parameters

Here we include, for completeness, a description of the different simulation parameters. which cover the simulation parameters chosen. Their value for each simulation is stored in `simulation_parameters.json`.

- `B_unit`, the unit of magnetic field strength. Multiplying code quantity by `B_unit` converts the quantity to units of Gauss.
- `DTd`, dump time cadence.
- `DTl`, log output time cadence.
- `DTp`, permanent restart file time cadence.
- `DTr`, temporary restart file time cadence.
- `Ledd`, (Photon) Eddington luminosity based on black hole mass.
- `L_unit`, length unit. Multiplying code quantity by `L_unit` converts it into units of cm.
- `M_unit`, mass unit. Multiplying code quantity by `M_unit` converts it into units of g.
- `Mbh`, black hole mass in units of g.
- `MdotEdd`, (Photon) Eddington accretion rate based on black hole mass.
- `N1`, number of grid points in X1 (radial) direction.
- `N2`, number of grid points in X2 (polar) direction.
- `N3`, number of grid points in X3 (azimuthal) direction.
- `PATH`, output directory for the original simulation.
- `RHO_unit`, density unit. Multiplying code quantity by `RHO_unit` converts it into units of g/cm^3.
- `Reh`, radius of the event horizon in code units.
- `Rin`, radius of the inner boundary in code units.
- `Risco`, radius of the innermost stable circular orbit in code units.
- `Rout_rad`, outer radius of neutrino transport.
- `Rout_vis`, radius used for 3D volume rendering.
- `TEMP_unit`, temperature unit. Converts from MeV (code units) to Kelvin.
- `T_unit`, time unit. Converts from code units to seconds.
- `U_unit`, energy density unit. Multiplying code quantity by `U_unit` converts it into units of erg/cm^3.
- `a`, dimensionless black hole spin.
- `cour`, dimensionless CFL factor used to set the timestep based on the grid spacing.
- `dx`, array of grid spacing in code coordinates. (Uniform.)
- `maxnscatt`, maximum number of scattering events per superphoton particle
- `mbh`, black hole mass in solar masses.
- `hslope`, `mks_smooth`, `poly_alpha`, `poly_xt` focusing terms used for coordinate transforms
- `startx`, array of starting coordinate values for `X1`,`X2`,`X3` in code coordinates.
- `stopx`, array of ending coordinate values for `X1`,`X2`,`X3` in code coordinates.
- `tf`, final simulation time.
- `variables` list of names of primitive state vector.

## What is interesting and challenging about the data:
**What phenomena of physical interest are catpured in the data:** The 2017 detection of the in-spiral and merger of two neutron stars
was a landmark discovery in astrophysics. Through a wealth of
multi-messenger data, we now know that the merger of these
ultracompact stellar remnants is a central engine of short gamma ray
bursts and a site of r-process nucleosynthesis, where the heaviest
elements in our universe are formed. The radioactive decay of unstable
heavy elements produced in such mergers powers an optical and
infra-red transient: The kilonova.

One key driver of nucleosynthesis and resultant electromagnetic
afterglow is wind driven by an accretion disk formed around the
compact remnant. Neutrino transport plays a key role in setting the
electron fraction in this outflow, thus controlling the
nucleosynthesis.

Collapsars are black hole accretion disks formed after the core of a
massive, rapidly rotating star collapses to a black hole. These
dramatic systems rely on much the same physics and modeling as
post-merger disks, and can also be a key driver of r-processes
nucleosynthesis.

**How to evaluate a new simulator operating in this space:** The electron fraction Ye of material blown off from the disk is the core
"delivarable." It determines how heavy elements are synthesized, which
in turn determines the electromagnetic counterpart as observed on
Earth. This is the most important piece to get right from an emulator.

Please cite these associated papers if you use this data in your research:

```
@article{miller2019nubhlight,
  title={$\nu$bhlight: radiation GRMHD for neutrino-driven accretion flows},
  author={Miller, Jonah M and Ryan, Ben R and Dolence, Joshua C},
  journal={The Astrophysical Journal Supplement Series},
  volume={241},
  number={2},
  pages={30},
  year={2019},
  publisher={IOP Publishing}
}
@article{miller2019full,
  title={Full transport model of GW170817-like disk produces a blue kilonova},
  author={Miller, Jonah M and Ryan, Benjamin R and Dolence, Joshua C and Burrows, Adam and Fontes, Christopher J and Fryer, Christopher L and Korobkin, Oleg and Lippuner, Jonas and Mumpower, Matthew R and Wollaeger, Ryan T},
  journal={Physical Review D},
  volume={100},
  number={2},
  pages={023008},
  year={2019},
  publisher={APS}
}
@article{miller2020full,
  title={Full transport general relativistic radiation magnetohydrodynamics for nucleosynthesis in collapsars},
  author={Miller, Jonah M and Sprouse, Trevor M and Fryer, Christopher L and Ryan, Benjamin R and Dolence, Joshua C and Mumpower, Matthew R and Surman, Rebecca},
  journal={The Astrophysical Journal},
  volume={902},
  number={1},
  pages={66},
  year={2020},
  publisher={IOP Publishing}
}
@article{curtis2023nucleosynthesis,
  title={Nucleosynthesis in outflows from black hole--neutron star merger disks with full gr ($\nu$) rmhd},
  author={Curtis, Sanjana and Miller, Jonah M and Fr{\"o}hlich, Carla and Sprouse, Trevor and Lloyd-Ronning, Nicole and Mumpower, Matthew},
  journal={The Astrophysical Journal Letters},
  volume={945},
  number={1},
  pages={L13},
  year={2023},
  publisher={IOP Publishing}
}
@article{lund2024magnetic,
  title={Magnetic Field Strength Effects on Nucleosynthesis from Neutron Star Merger Outflows},
  author={Lund, Kelsey A and McLaughlin, Gail C and Miller, Jonah M and Mumpower, Matthew R},
  journal={The Astrophysical Journal},
  volume={964},
  number={2},
  pages={111},
  year={2024},
  publisher={IOP Publishing}
}
```
