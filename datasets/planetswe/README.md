# PlanetSWE

**One line description of the data:** Forced hyperviscous rotating shallow water on a sphere with earth-like topography and daily/annual periodic forcings. 

**Longer description of the data:** The shallow water equations are fundamentally a 2D approximation of a 3D flow in the case where horizontal length scales are significantly longer than vertical length scales. They are derived from depth-integrating the incompressible Navier-Stokes equations. The integrated dimension then only remains in the equation as a variable describing the height of the pressure surface above the flow. These equations have long been used as a simpler approximation of the primitive equations in atmospheric modeling of a single pressure level, most famously in the Williamson test problems. This scenario can be seen as similar to Williamson Problem 7 as we derive initial conditions from the hPa 500 pressure level in ERA5. These are then simulated with realistic topography and two levels of periodicity. 

**Associated paper**: [Paper](https://openreview.net/forum?id=RFfUUtKYOG)

**Domain scientist**: [Michael McCabe](https://mikemccabe210.github.io/), Polymathic AI.

**Code or software used to generate the data**: Dedalus

**Equation**: 

```math
\begin{align}
\frac{ \partial \vec{u}}{\partial t} &= - \vec{u} \cdot \nabla u - g \nabla h - \nu \nabla^4 - 2\Omega \times \vec{u} \\
\frac{ \partial h }{\partial t} = -H \nabla \cdot \vec{u} - \nabla \cdot (h\vec{u}) - \nu \nabla^4h + F  
\end{align}
```
with $\rho$ the density, $\vec{v}$ the 2D velocity, $P$ the pressure, $E$ the total energy, and $t_{\rm cool}$ the cooling time.

![Gif](gif/density_normalized.gif)

| Dataset    | FNO | TFNO  | Unet | CNextU-net
|:-:|:-:|:-:|:-:|:-:|
| turbulent_radiative_layer_2D  | 0.967| 1.01 |0.576| 0.575|

Preliminary benchmarking, in VRMSE.


# About the data

**Dimension of discretized data:** 101 timesteps of 384x128 images.

**Fields available in the data:** Density (scalar field), pressure (scalar field), velocity (vector field).

**Number of trajectories:** 90 (10 different seeds for each of the 9 $t_{cool}$ values).

**Estimated size of the ensemble of all simulations:** 14GB

**Grid type:** uniform, cartesian coordinates.

**Initial conditions:** Analytic, described in the [paper](https://ui.adsabs.harvard.edu/abs/2020ApJ...894L..24F/abstract).

**Boundary conditions:** Periodic in the x-direction, zero-gradient for the y-direction.

**Simulation time-step ($\Delta t$):** varies with $t_{cool}$. Smallest $t_{cool}$ has $\Delta t = 1.36\times10^{-2}$ and largest $t_{cool}$ has $\Delta t = 1.74\times10^{-2}$. Not that this is not in seconds. This is in dimensionless simulation time.

**Data are stored separated by ($\delta t$):** 1.597033 in simulation time.

**Total time range ($t_{min}$ to $t_{max}$):** $t_{min} = 0$, $t_{max} = 159.7033$.

**Spatial domain size ($L_x$, $L_y$, $L_z$):** $x \in [-0.5, 0.5]$, $y \in [-1, 2]$ giving $L_x = 1$ and $L_y = 3$.

**Set of coefficients or non-dimensional parameters evaluated:** $t_{cool} = \{0.03, 0.06, 0.1, 0.18, 0.32, 0.56, 1.00, 1.78, 3.16\}$. 

**Approximate time to generate the data:** 84 seconds using 48 cores for one simulation. 100 CPU hours for everything.

**Hardware used to generate the data:** 48 CPU cores.

# What is interesting and challenging about the data:

**What phenomena of physical interest are catpured in the data:**
-	The mass flux from hot to cold phase.
-	The turbulent velocities.
-	Amount of mass per temperature bin (T = press/dens).


**How to evaluate a new simulator operating in this space:** See whether it captures the right mass flux, the right turbulent velocities, and the right amount of mass per temperature bin.
