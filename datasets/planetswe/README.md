# PlanetSWE

**One line description of the data:** Forced hyperviscous rotating shallow water on a sphere with earth-like topography and daily/annual periodic forcings.

**Longer description of the data:** The shallow water equations are fundamentally a 2D approximation of a 3D flow in the case where horizontal length scales are significantly longer than vertical length scales. They are derived from depth-integrating the incompressible Navier-Stokes equations. The integrated dimension then only remains in the equation as a variable describing the height of the pressure surface above the flow. These equations have long been used as a simpler approximation of the primitive equations in atmospheric modeling of a single pressure level, most famously in the Williamson test problems. This scenario can be seen as similar to Williamson Problem 7 as we derive initial conditions from the hPa 500 pressure level in ERA5. These are then simulated with realistic topography and two levels of periodicity.

**Associated paper**: [Paper](https://openreview.net/forum?id=RFfUUtKYOG).

**Domain expert**: [Michael McCabe](https://mikemccabe210.github.io/), Polymathic AI.

**Code or software used to generate the data**: [Dedalus](https://dedalus-project.readthedocs.io/en/latest/), adapted from [this example](https://dedalus-project.readthedocs.io/en/latest/pages/examples/ivp_sphere_shallow_water.html).

**Equation**:

$$
\begin{align*}
\frac{ \partial \vec{u}}{\partial t} &= - \vec{u} \cdot \nabla u - g \nabla h - \nu \nabla^4 \vec{u} - 2\Omega \times \vec{u} \\
\frac{ \partial h }{\partial t} &= -H \nabla \cdot \vec{u} - \nabla \cdot (h\vec{u}) - \nu \nabla^4h + F
\end{align*}
$$

with $h$ the deviation of pressure surface height from the mean, $H$ the mean height, $\vec{u}$ the 2D velocity, $\Omega$ the Coriolis parameter, and F the forcing which is defined:

```python
def find_center(t):
    time_of_day = t / day
    time_of_year = t / year
    max_declination = .4 # Truncated from estimate of earth's solar decline
    lon_center = time_of_day*2*np.pi # Rescale sin to 0-1 then scale to np.pi
    lat_center = np.sin(time_of_year*2*np.pi)*max_declination
    lon_anti = np.pi + lon_center  #2*np.((np.sin(-time_of_day*2*np.pi)+1) / 2)*pi
    return lon_center, lat_center, lon_anti, lat_center

def season_day_forcing(phi, theta, t, h_f0):
    phi_c, theta_c, phi_a, theta_a = find_center(t)
    sigma = np.pi/2
    coefficients = np.cos(phi - phi_c) * np.exp(-(theta-theta_c)**2 / sigma**2)
    forcing = h_f0 * coefficients
    return forcing
```

Visualization:

![Gif](https://users.flatironinstitute.org/~polymathic/data/the_well/datasets/planetswe/gif/planetswe.gif)

| Dataset    | FNO | TFNO  | Unet | CNextU-net
|:-:|:-:|:-:|:-:|:-:|
| `planetswe`  | 0.1727| $\mathbf{0.0853}$ | 0.3620 | 0.3724|

Table: VRMSE metrics on test sets (lower is better). Best results are shown in bold. VRMSE is scaled such that predicting the mean value of the target field results in a score of 1.

## About the data

**Dimension of discretized data:** 3024 timesteps of 256x512 images with "day" defined as 24 steps and "year" defined as 1008 in model time.

**Fields available in the data:** height (scalar field), velocity (vector field).

**Number of trajectories:** 40 trajectories of 3 model years.

**Estimated size of the ensemble of all simulations:** 185.8 GB.

**Grid type:** Equiangular grid, polar coordinates.

**Initial conditions:** Sampled from hPa 500 level of [ERA5](https://rmets.onlinelibrary.wiley.com/doi/10.1002/qj.3803), filtered for stable initialization and burned-in for half a simulation year.

**Boundary conditions:** Spherical.

**Simulation time-step ($\Delta t$):** CFL-based step size with safety factor of 0.4.

**Data are stored separated by ($\delta t$):** 1 hour in simulation time units.

**Total time range ($t_{min}$ to $t_{max}$):** $t_{min} = 0$, $t_{max} = 3024$.

**Spatial domain size:** $\phi \in [0, 2 \pi]$, $\theta \in [0, \pi]$.

**Set of coefficients or non-dimensional parameters evaluated:** $\nu$ normalized to mode 224.

**Approximate time to generate the data:** 45 minutes using 64 icelake cores for one simulation.

**Hardware used to generate the data:** 64 Icelake CPU cores.

## What is interesting and challenging about the data:

Spherical geometry and planet-like topography and forcing make for a proxy for real-world atmospheric dynamics where true dynamics are known. The dataset has annual and daily periodicity forcing models to either process a sufficient context length to learn these patterns or to be explicitly time aware. Furthermore, the system becomes stable making this a good system for exploring long run stability of models.

Please cite the associated paper if you use this data in your research:

```
@article{mccabe2023towards,
  title={Towards stability of autoregressive neural operators},
  author={McCabe, Michael and Harrington, Peter and Subramanian, Shashank and Brown, Jed},
  journal={arXiv preprint arXiv:2306.10619},
  year={2023}
}
```
