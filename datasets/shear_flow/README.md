
# Periodic shear flow

**One line description of the data:** 2D periodic incompressible shear flow. 

**Longer description of the data:** 
Shear flow are a type of fluid characterized by the continuous deformation of adjacent fluid layers sliding past each other with different velocities. This phenomenon is commonly observed in various natural and engineered systems, such as rivers, atmospheric boundary layers, and industrial processes involving fluid transport.
The dataset explores a 2D periodic shearflow governed by incompressible Navier-Stokes equation. 

**Associated paper**: https://journals.aps.org/prresearch/abstract/10.1103/PhysRevResearch.2.023068 [SHOULD WE PUT DEDALUS PAPER?]

**Domain scientist**: Keaton Burns?, MIT Applied Mathematics.

**Code or software used to generate the data**: https://dedalus-project.readthedocs.io/en/latest/pages/examples/ivp_2d_shear_flow.html

**Equation**:

While we solve equations in the frequency domain, the original time-domain problem is 
$$ 
\begin{align*}
\frac{\partial u}{\partial t} + \nabla p - \nu \Delta u & = -u\cdot\nabla u\,,
\\
\frac{\partial s}{\partial t} - D \Delta s & = -u \cdot\nabla s\,, 
\end{align*}
$$
where $\Delta = \nabla \cdot \nabla$ is the spatial Laplacian, $u = (u_1,u_2)$ with $u_1$ the shear and $u_2$ the velocity, $s$ is the tracer, and $p$ is the pressure, 
with the additional constraints $\int p = 0$ (pressure gauge).

These PDE are parameterized by the Reynolds and Schmidt numbers through $\nu$ and $D$.
$$
\begin{align*}
\text{(viscosity)} ~~~~~~~ \nu & = 1 \, / \, \text{Reynolds}
\\
\text{(diffusivity)} ~~~~~~~ D & = \nu \, / \, \text{Schmidt}
\end{align*}
$$
The tracer is passive and here for visualization purposes only.

![Simulation GIF](gif_data/shearflow_final.gif)

# About the data

Dimension of discretized data: 

$128\times256$ images with $200$ timesteps.
Five fields are available in the data: tracer, shear, vorticity, velocity, pressure.
Number of simulations: $1120$ ($28$ PDE parameters $\times$ $40$ initial conditions).

Size of the ensemble of all simulations: 277GB

Grid type: real Fourier

Initial conditions: the shear field $u_1$ is composed of $n_\text{shear}$ shears uniformly spaced along the $z$ direction. Each shear is implemented with a tanh (hyperbolic tangent) $\text{tanh}(5\frac{z-z_k}{n_\text{shear}w})$ where $z_k$ is the vertical position of the shear and $w$ is a width factor.
The velocity field $u_2$ is composed of sinusoids along the $x$ direction located at the shear. These sinusoids have an exponential decay away from the shear in the $z$ direction $\text{sin}(n_\text{blobs}\pi x)\,e^{\frac{25}{w^2}|z-z_k|^2}$.
The tracer matches the shear at initialization. The pressure is initialized to zero.
The initial condition is thus indexed by $n_\text{shear},n_\text{blobs},w$.

Boundary conditions: 2D-periodic.

Simulation time-step: 0.1.

Total time range ($t_{min}$ to $t_{max}$): $t_{\mathrm{min}} = 0$, $t_{\mathrm{max}} = 20$.

Spatial domain size: $ 0 \leq x \leq 1$ horizontally, and $-1 \leq z \leq 1$ vertically.

Set of coefficients or non-dimensional parameters evaluated: $\text{Reynolds}\in[1e4, 5e4, 1e5, 5e5], \text{Schmidt}\in[0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]$. For initial conditions $n_\text{shear}\in[2,4]$,$n_\text{blobs}\in[2,3,4,5]$,$w\in[0.25, 0.5, 1.0, 2.0, 4.0]$.


Approximate time to generate the data: per input parameter: $\sim 1500s$, total: $\sim 5$ hours 
[THIS IS THE TIME AFTER HEAVY PARALLELIZATION, BUT SHOULD WE INCLUDE THE TIME AS IF THE COMPUTATION WAS DONE ON A SINGLE GPU?].

Hardware used to generate the data and precision used for generating the data: 7 nodes of 64 CPU cores each with 32 tasks running in parallel on each node, in single precision.

# What is interesting and challenging about the data:

Shear flow are non-linear phenomena arrising in fluid mechanics and turbulence.
Predicting the behavior of the shear flow under different Reynolds and Schmidt numbers is essential for a number of applications in aerodynamics, automotive, biomedical. 
Furthermore, such flow are unstable at large Reynolds number. 


