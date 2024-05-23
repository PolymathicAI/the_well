# Magneto Hydrodynamics (MHD) compressible turbulence

**One line description of the data:** This is an MHD fluid flows in the compressible limit (subsonic, supersonic, sub-Alfvenic, super-Alfvenic)

**Longer description of the data:** 

**Associated paper**: https://iopscience.iop.org/article/10.3847/1538-4357/abc484/pdf

**Domain scientist**: Blakesley Burkhart, Center for Computational Astrophysics, CCA, Flatiron Institute & Rutgers University.

**Code or software used to generate the data**: Fortran + MPI.

**Equation**: 
$\begin{align}
\frac{\partial \rho}{\partial t} + \nabla \cdot (\rho \mathbf{v}) &= 0 \nonumber\\
\frac{\partial \rho \mathbf{v}}{\partial t} + \nabla \cdot (\rho \mathbf{v} \mathbf{v} - \mathbf{B} \mathbf{B}) + \nabla p &= 0 \nonumber\\
\frac{\partial \mathbf{B}}{\partial t} - \nabla \times (\mathbf{v} \times \mathbf{B}) &= 0 \nonumber\\
\end{align}
$
where $\rho$ is the density, $\mathbf{v}$ is the velocity, $\mathbf{B}$ is the magnetic field,$\mathbf{I}$ the identity matrix and $p$ is the gas pressure.

[ADD GIF OF THE SIMULATION]

# About the data

Dimension of discretized data: 100 timesteps of 256x256x256 cubes.

Fields available in the data: Density $\rho$, velocity in $x,y,z$ directions, magnetic field in $x,y,z$ directions.

Number of trajectories: 10 Initial conditions x 10 combination of parameters = 100 trajectories.

Estimated size of the ensemble of all simulations: 4.3TB

Grid type: uniform grid.

Initial conditions: uniform IC.

Boundary conditions: periodic boundary conditions.

Simulation time-step:

Data are stored separated by ($\Delta t$): 0.01

Total time range ($t_{min}$ to $t_{max}$): $t_min = 0$, $t_max = 1$.

Spatial domain size ($L_x$, $L_y$, $L_z$): dimensionless so 256 pixels.

Set of coefficients or non-dimensional parameters evaluated: all combinations of b = {0.1, 1} and p = {0.01, 0.1, 0.2, 1,  2}.

Approximate time to generate the data:2 days per simulation.

Hardware used to generate the data and precision used for generating the data: 64 cores.

# What is interesting and challenging about the data:

What phenomena of physical interest are catpured in the data: MHD fluid flows in the compressible limit (sub and super sonic, sub and super Alfvenic).

How to evaluate a new simulator operating in this space: Check metrics sur as Power spectrum, 2pcf and PDFs.
