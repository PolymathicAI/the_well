# Turbulent Radiative Layer - 2D

**One line description of the data:** Everywhere in astrophysical systems hot gas moves relative to cold gas, which leads to mixing, and mixing populates intermediate temperature gas that is highly reactive—in this case it is rapidly cooling.

**Longer description of the data:** In this simulation, there is cold, dense gas on the bottom and hot dilute gas on the top. They are moving relative to each other at highly subsonic velocities. This set up is unstable to the Kelvin Helmholtz instability, which is seeded with small scale noise that is varied between the simulations. The hot gas and cold gas are both in thermal equilibrium in the sense that the heating and cooling are exactly balanced. However, once mixing occurs as a result of the turbulence induced by the Kelvin Helmholtz instability the intermediate temperatures become populated. This intermediate temperature gas is not in thermal equilibrium and cooling beats heating. This leads to a net mass flux from the hot phase to the cold phase. This process occurs in the interstellar medium, and in the Circum-Galactic medium when cold clouds move through the ambient, hot medium. By understanding how the total cooling and mass transfer scale with the cooling rate we are able to constrain how this process controls the overall phase structure, energetics and dynamics of the gas in and around galaxies.

**Associated paper**: https://iopscience.iop.org/article/10.3847/2041-8213/ab8d2c/pdf

**Domain scientist**: Drummond Fielding, Flatiron Institute.

**Code or software used to generate the data**: Athena++

**Equation**: $\begin{align}
\frac{ \partial \rho}{\partial t} + \nabla \cdot \left( \rho \vec{v} \right) &= 0 \\
\frac{ \partial \rho \vec{v} }{\partial t} + \nabla \cdot \left( \rho \vec{v}\vec{v} + P \right) &= 0 \\
\frac{ \partial E }{\partial t} + \nabla \cdot \left( (E + P) \vec{v} \right) &= - \frac{E}{t_{\rm cool}} \\
E = P / (\gamma -1) \, \, \gamma &= 5/3
\end{align}$
with $\rho$ the density, $\vec{v}$ the 2D velocity, $P$ the pressure, $E$ the total energy, and $t_{\rm cool}$ the cooling time.

[ADD GIF OF THE SIMULATION]

# About the data

Dimension of discretized data: 101 timesteps of 384x128 images.

Fields available in the data: Density $\rho$, velocity $\vec{v}$, pressure $P$.

Number of trajectories:

Estimated size of the ensemble of all simulations:

Grid type

Initial conditions:

Boundary conditions:

Simulation time-step:

Data are stored separated by ($\Delta t$):

Total time range ($t_{min}$ to $t_{max}$):

Spatial domain size ($L_x$, $L_y$, $L_z$):

Set of coefficients or non-dimensional parameters evaluated:

Approximate time to generate the data:

Hardware used to generate the data and precision used for generating the data:

# What is interesting and challenging about the data:

What phenomena of physical interest are catpured in the data:

How to evaluate a new simulator operating in this space:
