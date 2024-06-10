# Acoustic Scattering - Maze

**One line description of the data:** Simple acoustic wave propogation through maze-like structures.

**Longer description of the data:** These variable-coefficient acoustic equations describe the propogation of an acoustic pressure wave through maze-like domains. Pressure waves emerge from point sources and propogate through domains consisting of low density maze paths and orders of magnitude higher density maze walls. This is built primarily as a challenge for machine learning methods, though has similar properties to optimal placement problems like WiFi in a building. 

**Domain scientist**: [Michael McCabe](https://mikemccabe210.github.io/), Polymathic AI.

**Code or software used to generate the data**: Clawpack, adapted from: http://www.clawpack.org/gallery/pyclaw/gallery/acoustics_2d_interface.html

**Equation**:

```math
\begin{align}
\frac{ \partial p}{\partial t} + K(x, y) \left( \frac{\partial u}{\partial x} + \frac{\partial v}{\partial y} \right) &= 0 \\
\frac{ \partial u  }{\partial t} + \frac{1}{\rho(x, y)} \frac{\partial p}{\partial x} &= 0 \\
\frac{ \partial v  }{\partial t} + \frac{1}{\rho(x, y)} \frac{\partial p}{\partial v} &= 0 
\end{align}
```
with $\rho$ the material density, $u, v$ the velocity in the $x, y$ directions respectively, $p$ the pressure, and $K$ the bulk modulus. 

Example material densities can be seen below:

![image](https://users.flatironinstitute.org/~polymathic/data/the_well/datasets/acoustic_scattering_maze_2d/gif/mazes_density.png)

Traversal can be seen:

![Gif](https://users.flatironinstitute.org/~polymathic/data/the_well/datasets/acoustic_scattering_maze_2d/gif/pressure_normalized.gif)

# About the data

**Dimension of discretized data:** $201$ steps of $256^2$ images

**Fields available in the data:** p, u, v, $\rho$, $K$

**Number of trajectories:** 2000

**Estimated size of the ensemble of all simulations:** 297 GB

**Grid type:** Cartesian, uniform

**Initial conditions:** Flat pressure static field with 1-6 high pressure rings randomly placed along paths of maze. The rings are defined with variable intensity $\sim \mathcal U(3., 5.)$ and radius $\sim \mathcal U(.01, .04)$. Any overlap with walls is removed. 

**Boundary conditions:** Open domain in $y$, reflective walls in $x$

**Simulation time-step:** Variable based on CFL with safety factor .25. 

**Data are stored separated by ($\Delta t$):** 2/201. 

**Total time range ($t_{min}$ to $t_{max}$):** [0,4.]

**Spatial domain size ($L_x$, $L_y$, $L_z$):** [-1, 1] x [-1, 1]

**Set of coefficients or non-dimensional parameters evaluated:**

$K$ is fixed at 4.0. 

$\rho$ is the primary coefficient here. We generated a maze with initial width between 6 and 16 pixels and upsample it via nearest neighbor resampling to create a 256 x 256 maze. The walls are set to $\rho=10^6$ while paths are set to  $\rho=3$.  

Approximate time to generate the data: ~20 minutes per simulation. 

Hardware used to generate the data and precision used for generating the data: 64 Intel Icelake cores per simulation. Generated in double precision.

# What is interesting and challenging about the data:

This is an example of simple dynamics in complicated geometry. The sharp discontinuities can be a significant problem for machine learning models, yet they are a common feature in many real-world physics. While visually the walls appear to stop the signal, it is actually simply the case that the speed of sound is much much lower inside the walls leading to partial reflection/absorbtion at the interfaces. 