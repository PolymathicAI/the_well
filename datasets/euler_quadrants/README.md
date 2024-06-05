# Euler equations - Riemann problems (compressible, inviscid fluid)

**One line description of the data:**  Evolution of different gases starting with piecewise constant initial data in quadrants.

**Longer description of the data:**  The evolution can give rise to shocks, rarefaction waves, contact discontinuities, interaction with each other and domain walls.

**Associated paper**: [Paper](https://epubs.siam.org/doi/pdf/10.1137/S1064827595291819?casa_token=vkASCwD4WngAAAAA:N0jy0Z6tshitF10_YRTlZzU-P7mAiPFr3v58sw7pmRsZOarAi824-b1CWhOQts1rvaG3YpJisw)

**Domain scientist**: [Marsha Berger](https://cs.nyu.edu/~berger/), Flatiron Institute & NYU.

**Code or software used to generate the data**: [Clawpack (AMRClaw)](http://www.clawpack.org/)

**Equation**:Euler equations for a compressible gas:
$\begin{align}
U_t + F(U)_x + G(U)_y &= 0 \nonumber\\
\textrm{where} \quad U = \begin{bmatrix} \rho \nonumber\\ \rho u \\ \rho v \\ e \end{bmatrix}, \quad F(U) = \begin{bmatrix} \rho u \\ \rho u^2 + p \\ \rho u v \\ u(e + p) \end{bmatrix},& \quad G(U) = \begin{bmatrix} \rho v \\ \rho u v \\ \rho v^2 + p \\ v(e + p) \end{bmatrix}, \quad \\ e = \frac{p}{(\gamma - 1)} + \frac{\rho (u^2 + v^2)}{2}&, \quad p = A\rho^{\gamma}. \nonumber
\end{align}$
with $\rho$ the density, $u$ and $v$ the $x$ and $y$ velocity components, $e$ the energy, $p$ the pressure, $\gamma$ the gas constant, and $A>0$ is a function of entropy.


[ADD GIF OF THE SIMULATION]

# About the data

Dimension of discretized data: 100 timesteps of 512x512 images.

Fields available in the data: ['density', 'momentum_x', 'momentum_y', 'energy', 'pressure']

Number of trajectories: 500 per set of parameters, 10,000 in total.

Estimated size of the ensemble of all simulations: 4.9TB.

Grid type: uniform grid.

Initial conditions: Randomly generated initial quadrants.

Boundary conditions: ['extrap', 'periodic']

Simulation time-step: variable.

Data are stored separated by ($\Delta t$): 0.015s (1.5s for 100 timesteps).

Total time range ($t_{min}$ to $t_{max}$): $t_{min} = 0$, $t_{max}=1.5s$

Spatial domain size ($L_x$, $L_y$, $L_z$): $L_x = 1$, $L_y = 1$ [TO BE CONFIRMED]

Set of coefficients or non-dimensional parameters evaluated: all combinations of $\gamma$ constant of the gas: $\gamma=\{1.3,1.4,1.13,1.22,1.33,1.76, 1.365,1.404,1.453,1.597\}$ and boundary conditions: {extrap, periodic}.

Approximate time to generate the data: 80 hours on 160 CPU cores for all data

Hardware used to generate the data and precision used for generating the data: Icelake nodes, double precision.

# What is interesting and challenging about the data:

What phenomena of physical interest are catpured in the data: capture the shock formations and interactions. Multiscale shocks.

How to evaluate a new simulator operating in this space: the new simulator should predict the shock at the right location and time, and the right shock strength, as compared to a ‘pressure’ gauge monitoring the ‘exact’ solution.
