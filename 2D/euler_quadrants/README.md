# Euler equations - Riemann problems

**One line description of the data:**  Evolution starting with piecewise constant initial data in quadrants, for different gases.

**Longer description of the data:**  The evolution can give rise to shocks, rarefaction waves, contact discontinuities, interaction witheach other and domain walls.

**Associated paper**: [ADD PAPER]

**Domain scientist**: Marsha Berger [ADD WEBSITE], Flatiron Institute and NYU.

**Code or software used to generate the data**: Clawpack [ADD WEBSITE]

[ADD GIF OF THE SIMULATION]

# About the data

Dimension of discretized data: 512x512 images with 100 timesteps.

Fields available in the data: ['density', 'momentum_x', 'momentum_y', 'energy', 'pressure']

Number of trajectories: 500 per set of parameters, 10,000 in total.

Estimated size of the ensemble of all simulations: 4.9Tb.

Grid type: uniform grid.

Initial conditions: [MIKE ADD DESCRIPTION OF THE INITIAL CONDITIONS]

Boundary conditions: ['extrap', 'periodic']

Simulation time-step: variable.

Data are stored separated by ($\Delta t$): 1.5s /100 timesteps.

Total time range ($t_{min}$ to $t_{max}$): $t_{min} = 0$, $t_{max}=1.5s$

Spatial domain size ($L_x$, $L_y$, $L_z$): $L_x = 1$, $L_y = 1$ [TO BE CONFIRMED]

Set of coefficients or non-dimensional parameters evaluated: $\gamma$ constant of the gas: $\gamma=\{1.3,1.4,1.13,1.22,1.33,1.76, 1.365,1.404,1.453,1.597\}$ and boundary conditions: {extrap, periodic}.

Approximate time to generate the data: 80 hours on 160 CPU cores for all data

Hardware used to generate the data and precision used for generating the data: Icelake nodes, double precision.

# What is interesting and challenging about the data:

What phenomena of physical interest are catpured in the data: captue the shock formations and interactions.

How to evaluate a new simulator operating in this space: the new simulator should predict the shock at the right location and time, and the right shock strength, as compared to a ‘pressure’ gauge monitoring the ‘exact’ solution.
