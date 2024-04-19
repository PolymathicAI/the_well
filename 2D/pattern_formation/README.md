# Pattern formation in the Gray-Scott reaciton-diffusion equations

**One line description of the data:** Stable Turing patterns emerge from randomness, with drastic qualitative differences in pattern dynamics depending on the equation parameters.

**Longer description of the data:** The Gray-Scott equations are a set of coupled reaction-diffusion equations describing two chemical species, $U$ and $V$, whose concentrations vary in space and time. The two parameters $F$ and $k$ control the “feed” and “kill” rates in the reaction. A zoo of qualitatively different static and dynamic patterns in the solutions are possible depending on these two parameters. There is a rich landscape of pattern formation hidden in these equations. 

**Associated paper**:

**Domain scientist**: Daniel Fortunato (Center for Computational Mathematics & Center for Computational Biology, Flatiron Institute)

**Code or software used to generate the data**: MATLAB R2023a, using the stiff PDE integrator implemented in Chebfun. The Fourier spectral method is used in space (with nonlinear terms evaluated pseudospectrally), and the exponential time-differencing fourth-order Runge-Kutta scheme (ETDRK4) is used in time.

**Equation describing the data** $
\begin{align}
\frac{\partial u}{\partial t} &= \delta_u\Delta u - uv^2 + F(1-u) \nonumber \\
\frac{\partial v}{\partial t} &= \delta_v\Delta v - uv^2 + (F+k)v \nonumber
\end{align}
$
The dimensionless parameters describing the behavior are: $F$, $k$, $\frac{\delta_u}{\delta_v}$
[ADD Description of variables, but should be good]

[ADD GIF OF THE SIMULATION]

# About the data

Dimension of discretized data: $128\times 128$ Fourier modes with $10,000$ time-steps [MODIFY AFTER DOWNSAMPLING]

Fields available in the data:
Two chemical species $u$ and $v$.

Number of trajectories: 6 sets of parameters, 200 initial conditions per set = 1200.

Estimated size of the ensemble of all simulations: 150GB

Grid type: uniform.

Initial conditions: Two types of initial conditions generated: random Fourier series and random clusters of Gaussians.

Boundary conditions: periodic.

Simulation time-step: 1 second.

Data are stored separated by ($\Delta t$): 10 seconds

Total time range ($t_{min}$ to $t_{max}$): $t_{min} =0$, $t_{max} = 10,000$.

Spatial domain size ($L_x$, $L_y$, $L_z$): $x,y\in[-1,1]$.

Set of coefficients or non-dimensional parameters evaluated: All simulations used $\delta_u = 2.10^{-5}$ and $\delta_v = 1.10^{-5}$.
"Gliders": $F = 0.014, k = 0.054$. "Bubbles": $F = 0.098, k =0.057$. "Maze": $F= 0.029, k = 0.057$. "Worms": $F= 0.058, k = 0.065$. "Spirals": $F=0.018, k = 0.051$. "Spots": $F= 0.03, k=0.062$.

Approximate time to generate the data: 5.5 hours per set of parameters, 33 hours total.

Hardware used to generate the data and precision used for generating the data: 40 CPU cores.

# What is interesting and challenging about the data:

What phenomena of physical interest are catpured in the data: Pattern formation: by sweeping the two parameters $F$ and $k$, a multitude of steady and dynamic patterns can form from random initial conditions.

How to evaluate a new simulator operating in this space: It would be impressive if a simulator—trained only on some of the patterns produced by a subset of the $(F, k)$ parameter space—could perform well on an unseen set of parameter values $(F, k)$ that produce fundamentally different patterns. Stability for steady-state patterns over long rollout times would also be impressive.
