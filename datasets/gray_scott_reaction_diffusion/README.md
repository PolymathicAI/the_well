# Pattern formation in the Gray-Scott reaction-diffusion equations

**One line description of the data:** Stable Turing patterns emerge from randomness, with drastic qualitative differences in pattern dynamics depending on the equation parameters.

**Longer description of the data:** The Gray-Scott equations are a set of coupled reaction-diffusion equations describing two chemical species, $A$ and $B$, whose concentrations vary in space and time. The two parameters $f$ and $k$ control the “feed” and “kill” rates in the reaction. A zoo of qualitatively different static and dynamic patterns in the solutions are possible depending on these two parameters. There is a rich landscape of pattern formation hidden in these equations. 

**Associated paper**: None.

**Domain expert**: [Daniel Fortunato](https://danfortunato.com/), CCM and CCB, Flatiron Institute.

**Code or software used to generate the data**: [Github repository](https://github.com/danfortunato/spectral-gray-scott) (MATLAB R2023a, using the stiff PDE integrator implemented in Chebfun. The Fourier spectral method is used in space (with nonlinear terms evaluated pseudospectrally), and the exponential time-differencing fourth-order Runge-Kutta scheme (ETDRK4) is used in time.)

**Equation describing the data** 
```math
\begin{align}
\frac{\partial A}{\partial t} &= \delta_A\Delta A - AB^2 + f(1-A) \nonumber \\\frac{\partial B}{\partial t} &= \delta_B\Delta B - AB^2 - (f+k)B \nonumber
\end{align}
```
The dimensionless parameters describing the behavior are: $f$ the rate at which $A$ is replenished (feed rate), $k$ the rate at which $B$ is removed from the system, and  $\delta_A, \delta_B$ the diffusion coefficients of both species.


![Gif](https://users.flatironinstitute.org/~polymathic/data/the_well/datasets/pattern_formation/gif/concentration_A_normalized.gif)

| Dataset    | FNO | TFNO  | Unet | CNextU-net
|:-:|:-:|:-:|:-:|:-:|
| `pattern_formation` | 0.178  | 0.177 | 0.322|$\mathbf{0.0512}$|

Preliminary benchmarking, in VRMSE.

# About the data

**Dimension of discretized data:** 1001 time-steps of $128\times 128$ images.

**Fields available in the data:** Two chemical species $A$ and $B$.

**Number of trajectories:** 6 sets of parameters, 200 initial conditions per set = 1200.

**Estimated size of the ensemble of all simulations:** 153.8 GB.

**Grid type:** uniform, cartesian coordinates.

**Initial conditions:** Two types of initial conditions generated: random Fourier series and random clusters of Gaussians.

**Boundary conditions:** periodic.

**Simulation time-step:** 1 second.

**Data are stored separated by ($\Delta t$):** 10 seconds.

**Total time range ($t_{min}$ to $t_{max}$):** $t_{min} =0$, $t_{max} = 10,000$.

**Spatial domain size ($L_x$, $L_y$, $L_z$):** $x,y\in[-1,1]$.

**Set of coefficients or non-dimensional parameters evaluated:** All simulations used $\delta_u = 2.10^{-5}$ and $\delta_v = 1.10^{-5}$.
"Gliders": $f = 0.014, k = 0.054$. "Bubbles": $f = 0.098, k =0.057$. "Maze": $f= 0.029, k = 0.057$. "Worms": $f= 0.058, k = 0.065$. "Spirals": $f=0.018, k = 0.051$. "Spots": $f= 0.03, k=0.062$.

**Approximate time to generate the data:** 5.5 hours per set of parameters, 33 hours total.

**Hardware used to generate the data:** 40 CPU cores.

# What is interesting and challenging about the data:

**What phenomena of physical interest are catpured in the data:** Pattern formation: by sweeping the two parameters $f$ and $k$, a multitude of steady and dynamic patterns can form from random initial conditions.

**How to evaluate a new simulator operating in this space:** It would be impressive if a simulator—trained only on some of the patterns produced by a subset of the $(f, k)$ parameter space—could perform well on an unseen set of parameter values $(f, k)$ that produce fundamentally different patterns. Stability for steady-state patterns over long rollout times would also be impressive.
