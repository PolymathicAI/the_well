# Active fluid simulations

**One line description of the data:**  Modeling and simulation of biological active matter.

**Longer description of the data:** Simulation of a continuum theory describing the dynamics of $N$ rod-like active particles immersed in a Stokes fluid having linear dimension $L$ and colume $L^2$.

**Associated paper**: https://arxiv.org/abs/2308.06675

**Domain scientist**: Suryanarayana Maddu [ADD WEBSITE], CCB, Flatiron Institute. 

**Code or software used to generate the data**: https://github.com/SuryanarayanaMK/Learning_closures/tree/master

**Equation**: [ADD EQUATION]

[ADD GIF OF THE SIMULATION]

# About the data

**Dimension of discretized data:** $81$ time-steps of $256\times256$ images with 1 channel for concentration $c$, 2 channels for velocity {$U_x$, $U_y$}, 3 channels for the orientation tensor {$D_{xx}$, $D_{xy}$, $D_{yy}$}, and 3 channels for the strain-rate tensor {$E_{xx}$, $E_{xy}$, $E_{yy}$}.

**Fields available in the data:** {$c$, $U_x$, $U_y$, $D_{xx}$, $D_{xy}$, $D_{yy}$, $E_{xx}$, $E_{xy}$, $E_{yy}$, time}.

**Number of trajectories:** $5$ trajectories per parameter-set, each generated with a different initialization of the state field {$c,D,U$}.

**Size of the ensemble of all simulations:** 36GB.

**Grid type:** Uniform grid.

**Initial conditions:** The concentration is set to constant value $c(x,t)=1$ and the orientation tensor is initialized as plane-wave perturbation about the isotropic state.

**Boundary conditions:** Periodic boundary conditions.

**Simulation time-step:** $3.90625\times 10^{-4}$ seconds.

**Data are stored separated by ($\Delta t$):** 0.25 seconds.

**Total time range ($t_{min}$ to $t_{max}$):** $0$ to $20$ seconds.

**Spatial domain size ($L_x$, $L_y$, $L_z$):** $L\times L$ (2D spatial grid) with $L =10$. [ASK UNIT]

**Set of coefficients or non-dimensional parameters evaluated:** $\alpha = \{-1,-2,-3,-4,-5\}$; $\beta  = \{0.8\}$; 
$\zeta = \{1,3,5,7,9,11,13,15,17\}$; 

**Approximate time and hardware to generate the data:** 20 minutes per simulation on an A100 GPU in `fp64` precision. There is a total of 225 simulations, which is approximately 75 hours.

# What is interesting and challenging about the data:

**What phenomena of physical interest are catpured in the data:** How is energy being transferred between scales? How is vorticity coupled to the orientation field? Where does the transition from isotropic state to nematic state occur with the change in alignment ($\zeta$) or dipole strength ($\alpha$)? 


**How to evaluate a new simulator operating in this space:** Reproducing some summary statistics like power spectra and average scalar order parameters. Additionally, being able to accurately capture the phase transition from isotropic to nematic state.
