# Rayleigh-Taylor instability

**One line description of the data:** Effect of spectral shape and component phases on development of Rayleigh-Taylor turbulence.

**Longer description of the data:** We consider the Rayleigh-Taylor instability for a range of Atwood numbers and initial perturbations, all of which have a log—normal horizontal energy spectrum with random phase. The dataset examines how varying the mean, standard deviation and the disparity of the random phase effects the transition to and statistics of the ensuing turbulent flow. 

**Associated paper**:

**Domain scientist**: Stefan Nixon, University of Cambridge.

**Code or software used to generate the data**: [TurMix3D](https://theses.hal.science/tel-00669707/document)

**Equation**: [ADD EQUATION]

[ADD GIF OF THE SIMULATION]

# About the data

Dimension of discretized data: 60 snapshots of 128x128x128 cubes.

Fields available in the data: Density $\rho$, velocity in $x,y,z$ directions.

Number of trajectories: 45 trajectories

Estimated size of the ensemble of all simulations: 124 GB

Grid type: uniform grid.

Initial conditions: Initial conditions have been set by imposing a log—normal profile for the shape of energy spectrum in wavenumber space, such that:
$$A(k) = \frac{1}{k\sigma\sqrt{2\pi}} \exp\Big(-\frac{(\ln (k) - \mu)^2}{2\sigma^2}\Big) \quad\textrm{with}\quad k = \sqrt{k^2_x+k^2_y}$$

where $\mu$ is the mean and $\sigma$ is the standard deviation of the profile. Furthermore, we have imposed a random phase to the corresponding complex Fourier component (i.e. a random value for the argument of the complex Fourier component) between zero and a varied maximum ($\phi_{max}$), finally after Fourier transforming to physical space the mean of the resulting profile is normalized to $3.10^5$ to ensure comparable power. 


Boundary conditions: Periodic boundary conditions on sides walls and slip conditions on the top and bottom walls.

Simulation time-step: $\Delta t$ is set such that the maximum Courant number is $\frac12(CFL_{max}=0.5)$. Therefore, the time step decreases as the flow accelerates.

Data are stored separated by ($\Delta t$): The time difference between frames varies as the flow accelerates, thus the largest occur at the beginning of the simulation ($\delta t \sim 5s$) and the smallest at the end ($\delta t \sim 0.1s$).

Total time range ($t_{min}$ to $t_{max}$): Varies from $t_{min}=0$ to $t_{max}$ between $\sim 30s$ and $\sim 100s$, deoending on Atwood number (see CSV for details)

Spatial domain size ($L_x$, $L_y$, $L_z$): $[0,1]\times[0,1]\times[0,1]$.

Set of coefficients or non-dimensional parameters evaluated: We run simulations with 13 different initializations for five different Atwood number $At\in{\frac34, \frac12, \frac14, \frac18, \frac{1}{16}}$. The first set on initial conditions considers varying the mean $\mu$ and standard deviation $\sigma$ of the profile $A(k)$ with $\mu\in{1, 4, 16}$ and $\sigma\in{\frac14, \frac12, 1}$, the phase (argument of the complex Fourier component) $\phi$ was set randomly in the range $[0,2\pi)$. The second set of initial conditions considers a fixed mean ($\mu=16$) and standard deviation ($\sigma =0.25$) and a varieed range of random phases (complex arguments $\phi\in[0,\phi_{max})$) given to each Fourier component. The four cases considered are specified by $\phi_{max}\in \{ \frac{\pi}{128}, \frac{\pi}{8}, \frac{\pi}{2}, \pi\}$. 

Approximate time to generate the data: 1 hour on 128 CPU cores for 1 simulation. 65 hours on 128 CPU cores for all simulations.

Hardware used to generate the data and precision used for generating the data: 128 CPU core on the Ocre supercomputer at CEA, Bruyères-le-Châtel, France.

# What is interesting and challenging about the data:

What phenomena of physical interest are catpured in the data: In this data there are three key aspects of physical interest. Firstly, impact of coherence on otherwise random initial conditions. Secondly, the effect of the shape of the initial energy spectrum on the structure of the flow. Finally, the transition from the Boussinesq to the non-Boussinesq regime where the mixing width transitions from symmetric to asymmetric growth.  

How to evaluate a new simulator operating in this space: [TO TIDY] From a fundamental standpoint we, would expect the density field to be advected and mixed rather than created or destroyed to give appropriate statistics. From a qualitative perspective, given that the underlying simulations are of comparable spatial resolution to the simulations run by the alpha group (Dimonte et. al. 2003) we would consider a good emulator to produce a comparable value for α as reported in their paper for an appropriately similar set of initial conditions. This parameter is derived by considering the flow after the initial transient. At this stage, the width of the turbulent mixing zone, L, is self-similar and grows as L=α At g t^2  . They reported a value of α=0.025±0.003. In addition, during this self-regime, we would expect to observe energy spectra with a similar shape to those reported in Cabot and Cook 2006, specifically exhibiting an appropriate k^(-5/3) cascade. From a structural perspective, we would expect that for an initialization with a large variety of modes in the initial spectrum to observe a range of bubbles and spikes (upward and downward moving structures), whereas in the other limit (where this only on mode in the initial spectrum) we expect to observe a single bubble and spike.  In addition, a good emulator would exhibit symmetric mixing with for low Atwood numbers in the Boussinesq regime (defined as At < 0.1 by Andrews and Dalziel 2010) and asymmetries in the mixing with for large Atwood number.  
