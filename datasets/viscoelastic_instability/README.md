# Multistability of viscoelastic fluids in a 2D channel flow

**One line description of the data:** Multistability in viscoelastic flows, i.e. four different attractors (statistically stable states) are observed for the same set of parameters depending on the initial conditions.

**Longer description of the data:** Elasto-inertial turbulence (EIT) is a recently discovered two-dimensional chaotic flow state observed in dilute polymer solutions. Two-dimensional direct numerical simulations show (up to) four coexistent attractors: the laminar state (LAM), a steady arrowhead regime (SAR), Elasto-inertial turbulence (EIT) and a ‘chaotic arrowhead regime’ (CAR). The SAR is stable for all parameters considered here, while the final pair of (chaotic) flow states are visually very similar and can be distinguished only by the presence of a weak polymer arrowhead structure in the CAR regime. Both chaotic regimes are maintained by an identical near-wall mechanism and the weak arrowhead does not play a role. The data set includes snapshots on the four attractors as well as two edge states. An edge state is an unstable state that exists on the boundary between two basins of attractors, the so-called edge manifold. Edge states have a single unstable direction out of the manifold and are relevant since the lie exactly on the boundary separating qualitatively different behaviours of the flow. The edge states in the present data set are obtained through edge tracking between the laminar state and EIT and between EIT and SAR.

**Associated paper**: [Paper](https://www.cambridge.org/core/services/aop-cambridge-core/content/view/D63B7EDB638451A6FC2FBBFDA85E1BBD/S0022112024000508a.pdf/multistability-of-elasto-inertial-two-dimensional-channel-flow.pdf).

**Domain experts**: [Miguel Beneitez](https://beneitez.github.io/) and [Richard Kerswell](https://www.damtp.cam.ac.uk/user/rrk26/), DAMTP, University of Cambridge, UK.

**Code or software used to generate the data**: [Dedalus](https://dedalus-project.readthedocs.io/en/latest/index.html).

**Equation**:

```math
\begin{align*}
Re(\partial_t \mathbf{u^*} + (\mathbf{u^*}\cdot\nabla)\mathbf{u^*} ) + \nabla p^* &= \beta \Delta \mathbf{u^*} + (1-\beta)\nabla\cdot \mathbf{T}(\mathbf{C^*}),\\
\partial_t \mathbf{C^*} + (\mathbf{u^*}\cdot\nabla)\mathbf{C^*} +\mathbf{T}(\mathbf{C^*}) &= \mathbf{C^*}\cdot\nabla \mathbf{u^*} + (\nabla \mathbf{u^*})^T \cdot \mathbf{C^*} + \epsilon \Delta \mathbf{C^*}, \\
\nabla \mathbf{u^*} &= 0,\\
\textrm{with} \quad \mathbf{T}(\mathbf{C^*}) &= \frac{1}{\text{Wi}}(f(\textrm{tr}(\mathbf{C^*}))\mathbf{C^*} - \mathbf{I}),\\
\textrm{and} \quad f(s) &:= \left(1- \frac{s-3}{L^2_{max}}\right)^{-1}.
\end{align*}
```

where $\mathbf{u^*} = (u^*,v^*)$ is the streamwise and wall-normal velocity components, $p^*$ is the pressure, $\mathbf{C^*}$ is the positive definite conformation tensor which represents the ensemble average of the produce of the end-to-end vector of the polymer molecules. In 2D, 4 components of the tensor are solved: $c_{xx}^{*}, c^{*}_{yy}, c^{*}_{zz}, c^{*}_{xy}$. $\mathbf{T}(\mathbf{C^{*}})$ is the polymer stress tensor given by the FENE-P model.


![Gif](https://users.flatironinstitute.org/~polymathic/data/the_well/datasets/viscoelastic_instability/gif/czz_normalized.gif)

| Dataset    | FNO | TFNO  | Unet | CNextU-net
|:-:|:-:|:-:|:-:|:-:|
| `viscoelastic_instability` | 0.7212 | 0.7102 | 0.4185 | $\mathbf{0.2499}$ |

Table: VRMSE metrics on test sets (lower is better). Best results are shown in bold. VRMSE is scaled such that predicting the mean value of the target field results in a score of 1.

## About the data

**Dimension of discretized data:**
- EIT: 34 trajectories with 60 timesteps, 512x512 images (chaotic solution).
- CAR: 39 trajectories with 60 timesteps, 512x512 images (chaotic solution).
- SAR: 20 trajectories with 20 timesteps, 512x512 images (simple periodic solutions).
- Transition to chaos between EIT and SAR: 36 snapshots with 20 timesteps of 512x512 images.
- Transition to non-chaotic state between EIT and SAR: 38 snapshots with 20 timesteps of 512x512 images.
- Transition to chaos between EIT and Laminar: 43 snapshots with 20 timesteps of 512x512 images.
- Transition to non-chaotic state between EIT and Laminar: 49 snapshots with 20 timesteps of 512x512 images.

**Fields available in the data:** pressure (scalar field), velocity (vector field), positive conformation tensor ( $c_{xx}^{*}, c^{*}_{yy},, c^{*}_{xy}$ are in tensor fields, $c^{*}_{zz}$ in scalar fields).

**Number of trajectories:** 260 trajectories.

**Estimated size of the ensemble of all simulations:** 66 GB.

**Grid type:** uniform cartesian coordinates.

**Initial conditions:**
- Edge trajectory: linear interpolation between a chaotic and a non-chaotic state.
- SAR: continuation of the solution obtained through a linear instability at a different parameter set using time-stepping.
- EIT: laminar state + blowing and suction at the walls.
- CAR: SAR + blowing and suction at the walls.

**Boundary conditions:** no slip conditions for the velocity $(u^*,v^*)=(0,0)$ at the wall and $\epsilon=0$ at the wall for the equation for $\mathbf{C^*}$.

**Simulation time-step:** various in the different states, but typically $\sim 10^{-4}$.

**Data are stored separated by ($\Delta t$):** various at different states, but typically 1.

**Total time range ($t_{min}$ to $t_{max}$):** depends on the simulation.

**Spatial domain size ($L_x$, $L_y$, $L_z$):** $0 \leq x \leq 2\pi$, $-1 \leq y \leq 1$.

**Set of coefficients or non-dimensional parameters evaluated:** Reynold number $Re=1000$, Weissenberg number $Wi = 50$, $\beta =0.9$, $\epsilon=2.10^{-6}$, $L_{max}=70$.

**Approximate time to generate the data:** 3 months to generate all the data. It takes typically 1 day to generate $\sim 50$ snapshots.

**Hardware used to generate the data:** typically 32 or 64 cores.

## What is interesting and challenging about the data:

**What phenomena of physical interest are catpured in the data:** The phenomena of interest in the data is: (i) chaotic dynamics in viscoelastic flows in EIT and CAR. Also note that they are separate states. (ii) multistability for the same set of parameters, the flow has four different behaviours depending on the initial conditions.

**How to evaluate a new simulator operating in this space:**
A new simulator would need to capture EIT/CAR adequately for a physically relevant parameter range.

Please cite the associated paper if you use this data in your research:

```
@article{beneitez2024multistability,
  title={Multistability of elasto-inertial two-dimensional channel flow},
  author={Beneitez, Miguel and Page, Jacob and Dubief, Yves and Kerswell, Rich R},
  journal={Journal of Fluid Mechanics},
  volume={981},
  pages={A30},
  year={2024},
  publisher={Cambridge University Press}
}
```
