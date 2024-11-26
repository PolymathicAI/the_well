# Helmholtz equation on a 2D staircase

**One line description of the data:** First high-order accurate solution of
acoustic scattering from a nonperiodic source by a periodic surface, relevant for its use in waveguide applications (antennae, diffraction from gratings, photonic/phononic crystals, noise cancellation, seismic filtering, etc.).

**Longer description of the data:**  Accurate solution of PDEs near infinite, periodic boundaries poses a numerical challenge due these surfaces serving as waveguides, allowing modes to propagate for long distances from the source. This property makes numerical truncation of the (infinite) solution domain unfeasible, as it would induce large artificial reflections and therefore errors. Periodization (reducing the computational domain to one unit cell) is only possible if the incident wave is also
periodic, such as plane waves, but not for nonperiodic sources, e.g. a point source. Computing a high-order accurate scattering solution from a point source, however, would be of scientific interest as it models applications such as remote sensing, diffraction from gratings, antennae, or acoustic/photonic metamaterials. We use a combination of the Floquet—Bloch transform (also known as array scanning method) and boundary integral equation methods to alleviate these challenges and recover the scattered solution as an integral over a family of quasiperiodic solutions parameterized by their on-surface wavenumber. The advantage of this approach is that each of the quasiperiodic solutions may be computed quickly by periodization, and accurately via high-order quadrature.

**Associated paper**: [Paper](https://arxiv.org/abs/2310.12486).

**Domain expert**: [Fruzsina Julia Agocs](https://fruzsinaagocs.github.io/), Center for Computational Mathematics, Flatiron Institute \& University of Colorado, Boulder.

**Code or software used to generate the data**: [Github repository](https://www.github.com/fruzsinaagocs/bies).

**Equations**:

While we solve equations in the frequency domain, the original time-domain problem is:

$$
\frac{\partial^2 U(t, \mathbf{x})}{\partial t^2} - \Delta U(t, \mathbf{x}) = \delta(t)\delta(\mathbf{x} - \mathbf{x}_0),
$$

where $\Delta = \nabla \cdot \nabla$ is the spatial Laplacian and $U$ the accoustic pressure. The sound-hard boundary $\partial \Omega$ imposes Neumann boundary conditions,

$$ U_n(t, \mathbf{x}) = \mathbf{n} \cdot \nabla U = 0, \quad t \in \mathbb{R}, \quad \mathbf{x} \in \partial \Omega. $$

Upon taking the temporal Fourier transform, we get the inhomogeneous Helmholtz Neumann boundary value problem

$$
\begin{align*}
-(\Delta + \omega^2)u &= \delta_{\mathbf{x}_0}, \quad \text{in } \Omega,\\
u_n &= 0 \quad \text{on } \partial \Omega,
\end{align*}
$$

with outwards radiation conditions as described in [1]. The region $\Omega$ lies above a corrugated boundary $\partial \Omega$, extending with spatial period $d$ in the $x_1$ direction, and is unbounded in the positive $x_2$ direction. The current example is a right-angled staircase whose unit cell consists of two equal-length line segments at $\pi/2$ angle to each other.

![Gif](https://users.flatironinstitute.org/~polymathic/data/the_well/datasets/helmholtz_staircase/gif/pressure_normalized.gif)

| Dataset    | FNO | TFNO  | Unet | CNextU-net
|:-:|:-:|:-:|:-:|:-:|
| `helmholtz_staircase`  |$\textbf{0.00046}$ | 0.00346 | 0.01931 | 0.02758|

Table: VRMSE metrics on test sets (lower is better). Best results are shown in bold. VRMSE is scaled such that predicting the mean value of the target field results in a score of 1.

## About the data

**Dimension of discretized data:** 50 time-steps of
1024 $\times$ 256 images.

**Fields available in the data:**
real and imaginary part of accoustic pressure (scalar field), the staircase mask (scalar field, stationary).

**Number of trajectories:** $512$ (combinations of $16$ input parameter $\omega$ and $32$ source positions $\mathbf{x}_0$).

**Size of the ensemble of all simulations:** 52.4 GB.

**Grid type:** uniform.

**Initial conditions:** The time-dependence is
analytic in this case: $U(t, \mathbf{x}) = u(\mathbf{x})e^{-i\omega t}.$ Therefore any spatial solution may serve as an initial condition.

**Boundary conditions:** Neumann conditions (normal
derivative of the pressure $u$ vanishes, with the normal defined as pointing up from
the boundary) are enforced at the boundary.

**Simulation time-step:** continuous in time (time-dependence is
analytic).

**Data are stored separated by ($\Delta t$):** $\Delta t =\frac{2\pi}{\omega N}$, with $N = 50$.

**Total time range ($t_{min}$ to $t_{max}$):** $t_{\mathrm{min}} = 0$, $t_{\mathrm{max}} =
\frac{2\pi}{\omega}$.

**Spatial domain size ($L_x$, $L_y$, $L_z$):** $-8.0 \leq x_1 \leq 8.0$ horizontally, and $-0.5 \geq x_2 \geq 3.5$ vertically.

**Set of coefficients or non-dimensional parameters evaluated:** $\omega$={0.06283032, 0.25123038, 0.43929689, 0.62675846, 0.81330465, 0.99856671, 1.18207893, 1.36324313, 1.5412579, 1.71501267, 1.88295798, 2.04282969, 2.19133479, 2.32367294, 2.4331094,  2.5110908}, with the sources coordinates being all combinations of $x$={-0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4} and $y$={-0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4}.

**Approximate time to generate the data:** per input parameter: $\sim 400s$, total: $\sim 50$ hours.

**Hardware used to generate the data:** 64 CPU cores.

## What is interesting and challenging about the data:

**What phenomena of physical interest are captured in the data:** The simulations capture the existence of trapped acoustic waves – modes that are guided along the corrugated surface. They also show that the on-surface wavenumber of trapped modes is different than the frequency of the incident radiation, i.e. they capture the trapped modes’ dispersion relation.

**How to evaluate a new simulator operating in this space:**
The (spatial) accuracy of a new simulator/method could be checked by requiring that it conserves flux – whatever the source injects into the system also needs to come out. The trapped modes’ dispersion relation may be another metric, my method generates this to 7-8 digits of accuracy at the moment, but 10-12 digits may also be obtained. The time-dependence learnt by a machine learning algorithm can be compared to the analytic solution $e^{-i\omega t}$, this can be used to evaluate temporal accuracy.

Please cite the associated paper if you use this data in your research:

```
@article{agocs2023trapped,
  title={Trapped acoustic waves and raindrops: high-order accurate integral equation method for localized excitation of a periodic staircase},
  author={Agocs, Fruzsina J and Barnett, Alex H},
  journal={arXiv preprint arXiv:2310.12486},
  year={2023}
}
```
