#!/usr/bin/env python
# encoding: utf-8
r"""
Two-dimensional variable-coefficient acoustics
==============================================

Solve the variable-coefficient acoustics equations in 2D:

.. math::
    p_t + K(x,y) (u_x + v_y) & = 0 \\
    u_t + p_x / \rho(x,y) & = 0 \\
    v_t + p_y / \rho(x,y) & = 0.

Here p is the pressure, (u,v) is the velocity, :math:`K(x,y)` is the bulk modulus,
and :math:`\rho(x,y)` is the density.

This example shows how to solve a problem with variable coefficients.
The left and right halves of the domain consist of different materials.
"""

from functools import partial

import numpy as np
from scipy.ndimage import gaussian_filter


def setup(
    kernel_language="Fortran",
    use_petsc=False,
    outdir="./_output",
    solver_type="classic",
    time_integrator="SSP104",
    lim_type=2,
    disable_output=False,
    num_cells=(256, 256),
    seed=None,
    include_splits=True,
    include_inclusions=True,
    T_max=2.0,
    num_steps=101,
):
    """
    Example python script for solving the 2d acoustics equations.
    """
    from clawpack import riemann

    if seed is None:
        seed = np.random.default_rng()
    if use_petsc:
        import clawpack.petclaw as pyclaw
    else:
        from clawpack import pyclaw

    if solver_type == "classic":
        solver = pyclaw.ClawSolver2D(riemann.vc_acoustics_2D)
        solver.dimensional_split = False
        solver.limiters = pyclaw.limiters.tvd.MC
    elif solver_type == "sharpclaw":
        solver = pyclaw.SharpClawSolver2D(riemann.vc_acoustics_2D)
        solver.time_integrator = time_integrator
        if time_integrator == "SSPLMMk2":
            solver.lmm_steps = 3
            solver.cfl_max = 0.25
            solver.cfl_desired = 0.24

    solver.bc_lower[0] = pyclaw.BC.wall
    solver.bc_upper[0] = pyclaw.BC.extrap
    solver.bc_lower[1] = pyclaw.BC.wall
    solver.bc_upper[1] = pyclaw.BC.extrap
    solver.aux_bc_lower[0] = pyclaw.BC.wall
    solver.aux_bc_upper[0] = pyclaw.BC.extrap
    solver.aux_bc_lower[1] = pyclaw.BC.wall
    solver.aux_bc_upper[1] = pyclaw.BC.extrap

    x = pyclaw.Dimension(-1.0, 1.0, num_cells[0], name="x")
    y = pyclaw.Dimension(-1.0, 1.0, num_cells[1], name="y")
    domain = pyclaw.Domain([x, y])

    num_eqn = 3
    num_aux = 2  # density, sound speed
    state = pyclaw.State(domain, num_eqn, num_aux)

    grid = state.grid
    X, Y = grid.p_centers
    is_vert = seed.integers(0, 2)
    midpoint = seed.uniform(-0.8, 0.8)
    rho_left = seed.uniform(0.2, 7)  # 4.0  # Density in left half
    rho_right = seed.uniform(0.2, 7)  # 1.0  # Density in right half
    bulk_left = 4.0  # Bulk modulus in left half
    bulk_right = 4.0  # Bulk modulus in right half

    def gaussian_bump(
        aux, mask, seed, rho_low=1, rho_high=7.0, sigma_low=0.1, sigma_high=5
    ):
        rho_bump = seed.uniform(rho_low, rho_high)
        rho_base = seed.uniform(rho_low, rho_high)

        Xmask = X[mask]
        xmax = Xmask.max()
        xmin = Xmask.min()
        Ymask = Y[mask]
        ymax = Ymask.max()
        ymin = Ymask.min()

        xc = seed.uniform(xmin, xmax)
        yc = seed.uniform(ymin, ymax)
        sigma = seed.uniform(sigma_low, sigma_high)
        rho = rho_base + (rho_bump - rho_base) * np.exp(
            -((Xmask - xc) ** 2 + (Ymask - yc) ** 2) / (sigma)
        )
        c = np.sqrt(bulk_left / rho)
        aux[0][mask] = rho
        aux[1][mask] = c

    def linear_gradient(aux, mask, seed, rho_low=1, rho_high=7.0):
        rho_x0 = seed.uniform(rho_low, rho_high)
        rho_x1 = seed.uniform(rho_low, rho_high)
        rho_y0 = seed.uniform(rho_low, rho_high)
        rho_y1 = seed.uniform(rho_low, rho_high)

        # Bilinearly interpolate between the four values
        Xmask = (X[mask] + 1) / 2
        xmax = Xmask.max()
        xmin = Xmask.min()
        Ymask = (Y[mask] + 1) / 2
        ymax = Ymask.max()
        ymin = Ymask.min()

        Xrel = (Xmask - xmin) / (xmax - xmin)
        Yrel = (Ymask - ymin) / (ymax - ymin)

        rho = (
            (1 - Xrel) * (1 - Yrel) * rho_x0
            + Xrel * (1 - Yrel) * rho_x1
            + (1 - Xrel) * Yrel * rho_y0
            + Xrel * Yrel * rho_y1
        )
        c = np.sqrt(bulk_left / rho)
        aux[0][mask] = rho
        aux[1][mask] = c

    def constant(aux, mask, seed, rho_low=1, rho_high=7.0):
        rho = seed.uniform(rho_low, rho_high)
        c = np.sqrt(bulk_left / rho)
        aux[0][mask] = rho
        aux[1][mask] = c

    def smoothed_gaussian_noise(
        aux, mask, seed, rho_low=1, rho_high=7.0, std=2, sigma_low=5, sigma_high=10
    ):
        rho = seed.uniform(rho_low, rho_high)
        background = seed.standard_normal(mask.shape)
        sigma = seed.uniform(sigma_low, sigma_high)

        background = gaussian_filter(background, sigma)
        rho = rho + background[mask]
        c = np.sqrt(bulk_left / rho)
        aux[0][mask] = rho
        aux[1][mask] = c

    gen_funcs = [gaussian_bump, linear_gradient, constant, smoothed_gaussian_noise]

    c_left = np.sqrt(bulk_left / rho_left)  # Sound speed (left)
    if include_splits:
        if is_vert:
            mask = Y < midpoint
        else:
            mask = X < midpoint
        seed.choice(gen_funcs)(state.aux, (~mask), seed)
    else:
        mask = np.ones_like(X, dtype=bool)
    seed.choice(gen_funcs)(state.aux, mask, seed)

    state.q[0, :, :] = 0.0
    state.q[1, :, :] = 0.0
    state.q[2, :, :] = 0.0
    # Set initial condition
    n_waves = seed.integers(1, 4)
    for i in range(n_waves):
        center = seed.uniform(-0.95, 0.95, 2)
        x0 = center[0]
        y0 = center[1]
        width = seed.uniform(0.05, 0.15)
        rad = seed.uniform(width + 0.01, 0.3)
        intensity = seed.uniform(0.5, 2.0)
        # x0 = -0.5; y0 = 0.
        r = np.sqrt((X - x0) ** 2 + (Y - y0) ** 2)
        # width = 0.1; rad = 0.25
        state.q[0, :, :] += (np.abs(r - rad) <= width) * (
            intensity + np.cos(np.pi * (r - rad) / width)
        )

    if include_inclusions:
        n_inclusions = seed.integers(0, 15)
        for i in range(n_inclusions):
            # Copied elipse code from
            g_ell_center = seed.uniform(-0.95, 0.95, 2)
            rads = seed.uniform(0.05, 0.6, 2)
            g_ell_width = rads[0]
            g_ell_height = rads[1]
            angle = seed.uniform(-45, 45)

            cos_angle = np.cos(np.radians(180.0 - angle))
            sin_angle = np.sin(np.radians(180.0 - angle))

            xc = X - g_ell_center[0]
            yc = Y - g_ell_center[1]

            xct = xc * cos_angle - yc * sin_angle
            yct = xc * sin_angle + yc * cos_angle

            rad_cc = (xct**2 / (g_ell_width / 2.0) ** 2) + (
                yct**2 / (g_ell_height / 2.0) ** 2
            )

            inclusion_rho = np.exp(seed.uniform(-1, 10))
            # r = np.sqrt((X-x0)**2 + (Y-y0)**2)
            c_left = np.sqrt(bulk_left / inclusion_rho)  # Sound speed (left)
            state.aux[0][rad_cc <= 1] = inclusion_rho
            state.aux[1][rad_cc <= 1] = c_left
            state.q[0][rad_cc <= 1] = 0.0

    claw = pyclaw.Controller()
    claw.keep_copy = True
    if disable_output:
        claw.output_format = None
    claw.solution = pyclaw.Solution(state, domain)
    claw.solver = solver
    claw.outdir = outdir
    claw.tfinal = T_max
    claw.num_output_times = num_steps
    claw.write_aux_init = True
    claw.setplot = setplot
    claw.output_options = {"format": "binary"}
    if use_petsc:
        claw.output_options = {"format": "binary"}

    return claw


def setplot(plotdata):
    """
    Plot solution using VisClaw.

    This example shows how to mark an internal boundary on a 2D plot.
    """

    from clawpack.visclaw import colormaps

    plotdata.clearfigures()  # clear any old figures,axes,items data

    # Figure for pressure
    plotfigure = plotdata.new_plotfigure(name="Pressure", figno=0)

    # Set up for axes in this figure:
    plotaxes = plotfigure.new_plotaxes()
    plotaxes.title = "Pressure"
    plotaxes.scaled = True  # so aspect ratio is 1
    plotaxes.afteraxes = mark_interface

    # Set up for item on these axes:
    plotitem = plotaxes.new_plotitem(plot_type="2d_pcolor")
    plotitem.plot_var = 0
    plotitem.pcolor_cmap = colormaps.yellow_red_blue
    plotitem.add_colorbar = True
    plotitem.pcolor_cmin = 0.0
    plotitem.pcolor_cmax = 1.0

    # Figure for x-velocity plot
    plotfigure = plotdata.new_plotfigure(name="x-Velocity", figno=1)

    # Set up for axes in this figure:
    plotaxes = plotfigure.new_plotaxes()
    plotaxes.title = "u"
    plotaxes.afteraxes = mark_interface

    plotitem = plotaxes.new_plotitem(plot_type="2d_pcolor")
    plotitem.plot_var = 1
    plotitem.pcolor_cmap = colormaps.yellow_red_blue
    plotitem.add_colorbar = True
    plotitem.pcolor_cmin = -0.3
    plotitem.pcolor_cmax = 0.3

    return plotdata


def mark_interface(current_data):
    import matplotlib.pyplot as plt

    plt.plot((0.0, 0.0), (-1.0, 1.0), "-k", linewidth=2)


if __name__ == "__main__":
    from clawpack.pyclaw.util import run_app_from_main

    setup_wrapped = partial(setup, seed=np.random.default_rng(42))
    output = run_app_from_main(setup_wrapped, setplot)
