"""
Adapted from Dedalus SWE example:

https://dedalus-project.readthedocs.io/en/latest/pages/examples/ivp_sphere_shallow_water.html
"""

import argparse
import logging
import multiprocessing as mp
import os
from glob import glob

import dedalus.public as d3
import numpy as np

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["NUMEXPR_MAX_THREADS"] = "1"

logger = logging.getLogger(__name__)


# Simulation units
meter = 1 / 6.37122e6
hour = 1
second = hour / 3600
day = hour * 24
year = (
    hour * 1008
)  # 42 day years - Chosen based on the fact that the sim gets boring ~ 4000 hours


def run_ic_file(ic_file, output_dir):
    output_dir = output_dir + ic_file.split("/")[-1].split(".")[0]
    # Parse IC file for PM xfer
    print(output_dir)
    # Parameters
    Nphi = 512
    Ntheta = 256
    dealias = 3 / 2
    R = 6.37122e6 * meter
    Omega = 7.292e-5 / second
    nu = 1e5 * meter**2 / second / (160) ** 2  # Hyperdiffusion matched at ell=96
    g = 9.80616 * meter / second**2
    timestep = 60 * second
    burn_in = 0.25 * year
    stop_sim_time = burn_in + 3 * year  # 1*year
    dtype = np.float64

    # Bases
    coords = d3.S2Coordinates("phi", "theta")
    dist = d3.Distributor(coords, dtype=dtype)
    basis = d3.SphereBasis(
        coords, (Nphi, Ntheta), radius=R, dealias=dealias, dtype=dtype
    )
    # Fields
    u = dist.VectorField(coords, name="u", bases=basis)
    h = dist.Field(name="h", bases=basis)
    # Substitutions
    zcross = lambda A: d3.MulCosine(d3.skew(A))  # noqa: E731

    # Copy ICs from hpa 500 fields
    ICs = np.load(ic_file)
    ICs = np.swapaxes(ICs, 1, 2)
    ICs = np.flip(ICs, 2)
    u0 = ICs[:2] * meter / second  # * .3
    h0 = ICs[2] * meter / g  # Conversion from geopotential to gp height
    H = h0.mean()  # Should be about 5500 meters
    h0 = h0 - H
    hs0 = ICs[3] * meter - H
    hs = dist.Field(name="hs", bases=basis)
    hs.load_from_global_grid_data(hs0)
    hs.low_pass_filter((128, 256))
    u.load_from_global_grid_data(u0)
    h.load_from_global_grid_data(h0)

    # # Initial conditions: balanced height
    c = dist.Field(name="c")
    problem = d3.LBVP([h, c], namespace=locals())
    problem.add_equation("g*lap(h) + c = - div(u@grad(u) + 2*Omega*zcross(u))")
    problem.add_equation("ave(h) = 0")

    solver_init = problem.build_solver()
    # solver_init.solve()

    # Momentum forcing - seasonal
    def find_center(t):
        time_of_day = t / day
        time_of_year = t / year
        max_declination = 0.4  # Truncated from estimate of earth's solar decline
        lon_center = time_of_day * 2 * np.pi  # Rescale sin to 0-1 then scale to np.pi
        lat_center = np.sin(time_of_year * 2 * np.pi) * max_declination
        lon_anti = np.pi + lon_center
        return lon_center, lat_center, lon_anti, lat_center

    def season_day_forcing(phi, theta, t, h_f0):
        phi_c, theta_c, phi_a, theta_a = find_center(t)
        sigma = 2 * np.pi / 3
        # Coefficients aren't super well-designed - idea is one side of the planet increases
        # the other side decreases and the effect is centered around a seasonally-shifting Gaussian.
        # The original thought was to have this act on momentum, but this was harder to implement in a stable way
        # since increasing/decreasing by same factor is net energy loss.
        coefficients = np.cos(phi - phi_c) * np.exp(
            -((theta - theta_c) ** 2) / sigma**2
        )
        # coefficients = np.exp(-(phi - phi_c)**2 / sigma) * np.exp(-(theta-theta_c)**2 / sigma**2)

        forcing = h_f0 * coefficients
        return forcing

    phi, theta = dist.local_grids(basis)
    t = dist.Field(name="t")
    lat = np.pi / 2 - theta + 0 * phi
    phi_var = dist.Field(name="phi_var", bases=basis)
    phi_var["g"] += phi
    theta_var = dist.Field(name="theta_var", bases=basis)
    theta_var["g"] += lat
    h_f0 = (
        2 * meter
    )  # Increasing this starts leading to fast waves (or maybe it just looks that way at 60 FPS/ 2.x day per sec)
    h_f = season_day_forcing(phi_var, theta_var, t, h_f0)

    # Problem
    problem = d3.IVP([u, h], namespace=locals(), time=t)
    problem.add_equation(
        "dt(u) + nu*lap(lap(u)) + g*grad(h)  + 2*Omega*zcross(u) = - u@grad(u)"
    )
    problem.add_equation("dt(h) + nu*lap(lap(h)) + (H)*div(u) = - div(u*(h-hs)) + h_f")
    # Init to remove fast waves in sim - should probably just filter in time here, but this works.
    solver = problem.build_solver(d3.RK222)
    solver.stop_sim_time = burn_in
    CFL = d3.CFL(
        solver,
        initial_dt=10 * second,
        cadence=1,
        safety=0.1,
        threshold=0.05,
        max_dt=1 * hour,
    )
    CFL.add_velocity(u)
    logger.info("Trying init loop to get rid of fast waves")
    for i in range(10):
        logger.info("Starting init cycle %s" % i)
        solver_init.solve()
        for j in range(10 + i * 30):
            timestep = CFL.compute_timestep()
            solver.step(timestep)
    solver_init.solve()
    # Now do burn-in
    try:
        logger.info("Starting burn-in loop")
        while solver.proceed:
            timestep = CFL.compute_timestep()
            solver.step(timestep)
            # print(uf.evaluate()['g'])
            if (solver.iteration - 1) % 10 == 0:
                logger.info(
                    "Burn-in Iteration=%i, Time=%e, dt=%e"
                    % (solver.iteration, solver.sim_time, timestep)
                )
    except:
        logger.error("Exception raised, triggering end of burn loop.")
        raise
    # Now define real problem
    solver = problem.build_solver(d3.RK222)
    solver.stop_sim_time = stop_sim_time

    # Analysis
    snapshots = solver.evaluator.add_file_handler(
        output_dir, sim_dt=1 * hour, max_writes=1 * year
    )
    snapshots.add_tasks(solver.state, layout="g")
    # CFL
    CFL = d3.CFL(
        solver,
        initial_dt=10 * second,
        cadence=1,
        safety=0.1,
        threshold=0.05,
        max_dt=1 * hour,
    )
    CFL.add_velocity(u)
    # Main loo
    logger.info("Starting main loop")
    while solver.proceed:
        timestep = CFL.compute_timestep()
        solver.step(timestep)
        if (solver.iteration - 1) % 10 == 0:
            logger.info(
                "Iteration=%i, Time=%e, dt=%e"
                % (solver.iteration, solver.sim_time, timestep)
            )

    solver.log_stats()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    n_cores = mp.cpu_count()
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--ic_dir", default="../data_stubs/")
    parser.add_argument(
        "--output_dir",
        default="/mnt/home/polymathic/ceph/the_well/testing_before_adding/earthswe/",
    )
    args = parser.parse_args()

    ind = int(args.index)
    all_files = sorted(glob(f"{args.ic_dir}IC_*.npy"))
    output_dir = args.output_dir
    print("Processing IC", ind)
    run_ic_file(all_files[ind], output_dir)
