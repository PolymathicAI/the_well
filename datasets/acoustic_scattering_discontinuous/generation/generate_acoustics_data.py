import argparse
import multiprocessing as mp
from functools import partial

import numpy as np
from acoustics_2d_interface_maze import setup as maze_setup
from acoustics_2d_interface_random_medium import setup as random_setup

# Time/steps/samples
steps_map = {
    "continuous": (2.0, 101, 2000),
    "discontinuous": (2.0, 101, 2000),
    "inclusions": (2.0, 101, 4000),
    "maze": (4.0, 201, 2000),
}


def mp_wrapper(
    seed,
    discontinuous,
    inclusions,
    maze,
    output_dir="/mnt/home/polymathic/ceph/the_well/testing_before_adding/clawpack_data/acoustics_2d_variable/",
):
    if discontinuous:
        run_func = partial(
            inner_gen_sample,
            discontinuous=True,
            inclusions=False,
            maze=False,
            output_dir=output_dir,
        )
        num_samples = 2000
    elif inclusions:
        run_func = partial(
            inner_gen_sample,
            discontinuous=False,
            inclusions=True,
            maze=False,
            output_dir=output_dir,
        )
        num_samples = 4000
    elif maze:
        run_func = partial(
            inner_gen_sample,
            discontinuous=False,
            inclusions=False,
            maze=True,
            output_dir=output_dir,
        )
        num_samples = 2000
    else:
        run_func = partial(
            inner_gen_sample,
            discontinuous=False,
            inclusions=False,
            maze=False,
            output_dir=output_dir,
        )
        num_samples = 2000
    cores = mp.cpu_count()
    seeds = seed.spawn(num_samples)
    with mp.Pool(cores // 2) as pool:
        pool.map(run_func, seeds)
    # run_func(seeds[0])


def inner_gen_sample(
    seed=0, discontinuous=False, inclusions=False, maze=False, output_dir=""
):
    """
    Iterate num samples times and enerate sample file. Use
    it to overwrite qinit, then run .make output to generate trajectory.

    Make sure overwrite is False in the make file before running.
    """
    # Check conditions and set up names
    file_suffix = f"{str(seed.bit_generator.seed_seq.entropy)}_{str(seed.bit_generator.seed_seq.spawn_key)}"
    if discontinuous:
        file_suffix = "discontinuous_" + file_suffix
        run_func = partial(
            random_setup,
            seed=seed,
            include_splits=True,
            include_inclusions=False,
            outdir=output_dir + file_suffix,
            T_max=2.0,
            num_steps=101,
        )
    elif inclusions:
        file_suffix = "inclusions_" + file_suffix
        run_func = partial(
            random_setup,
            seed=seed,
            include_splits=True,
            include_inclusions=True,
            outdir=output_dir + file_suffix,
            T_max=2.0,
            num_steps=101,
        )
    elif maze:
        file_suffix = "maze_" + file_suffix
        run_func = partial(
            maze_setup,
            seed=seed,
            outdir=output_dir + file_suffix,
            T_max=4.0,
            num_steps=201,
        )
    else:
        file_suffix = "continuous_" + file_suffix
        run_func = partial(
            random_setup,
            seed=seed,
            include_splits=False,
            include_inclusions=False,
            outdir=output_dir + file_suffix,
            T_max=2.0,
            num_steps=101,
        )

    claw = run_func(output_dir + file_suffix)
    claw.run()


if __name__ == "__main__":
    # print(len(gases))
    parser = argparse.ArgumentParser(
        description="Generate initial conditions for 2D Euler quadrants"
    )
    # parser.add_argument('--num_samples', type=int, default=1000, help='Number of samples to generate')
    parser.add_argument(
        "--discontinuity",
        action="store_true",
        help="Whether to generate random samples",
    )
    parser.add_argument(
        "--inclusions", action="store_true", help="Whether to generate random samples"
    )
    parser.add_argument(
        "--switch_to_maze",
        action="store_true",
        help="Whether to generate random samples",
    )
    # parser.add_argument('--bc', type=str, default='extrap', help='Boundary conditions')
    # parser.add_argument('--gas_index', type=int, default=0, help='Index of gas to use (0-9 inclusive)')
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for random samples - use different one per gas/bc if par",
    )
    parser.add_argument(
        "--raw_output_dir",
        type=str,
        default=" /mnt/home/polymathic/ceph/the_well/testing_before_adding/clawpack_data/",
        help="Directory to store raw output",
    )
    args = parser.parse_args()
    seed = np.random.default_rng(
        args.seed
        + 100 * int(args.discontinuity)
        + 1000 * int(args.inclusions)
        + 10000 * int(args.switch_to_maze)
    )
    mp_wrapper(seed, args.discontinuity, args.inclusions, args.switch_to_maze)
