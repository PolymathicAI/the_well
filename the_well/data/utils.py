import itertools
from typing import Dict, Tuple

import torch

WELL_DATASETS = [
    "acoustic_scattering_maze",
    "acoustic_scattering_inclusions",
    "acoustic_scattering_discontinuous",
    "active_matter",
    "convective_envelope_rsg",
    "euler_multi_quadrants_openBC",
    "euler_multi_quadrants_periodicBC",
    "helmholtz_staircase",
    "MHD_64",
    "MHD_256",
    "gray_scott_reaction_diffusion",
    "planetswe",
    "post_neutron_star_merger",
    "rayleigh_benard",
    "rayleigh_benard_uniform",
    "rayleigh_taylor_instability",
    "shear_flow",
    "supernova_explosion_64",
    "supernova_explosion_128",
    "turbulence_gravity_cooling",
    "turbulent_radiative_layer_2D",
    "turbulent_radiative_layer_3D",
    "viscoelastic_instability",
]


IO_PARAMS = {
    "fsspec_params": {
        # "skip_instance_cache": True
        "cache_type": "blockcache",  # or "first" with enough space
        "block_size": 8 * 1024 * 1024,  # could be bigger
    },
    "h5py_params": {
        "driver_kwds": {  # only recent versions of xarray and h5netcdf allow this correctly
            "page_buf_size": 8 * 1024 * 1024,  # this one only works in repacked files
            "rdcc_nbytes": 8 * 1024 * 1024,  # this one is to read the chunks
        }
    },
}


def is_dataset_in_the_well(dataset_name: str) -> bool:
    """Tell whether a dataset is in the Well or not.
    Accept `dummy` as a valid dataset.

    Args:
        dataset_name: The name of the dataset.

    Returns:
        True if the dataset is in the Well, False otherwise.
    """
    is_valid = dataset_name in WELL_DATASETS or dataset_name == "dummy"
    return is_valid


def preprocess_batch(
    batch: Dict[str, torch.Tensor],
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    """Given a batch provided by a Dataloader iterating over a WellDataset,
    split the batch as such to provide input and output to the model.

    """
    time_step = batch["output_time_grid"] - batch["input_time_grid"]
    parameters = batch["constant_scalars"]
    x = batch["input_fields"]
    x = {"x": x, "time": time_step, "parameters": parameters}
    y = batch["output_fields"]
    return x, y


def flatten_field_names(metadata, include_constants=True):
    flat_field_names = itertools.chain(*metadata.field_names.values())
    flat_constant_field_names = itertools.chain(*metadata.constant_field_names.values())

    if include_constants:
        return [*flat_field_names, *flat_constant_field_names]
    else:
        return [*flat_field_names]


def raw_steps_to_possible_sample_t0s(
    total_steps_in_trajectory: int,
    n_steps_input: int,
    n_steps_output: int,
    dt_stride: int,
):
    """Given the total number of steps in a trajectory returns the number of samples that can be taken from the
      trajectory such that all samples have at least n_steps_input + n_steps_output steps with steps separated
      by dt_stride.

    ex1: total_steps_in_trajectory = 5, n_steps_input = 1, n_steps_output = 1, dt_stride = 1
        Possible samples are: [0, 1], [1, 2], [2, 3], [3, 4]
    ex2: total_steps_in_trajectory = 5, n_steps_input = 1, n_steps_output = 1, dt_stride = 2
        Possible samples are: [0, 2], [1, 3], [2, 4]
    ex3: total_steps_in_trajectory = 5, n_steps_input = 1, n_steps_output = 1, dt_stride = 3
        Possible samples are: [0, 3], [1, 4]
    ex4: total_steps_in_trajectory = 5, n_steps_input = 2, n_steps_output = 1, dt_stride = 2
        Possible samples are: [0, 2, 4]

    """
    elapsed_steps_per_sample = 1 + dt_stride * (
        n_steps_input + n_steps_output - 1
    )  # Number of steps needed for sample
    return max(0, total_steps_in_trajectory - elapsed_steps_per_sample + 1)


def maximum_stride_for_initial_index(
    time_idx: int,
    total_steps_in_trajectory: int,
    n_steps_input: int,
    n_steps_output: int,
):
    """Given the total number of steps in a file and the current step returns the maximum stride
    that can be taken from the file such that all samples have at least n_steps_input + n_steps_output steps with a stride of
      dt_stride
    """
    used_steps_per_sample = n_steps_input + n_steps_output
    return max(
        0,
        int((total_steps_in_trajectory - time_idx - 1) // (used_steps_per_sample - 1)),
    )
