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
