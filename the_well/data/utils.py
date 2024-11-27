from typing import Dict, Tuple

import torch


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
