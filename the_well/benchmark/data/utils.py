from typing import Dict, Tuple

import torch


def preprocess_batch(
    batch: Dict[str, torch.Tensor],
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    """Given a batch provided by a Dataloader iterating over a GenericWellDataset,
    split the batch as such to provide input and output to the model.

    """
    time_step = batch["output_time_grid"] - batch["input_time_grid"]
    parameters = batch["constant_scalars"]
    x = batch["input_fields"]
    x = {"x": x, "time": time_step, "parameters": parameters}
    y = batch["output_fields"]
    return x, y
