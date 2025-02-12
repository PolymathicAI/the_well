import argparse
import math
import os

import h5py as h5
import torch
import yaml

from the_well.data.datasets import WellDataset
from the_well.data.utils import WELL_DATASETS


def compute_statistics(train_path: str, stats_path: str):
    assert not os.path.isfile(stats_path), f"{stats_path} already exists."

    ds = WellDataset(train_path, use_normalization=False)

    counts = {}
    means = {}
    variances = {}
    stds = {}

    for p in ds.files_paths:
        with h5.File(p, "r") as f:
            for i in range(3):
                ti = f"t{i}_fields"

                for field in f[ti].attrs["field_names"]:
                    data = f[ti][field][:]
                    data = torch.as_tensor(data, dtype=torch.float64)

                    count = math.prod(data.shape[: data.ndim - i])
                    var, mean = torch.var_mean(
                        data,
                        dim=tuple(range(0, data.ndim - i)),
                        unbiased=False,
                    )

                    if field in counts:
                        counts[field].append(count)
                        means[field].append(mean)
                        variances[field].append(var)
                    else:
                        counts[field] = [count]
                        means[field] = [mean]
                        variances[field] = [var]

    for field in counts:
        weights = torch.as_tensor(counts[field], dtype=torch.int64)
        weights = weights / weights.sum()
        weights = torch.as_tensor(weights, dtype=torch.float64)

        means[field] = torch.stack(means[field])
        variances[field] = torch.stack(variances[field])

        # https://wikipedia.org/wiki/Mixture_distribution#Moments
        first_moment = torch.einsum("i...,i", means[field], weights)
        second_moment = torch.einsum(
            "i...,i", variances[field] + means[field] ** 2, weights
        )

        mean = first_moment
        std = (second_moment - first_moment**2).sqrt()

        assert torch.all(
            std > 1e-4
        ), f"The standard deviation of the '{field}' field is abnormally low."

        means[field] = mean.tolist()
        stds[field] = std.tolist()

    stats = {"mean": means, "std": stds}

    with open(stats_path, mode="x", encoding="utf8") as f:
        yaml.dump(stats, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Compute the Well dataset statistics")
    parser.add_argument("the_well_dir", type=str)
    args = parser.parse_args()
    data_dir = args.the_well_dir

    for dataset in WELL_DATASETS:
        compute_statistics(
            train_path=os.path.join(data_dir, dataset, "data/train"),
            stats_path=os.path.join(data_dir, dataset, "stats.yaml"),
        )
