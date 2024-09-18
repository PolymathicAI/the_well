import argparse
import json
import math
import os

import h5py as h5
import torch

from the_well.benchmark.data.datasets import GenericWellDataset, well_paths


def compute_statistics(train_path: str, stats_path: str):
    ds = GenericWellDataset(train_path, use_normalization=False)
    paths = ds.files_paths

    counts = {}
    means = {}
    variances = {}
    stds = {}

    for p in paths:
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

        means[field] = torch.einsum("i...,i", torch.stack(means[field]), weights)
        variances[field] = torch.einsum(
            "i...,i", torch.stack(variances[field]), weights
        )

        means[field] = means[field].tolist()
        stds[field] = variances[field].sqrt().tolist()

    stats = {"mean": means, "std": stds}

    with open(stats_path, mode="x") as f:
        json.dump(stats, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Compute the Well dataset statistics")
    parser.add_argument("the_well_dir", type=str)
    args = parser.parse_args()
    data_dir = args.the_well_dir

    for dataset_path in well_paths.values():
        compute_statistics(
            train_path=os.path.join(data_dir, dataset_path, "data/train"),
            stats_path=os.path.join(data_dir, dataset_path, "stats.json"),
        )
