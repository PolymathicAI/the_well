import argparse
import os

import h5py as h5
import numpy as np
import torch

from the_well.benchmark.data.datasets import GenericWellDataset, well_paths


def compute_statistics(train_path, original_path):
    ds = GenericWellDataset(train_path, use_normalization=False)
    paths = ds.files_paths
    means = {}
    counts = {}
    fields = ["t0_fields", "t1_fields", "t2_fields"]
    print("started computing the statistics of", original_path)
    for p in paths:
        os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
        with h5.File(p, "r") as f:
            for fi in fields:
                for field in f[fi].attrs["field_names"]:
                    data = f[fi][field][:]
                    if field not in means:
                        means[field] = data.sum()
                        counts[field] = np.prod(data.shape)
                    else:
                        means[field] += data.sum()
                        counts[field] += np.prod(data.shape)

    # Compute means from sum and counts
    for field in means:
        means[field] /= counts[field]
    print(means)

    # Now let's compute the variance
    variances = {}
    for p in paths:
        os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
        with h5.File(p, "r") as f:
            for fi in fields:
                for field in f[fi].attrs["field_names"]:
                    data = f[fi][field][:]
                    if field not in variances:
                        variances[field] = ((data - means[field]) ** 2).sum()
                    else:
                        variances[field] += ((data - means[field]) ** 2).sum()

    # Compute vars from sum and counts
    for field in variances:
        variances[field] /= counts[field] - 1

    stds = {k: np.sqrt(v) for k, v in variances.items()}
    print("saving for ", original_path)
    # delete "data" in the original path
    original_path = original_path[:-4]
    stats_path = original_path + "stats_new"
    if not os.path.exists(stats_path):
        os.makedirs(stats_path)
    with open(stats_path + "/means.pkl", "wb") as f:
        torch.save(means, f)

    with open(stats_path + "/stds.pkl", "wb") as f:
        torch.save(stds, f)


def recompute_statistics(data_register):
    for data_path in data_register:
        data_train_path = f"{data_path}/train/"
        compute_statistics(data_train_path, data_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Compute the Well dataset statistics")
    parser.add_argument("the_well_dir", type=str)
    args = parser.parse_args()
    data_dir = args.the_well_dir
    data_register = [
        os.path.join(data_dir, dataset_path) for dataset_path in well_paths.values()
    ]
    recompute_statistics(data_register)
