import argparse
import glob
import os
import subprocess
from typing import List, Optional

import h5py
import yaml

from the_well.data.utils import WELL_DATASETS

WELL_REGISTRY: str = os.path.join(os.path.dirname(__file__), "registry.yaml")

# Messages printed before downloading a dataset
DATASET_NOTICES = {
    "shear_flow": (
        "shear_flow was updated to v2.0 in October 2026. In v1.0, about 80% of the "
        "validation and test trajectories were exact copies of training trajectories. "
        "Results obtained on the v1.0 validation and test sets are not comparable with v2.0. "
        "The Hugging Face copy (hf://datasets/polymathic-ai/shear_flow) is still v1.0 for now."
    ),
}


def _is_shear_flow_v1(file: str) -> bool:
    """Whether a local shear_flow file is from v1.0, which lacks the initial condition scalars."""
    try:
        with h5py.File(file, "r") as f:
            return "width" not in f["scalars"]
    except Exception:
        # Partially downloaded file
        return False


def _print_notice(dataset: str, path: str):
    if dataset not in DATASET_NOTICES:
        return
    print(f"WARNING: {DATASET_NOTICES[dataset]}", flush=True)
    if dataset == "shear_flow":
        local_files = glob.glob(os.path.join(path, "data", "*", "*.hdf5"))
        old_files = [file for file in local_files if _is_shear_flow_v1(file)]
        if old_files:
            print(
                f"WARNING: {len(old_files)} v1.0 file(s) found in {path}. Existing files are resumed, "
                f"not replaced, so they would be kept as they are. Delete {path} and run this command again.",
                flush=True,
            )


def create_url_registry(
    registry_path: str = WELL_REGISTRY,
    base_path: str = "/mnt/ceph/users/polymathic/the_well",
    base_url: str = "https://sdsc-users.flatironinstitute.org/~polymathic/data/the_well",
):
    """Create The Well URL registry.

    Args:
        registry_path: The path to the YAML registry file containing file URLs.
        base_path: The path where the 'datasets' directory is located.
        base_url: The base URL of the files.
    """
    splits = ["train", "valid", "test"]

    registry = {}

    for dataset in WELL_DATASETS:
        registry[dataset] = {}

        stat_file = os.path.join(base_path, f"datasets/{dataset}/stats.yaml")
        assert os.path.exists(stat_file), f"{stat_file} does not exist"
        registry[dataset]["stats"] = stat_file.replace(base_path, base_url)

        for split in splits:
            registry[dataset][split] = []

            path = os.path.join(base_path, f"datasets/{dataset}/data/{split}")
            files = glob.glob(os.path.join(path, "*.hdf5")) + glob.glob(
                os.path.join(path, "*.h5")
            )

            for file in files:
                registry[dataset][split].append(file.replace(base_path, base_url))

    with open(registry_path, mode="w") as f:
        yaml.dump(registry, f)


def _trigger_download(download_command: List[str]):
    try:
        subprocess.run(download_command)
    except KeyboardInterrupt:
        raise KeyboardInterrupt(
            "Uh-oh, you pressed ctrl+c! No worries, restarting the download will resume where you left off."
        ) from None


def well_download(
    base_path: str,
    dataset: Optional[str] = None,
    split: Optional[str] = None,
    first_only: bool = False,
    parallel: bool = False,
    registry_path: str = WELL_REGISTRY,
):
    """Download The Well dataset files.

    This function uses `curl` to download files.

    Args:
        path: The path where the 'datasets' directory is located.
        dataset: The name of a dataset to download. If omitted, downloads all datasets.
        split: The dataset split ('train', 'valid' or 'test') to download. If omitted, downloads all splits.
        first_only: Whether to only download the first file of the dataset.
        parallel: Whether to download files in parallel.
        registry_path: The path to the YAML registry file containing file URLs.
    """

    base_path = os.path.abspath(os.path.expanduser(base_path))

    # --create-dirs ensures that parent directories exist
    # --continue-at resumes download where it previously stopped
    # --parallel downloads files concurrently
    base_download_command = ["curl", "--create-dirs", "--continue-at", "-"]
    if parallel:
        base_download_command.append("--parallel")

    with open(registry_path, mode="r") as f:
        registry = yaml.safe_load(f)

    if dataset is None:
        datasets = list(registry.keys())
    else:
        datasets = [dataset]

    if split is None:
        splits = ["train", "valid", "test"]
    else:
        splits = [split]

    path = os.path.join(os.path.abspath(os.path.expanduser(base_path)), "datasets")

    for dataset in datasets:
        assert (
            dataset in registry
        ), f"unknown dataset '{dataset}', expected one of {list(registry.keys())}"

        _print_notice(dataset, os.path.join(base_path, f"datasets/{dataset}"))

        # Download file containing dataset statistics
        stat_file_url = registry[dataset]["stats"]
        stat_file_path = os.path.join(base_path, f"datasets/{dataset}/stats.yaml")
        download_command = base_download_command.copy()
        download_command.extend(["-o", stat_file_path, stat_file_url])
        _trigger_download(download_command)

        # Download each split independently
        for split in splits:
            path = os.path.join(base_path, f"datasets/{dataset}/data/{split}")

            print(f"Downloading {dataset}/{split} to {path}")

            urls = registry[dataset][split]

            if first_only:
                urls = urls[:1]

            files = [os.path.join(path, os.path.basename(url)) for url in urls]

            download_command = base_download_command.copy()

            for file, url in zip(files, urls):
                download_command.extend(["-o", file, url])

            _trigger_download(download_command)


def main():
    parser = argparse.ArgumentParser(description="Download The Well dataset files.")
    parser.add_argument(
        "--dataset",
        type=str,
        help="The name of the dataset to download. If omitted, downloads all datasets.",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "valid", "test"],
        help="The dataset split ('train', 'valid' or 'test') to download. If omitted, downloads all splits.",
    )
    parser.add_argument(
        "--base-path",
        type=str,
        default=os.path.abspath("."),
        help="The path where the 'datasets' directory is located.",
    )
    parser.add_argument(
        "--first-only",
        action="store_true",
        default=False,
        help="Whether to only download the first file of the dataset.",
    )
    parser.add_argument(
        "--parallel",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether to download files in parallel.",
    )
    parser.add_argument(
        "--registry-path",
        type=str,
        default=WELL_REGISTRY,
        help="The path to the YAML registry file containing file URLs.",
    )

    args = parser.parse_args()

    try:
        well_download(**vars(args))
    except KeyboardInterrupt as e:
        print()
        print(e)


if __name__ == "__main__":
    main()
