import argparse
import glob
import os
import subprocess
from typing import Optional

import yaml

WELL_REGISTRY: str = os.path.join(os.path.dirname(__file__), "registry.yaml")


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

    datasets = [
        "acoustic_scattering_discontinuous",
        "acoustic_scattering_inclusions",
        "acoustic_scattering_maze",
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

    splits = ["train", "valid", "test"]

    registry = {}

    for dataset in datasets:
        registry[dataset] = {}

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

        for split in splits:
            path = os.path.join(base_path, f"datasets/{dataset}/data/{split}")

            print(f"Downloading {dataset}/{split} to {path}")

            urls = registry[dataset][split]

            if first_only:
                urls = urls[:1]

            files = [os.path.join(path, os.path.basename(url)) for url in urls]

            # --create-dirs ensures that parent directories exist
            # --continue-at resumes download where it previously stopped
            # --parallel downloads files concurrently
            command = ["curl", "--create-dirs", "--continue-at", "-"]

            if parallel:
                command.append("--parallel")

            for file, url in zip(files, urls):
                command.extend(["-o", file, url])

            try:
                subprocess.run(command)
            except KeyboardInterrupt:
                raise KeyboardInterrupt(
                    "Uh-oh, you pressed ctrl+c! No worries, restarting the download will resume where you left off."
                ) from None


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
