#!/usr/bin/env python3

import argparse

from the_well.data.datasets import WellDataset
from the_well.data.miniwell import create_mini_well


def main():
    parser = argparse.ArgumentParser(
        description="Create a minified version of The Well."
    )
    parser.add_argument(
        "output_base_path", type=str, help="Base path for the output dataset."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        help="Name of the dataset to create a mini version of.",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="/mnt/ceph/users/polymathic/the_well/datasets",
        help="Path to the dataset.",
    )
    parser.add_argument(
        "--spatial-downsample-factor",
        type=int,
        default=4,
        help="Factor by which to downsample spatial dimensions.",
    )
    parser.add_argument(
        "--time-downsample-factor",
        type=int,
        default=2,
        help="Factor by which to downsample time dimensions.",
    )
    parser.add_argument(
        "--max-trajectories-per-train",
        type=int,
        default=100,
        help="Maximum number of trajectories to process for the training split.",
    )
    parser.add_argument(
        "--max-trajectories-per-val",
        type=int,
        default=20,
        help="Maximum number of trajectories to process for the validation split.",
    )
    parser.add_argument(
        "--max-trajectories-per-test",
        type=int,
        default=20,
        help="Maximum number of trajectories to process for the test split.",
    )
    parser.add_argument(
        "--time-fraction",
        type=float,
        default=1.0,
        help="Fraction of the time dimension to keep, starting from the first timestep. Default is 1.0 (keep all).",
    )

    args = parser.parse_args()

    # Call the create_mini_well function for each split
    for split, max_trajectories in zip(
        ["train", "valid", "test"],
        [
            args.max_trajectories_per_train,
            args.max_trajectories_per_val,
            args.max_trajectories_per_test,
        ],
    ):
        # Load the dataset
        dataset = WellDataset(
            well_base_path=args.dataset_path,
            well_dataset_name=args.dataset,
            well_split_name=split,
        )
        mini_metadata = create_mini_well(
            dataset=dataset,
            output_base_path=args.output_base_path,
            spatial_downsample_factor=args.spatial_downsample_factor,
            time_downsample_factor=args.time_downsample_factor,
            max_trajectories=max_trajectories,  # Changed to max_trajectories
            split=split,
            time_fraction=args.time_fraction,
        )

        # Optionally, save the mini_metadata or print it
        print(f"Mini dataset created for {split} split with metadata:", mini_metadata)


if __name__ == "__main__":
    main()
