#!/usr/bin/env python3

import argparse
from the_well.benchmark.data.miniwell import create_mini_well
from the_well.benchmark.data.datasets import GenericWellDataset


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
        default="/mnt/ceph/users/polymathic/the_well/",
        help="Path to the dataset. Default is /mnt/ceph/users/polymathic/the_well/",
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
        "--max-files-per-train",
        type=int,
        default=10,
        help="Maximum number of files to process for the training split.",
    )
    parser.add_argument(
        "--max-files-per-val",
        type=int,
        default=2,
        help="Maximum number of files to process for the validation split.",
    )
    parser.add_argument(
        "--max-files-per-test",
        type=int,
        default=2,
        help="Maximum number of files to process for the test split.",
    )

    args = parser.parse_args()

    # Call the create_mini_well function for each split
    for split, max_files in zip(
        ["train", "valid", "test"],
        [args.max_files_per_train, args.max_files_per_val, args.max_files_per_test],
    ):
        # Load the dataset
        dataset = GenericWellDataset(
            well_base_path=args.dataset_path,
            well_dataset_name=args.dataset,
            well_split_name=split,
        )
        mini_metadata = create_mini_well(
            dataset=dataset,
            output_base_path=args.output_base_path,
            spatial_downsample_factor=args.spatial_downsample_factor,
            time_downsample_factor=args.time_downsample_factor,
            max_files=max_files,
            split=split,
        )

        # Optionally, save the mini_metadata or print it
        print(f"Mini dataset created for {split} split with metadata:", mini_metadata)


if __name__ == "__main__":
    main()
