import argparse
import json
import os
import signal
import sys


def signal_handler(sig, frame):
    print("You pressed Ctrl+C!")
    sys.exit(0)


def download_files(json_file, dataset_name=None, sample_only: bool = False):
    """
    Download files listed in a JSON file for a specified dataset or all datasets.

    :param json_file: Path to the JSON file containing file URLs.
    :param dataset_name: Name of the dataset to download. If None, downloads all datasets.
    """
    # Load the JSON file with dataset information
    with open(json_file, "r") as file:
        datasets = json.load(file)
    if dataset_name is not None and dataset_name not in datasets:
        print(f"No dataset found with the name {dataset_name}.")
        return

    # Determine which datasets to download
    if dataset_name:
        if sample_only:
            datasets_to_download = {dataset_name: [datasets[dataset_name][0]]}
        else:
            datasets_to_download = {dataset_name: datasets[dataset_name]}
    else:
        datasets_to_download = datasets

    # Create target directories and download files
    for name, file_urls in datasets_to_download.items():
        if "2D" in file_urls[0]:
            target_directory_base = os.path.abspath(
                os.path.join(os.path.dirname(__file__), f"../../2D/{name}/data")
            )
        elif "3D" in file_urls[0]:
            target_directory_base = os.path.abspath(
                os.path.join(os.path.dirname(__file__), f"../../3D/{name}/data")
            )
        else:
            raise NameError(f"Expect 2D or 3D dimension in file {file_urls[0]}")

        for url in file_urls:
            if "train" in url:
                target_directory = os.path.join(target_directory_base, "train")
                if not os.path.exists(target_directory):
                    os.makedirs(target_directory, exist_ok=False)
                    print("creating directory")
            elif "test" in url:
                target_directory = os.path.join(target_directory_base, "test")
                if not os.path.exists(target_directory):
                    os.makedirs(target_directory, exist_ok=False)
                    print("creating directory")
            elif "valid" in url:
                target_directory = os.path.join(target_directory_base, "valid")
                if not os.path.exists(target_directory):
                    os.makedirs(target_directory, exist_ok=False)
                    print("creating directory")
            filename = os.path.basename(url)
            print(f"Downloading {filename} to {target_directory}")
            os.system(
                f"curl --retry 5  --create-dirs -o {os.path.join(target_directory, filename)} {url}"
            )


def main():
    signal.signal(signal.SIGINT, signal_handler)
    parser = argparse.ArgumentParser(
        description="Download files from specified datasets based on a JSON registry."
    )
    parser.add_argument(
        "--json_file",
        type=str,
        default="data_registry.json",
        help="Path to the JSON file with file URLs.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        help="Name of the dataset to download. If omitted, all datasets will be downloaded.",
    )

    args = parser.parse_args()

    # Call download_files based on the parsed arguments
    download_files(args.json_file, args.dataset)


if __name__ == "__main__":
    main()
