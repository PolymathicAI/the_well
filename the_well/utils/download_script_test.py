import os
import json
import argparse
import re

def download_files(json_file, dataset_name=None):
    """
    Download files listed in a JSON file for a specified dataset or all datasets.

    :param json_file: Path to the JSON file containing file URLs.
    :param dataset_name: Name of the dataset to download. If None, downloads all datasets.
    """
    # Load the JSON file with dataset information
    with open(json_file, 'r') as file:
        datasets = json.load(file)
    print(datasets)
    if dataset_name is not None and dataset_name not in datasets:
        print(f"No dataset found with the name {dataset_name}.")
        return

    # Determine which datasets to download
    datasets_to_download = datasets if dataset_name is None else {dataset_name: datasets[dataset_name]}

    # Create target directories and download files
    for name, file_urls in datasets_to_download.items():
        if re.search(r'the_well/2D', file_urls[0]):
            target_directory_base = f'../../2D/{name}/data_test/'
        elif re.search(r'the_well/3D', file_urls[0]):
            target_directory_base = f'../../3D/{name}/data_test/'
        if not os.path.exists(target_directory_base):
            os.makedirs(target_directory_base, exist_ok=False)
            print('creating directory')
        for url in file_urls:
            if 'train' in url:
                target_directory = target_directory_base + 'train/'
                if not os.path.exists(target_directory):
                    os.makedirs(target_directory, exist_ok=False)
                    print('creating directory')
            elif 'test' in url:
                target_directory = target_directory_base + 'test/'
                if not os.path.exists(target_directory):
                    os.makedirs(target_directory, exist_ok=False)
                    print('creating directory')
            elif 'valid' in url:
                target_directory = target_directory_base + 'valid/'
                if not os.path.exists(target_directory):
                    os.makedirs(target_directory, exist_ok=False)
                    print('creating directory')
            filename = os.path.basename(url)
            print(f"Downloading {filename} to {target_directory}")
            os.system(f"curl -o {os.path.join(target_directory, filename)} {url}")

def main():
    parser = argparse.ArgumentParser(description="Download files from specified datasets based on a JSON registry.")
    parser.add_argument("--json_file", type=str, default='data_registry.json', help="Path to the JSON file with file URLs.")
    parser.add_argument("--dataset", type=str, help="Name of the dataset to download. If omitted, all datasets will be downloaded.")
    
    args = parser.parse_args()

    # Call download_files based on the parsed arguments
    download_files(args.json_file, args.dataset)

if __name__ == "__main__":
    main()
