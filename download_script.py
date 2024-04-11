import os
import argparse

# Define the base URLs for your datasets
datasets = {
    "active_matter": {"base_url": "https://users.flatironinstitute.org/~polymathic/data/the_well/2D/active_matter/data/", "files": [f"L_10.0_zeta_{zeta}_alpha_{alpha}.hdf5" for zeta in [1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0] for alpha in [-1.0, -2.0, -3.0, -4.0, -5.0]]},
    "euler_quadrants": {"base_url": "https://users.flatironinstitute.org/~polymathic/data/the_well/2D/euler_quadrants/data/", "files": [f"gamma{gas_name}_{bc}.hdf5" for gas_name in ["1.3_CO2_20.0", "1.4_Dry_air_20.0", "1.13_C3H8_16.0","1.22_C2H6_15.0","1.33_H2O_20.0", "1.76_Ar_-180.0", "1.365_Dry_air_1000.0","1.404_H2_100.0_Dry_air_-15.0", "1.453_H2_-76.0","1.597_H2_-181.0"] for bc in ["periodic", "extrap"]]},
    # Add more datasets as needed
}

# Define the sample files for each dataset
sample_files = {
    "dataset1": "sample_dataset1.hdf5",
    "dataset2": "sample_dataset2.hdf5",
    # Add more sample files as needed
}

def download_files(dataset_name=None, sample_only=False):
    """
    Download files from a specified dataset or all datasets.

    :param dataset_name: Name of the dataset to download. If None, download all datasets.
    :param sample_only: If True, download only the sample file.
    """
    if dataset_name:
        datasets_to_download = {dataset_name: datasets[dataset_name]}
    else:
        # If no specific dataset name is provided, download from all datasets
        datasets_to_download = datasets
    
    for name, data in datasets_to_download.items():
        base_url = data["base_url"]
        
        if sample_only:
            # Download only the sample file
            sample_file = sample_files.get(name)
            if sample_file:
                os.system(f"wget {base_url}{sample_file}")
                print(f"Downloaded sample file for {name}: {sample_file}")
            else:
                print(f"No sample file defined for {name}")
        else:
            # Download all files from the dataset
            for file_name in data["files"]:
                os.system(f"wget {base_url}{file_name}")
            print(f"Downloaded all files for {name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Download files from specified datasets.')
    parser.add_argument('--dataset', type=str, help='Name of the dataset to download. If omitted, all datasets will be downloaded.')
    parser.add_argument('--sample_only', action='store_true', help='Download only the sample file if specified.')

    args = parser.parse_args()

    # Call download_files based on the parsed arguments
    download_files(dataset_name=args.dataset, sample_only=args.sample_only)