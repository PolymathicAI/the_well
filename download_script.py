import argparse
import os

# Define the base URLs for your datasets
datasets = {
    "active_matter": {"base_url": "https://users.flatironinstitute.org/~polymathic/data/the_well/2D/active_matter/data/", "files": [f"L_10.0_zeta_{zeta}_alpha_{alpha}.hdf5" for zeta in [1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0] for alpha in [-1.0, -2.0, -3.0, -4.0, -5.0]], "target_directory": "2D/active_matter/data/"},

    "euler_quadrants": {"base_url": "https://users.flatironinstitute.org/~polymathic/data/the_well/2D/euler_quadrants/data/", "files": [f"gamma{gas_name}_{bc}.hdf5" for gas_name in ["1.3_CO2_20.0", "1.4_Dry_air_20.0", "1.13_C3H8_16.0","1.22_C2H6_15.0","1.33_H2O_20.0", "1.76_Ar_-180.0", "1.365_Dry_air_1000.0","1.404_H2_100.0_Dry_air_-15.0", "1.453_H2_-76.0","1.597_H2_-181.0"] for bc in ["periodic", "extrap"]], "target_directory": "2D/euler_quadrants/data/"},

    "pattern_formation": {"base_url": "https://users.flatironinstitute.org/~polymathic/data/the_well/2D/pattern_formation/data/", "files": ["bubbles_F=0.098_k=0.057.h5","gliders_F=0.014_k=0.054.h5","maze_F=0.029_k=0.057.h5","spots_F=0.03_k=0.062.h5", "worms_F=0.058_k=0.065.h5"], "target_directory": "2D/pattern_formation/data/"},

    "turbulent_radiative_layer_2D": {"base_url": "https://users.flatironinstitute.org/~polymathic/data/the_well/2D/turbulent_radiative_layer/data/", "files": [f"tcool_{tcool:.2f}" for tcool in [0.03, 0.06, 0.10, 0.18, 0.32, 0.56, 1.00, 1.78, 3.16]]},
    # Add more datasets as needed
}

# Define the sample files for each dataset
sample_files = {
    "active_matter": "L_10.0_zeta_1.0_alpha_-1.0.hdf5.hdf5", #select best
    "euler_quadrants": "gamma1.4_Dry_air_20.0_extrap.hdf5", #select best
    "pattern_formation": "bubbles_F=0.098_k=0.057.h5", #select best
    "turbulent_radiative_layer_2D": "tcool_0.03.hdf5", #select best
    # Add more sample files as needed
}

def download_files(dataset_name=None, sample_only=False):
    """
    Download files from a specified dataset or all datasets.

    :param dataset_name: Name of the dataset to download. If None, download all datasets.
    :param sample_only: If True, download only the sample file.
    """
    if dataset_name == 'all':
        datasets_to_download = datasets
    else:
        datasets_to_download = {dataset_name: datasets[dataset_name]}
    
    for name, data in datasets_to_download.items():
        base_url = data["base_url"]
        target_directory = data["target_directory"]
        
        if sample_only:
            # Download only the sample file
            sample_file = sample_files.get(name)
            if sample_file:
                os.system(f"wget -P {target_directory} {base_url}{sample_file}")
                print(f"Downloaded sample file for {name}: {sample_file}")
            else:
                print(f"No sample file defined for {name}")
        else:
            # Download all files from the dataset
            for file_name in data["files"]:
                os.system(f"wget -P -b {target_directory} {base_url}{file_name}") # -P is for saving the file in the specified directory. -b is for running the download in the background.
            print(f"Downloaded all files for {name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Download files from specified datasets.')
    parser.add_argument('--dataset', type=str, help='Name of the dataset to download. If all, all datasets will be downloaded.')
    parser.add_argument('--sample_only', action='store_true', help='Download only the sample file if specified.')

    args = parser.parse_args()

    # Call download_files based on the parsed arguments
    download_files(dataset_name=args.dataset, sample_only=args.sample_only)