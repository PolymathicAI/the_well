import argparse
import os.path

from omegaconf import OmegaConf
from tqdm import tqdm

from the_well.data.datasets import WELL_DATASETS, WellDataset


def generate_metadata(dataset_dir: str, output_dir: str):
    for dataset_name in tqdm(WELL_DATASETS):
        dataset = WellDataset(
            well_base_path=dataset_dir,
            well_dataset_name=dataset_name,
            n_steps_input=1,
            n_steps_output=1,
            well_split_name="valid",
            use_normalization=False,
        )
        metadata = dataset.metadata
        metadata_conf = OmegaConf.create(metadata.__dict__)
        filename = os.path.join(output_dir, f"{dataset_name}.yaml")
        with open(filename, "w") as metadata_file:
            metadata_file.write(OmegaConf.to_yaml(metadata_conf))


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Dataset metadata generator")
    parser.add_argument("dataset_dir", type=str)
    parser.add_argument("output_dir", type=str)
    args = parser.parse_args()
    dataset_dir = args.dataset_dir
    output_dir = args.output_dir
    generate_metadata(dataset_dir, output_dir)
