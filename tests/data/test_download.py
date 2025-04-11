import glob
import pathlib


def test_download_dataset(download_dataset):
    dataset_dir: pathlib.Path = download_dataset
    hdf5_files = glob.glob(f"{dataset_dir}/data/train/*.hdf5")
    assert len(hdf5_files) == 1
