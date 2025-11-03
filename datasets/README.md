# Datasets

Here are presented and stored the Well datasets.
Each dataset is organized as followed:
```
dataset_name
    ├── data/
        ├── train/
        ├── valid/
        └── test/
    ├── stats.yaml
    ├── dataset_name.yaml
    ├── visualization_dataset_name.ipynb
    └── README.md
```

The `stats.yaml` files contain the means and standard deviations of each field, computed on the train set.

The `visualization_dataset_name.ipynb` is a notebook used to visualize the dataset from the HDF5 file directly.

The `dataset_name.yaml` contains the metadata from the HDF5 file.

The `README.md` file contains a detail description of the dataset and the simulations it contains.
