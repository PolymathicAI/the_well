# Datasets

Here are presented and stored the Well datasets.
Each dataset is organized as followed:
```
dataset_name
    ├── data/
        ├── train/
        ├── valid/
        └── test/
    ├── stats/
        ├── means.pkl
        └── stds.pkl
    ├── visualization_dataset_name.ipynb
    └── README.md
```

The `stats/means.pkl` and `stats/stds.pkl` files contain the means and standard deviations computed on the train set.

The `visualization_dataset_name.ipynb` is a notebook used to visualize the dataset using the custom Dataloader.

The `README.md` file contains a detail description of the dataset and the simulations it contains.
