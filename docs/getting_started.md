# Getting Started

Welcome to The Well, a large-scale collection of diverse scientific numerical simulations designed to accelerate research in machine learning and computational sciences. This guide will help you get started with accessing, installing, and utilizing The Well datasets.

## Prerequisites

The Well datasets range between 6.7GB and 4.9TB of data each, for a total of 15TB for the full collection. Ensure that your system has enough free disk space to accomodate the datasets you wish to download.

If you plan to use The Well datasets to train or evaluate deep learning models, we recommend to use a machine with enough computing resources.

## Installation

First, create and activate a new Python (>=3.10) environment, for example with [venv](https://docs.python.org/3/library/venv.html).

```console
~ $ python -m venv path/to/myenv
~ $ source path/to/myenv/bin/activate
```

Then, clone the [project repository](https://github.com/PolymathicAI/the_well) and install the `the_well` package with its dependencies.

```console
(myenv) ~ $ git clone https://github.com/PolymathicAI/the_well
(myenv) ~ $ cd the_well
(myenv) ~/the_well $ pip install . --extra-index-url https://download.pytorch.org/whl/cu121
```

You should replace `cu121` depending on you hardware. If you want to run the benchmarks, you should install additional dependencies.

```console
(myenv) ~/the_well $ pip install .[benchmark] --extra-index-url https://download.pytorch.org/whl/cu121
```

## Downloading a dataset

Once `the_well` is installed, you can use the `the-well-download` command to download any dataset of The Well. By default, the dataset will be downloaded in the corresponding subfolder of datasets as provided with this repository. However, it is possible to override the download location with the `--output_dir` argument.

```console
(myenv) ~/the_well $ the-well-download --dataset active_matter --output_dir path/to/datasets
```

If the `--dataset` argument is omitted, all datasets will be downloaded. This could take a while!

## Using a dataset

Please refer to the [tutorials](/tutorials/dataset).
