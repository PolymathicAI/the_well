# Welcome to the Well. 
## Installation

You can install the code with the following command, which will directly download the code. The package is not yet available on PyPi.

```bash
pip install git+https://https://github.com/PolymathicAI/the_well.git
```

Once installed, you can use the `the-well-download` utility to download dataset.
For instance, run the following to download only samples of the `active_matter` dataset.
```bash
the-well-download --dataset active_matter --sample-only
```

## How To Contribute


### Code Style

The code should follow the standard [PEP8 Style Guide for Python Code](https://peps.python.org/pep-0008/). This is enforced by using the [ruff](https://docs.astral.sh/ruff/) code linter and formatter. 

To help with maintaining compliance with the format, we provide a `pre-commit` hook that automatically runs `ruff` on the modified code. To install and learn more about `pre-commit`, please refer to [the official documentation](https://pre-commit.com/).