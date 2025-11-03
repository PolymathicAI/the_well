# Interface

The purpose of the interface is to check whether data provided by a dataset of the Well can be ingested by any model.

## Creation

Interface are created from dataset information. The class provides several class methods to create an interface instance:
- `from_dataset` retrieve the metadata from an existing `dataset` instance to create the interface.
- `from_yaml` parses a metadata file, whose path is provided by the `filename` argument. Such file would have previous been created by savings the metadata of a dataset instance from the Well.

## Specification

One can create class inheriting the base `Interface` and implement the `pipe_{one_step,rollout}_{input,output}` methods to pipe data provided by the Well dataset or predicted by the model into the correct format.

## Check

The `Inteface` class provides two methods to check the model can properly ingest the data from the Well dataset: `check_one_step` and `check_rollout`. Only the first one is actually implemented.
