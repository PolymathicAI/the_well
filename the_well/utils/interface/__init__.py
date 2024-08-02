import torch
from omegaconf import OmegaConf

from the_well.benchmark.data.datasets import GenericWellDataset, GenericWellMetadata


class Interface:
    def __init__(self, metadata: GenericWellMetadata):
        self.dataset_metadata = metadata

    @classmethod
    def from_dataset(cls, dataset: GenericWellDataset):
        return cls(dataset.metadata)

    @classmethod
    def from_yaml(cls, filename: str):
        conf = OmegaConf.load(filename)
        metadata_dict = {
            key: val for key, val in dict(conf).items() if key != "sample_shapes"
        }
        metadata = GenericWellMetadata(metadata_dict)
        return cls(metadata)

    def check(self, model: torch.nn.Module, history: int, horizon: int) -> bool:
        batch_size = 2
        input_shape = (
            batch_size,
            history,
            *self.dataset_metadata.sample_shapes["input_fields"],
        )
        output_shape = (
            batch_size,
            horizon,
            *self.dataset_metadata.sample_shapes["output_fields"],
        )
        fake_input = torch.rand(input_shape)
        try:
            pred = model(fake_input)
        except RuntimeError as e:
            print(f"Model {model} cannot ingest input: {e}")
            return False
        else:
            return pred.shape == output_shape
