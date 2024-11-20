import torch
from omegaconf import OmegaConf

from the_well.data.datasets import WellDataset, WellMetadata


class Interface:
    def __init__(self, metadata: WellMetadata):
        self.dataset_metadata = metadata

    @classmethod
    def from_dataset(cls, dataset: WellDataset):
        return cls(dataset.metadata)

    @classmethod
    def from_yaml(cls, filename: str):
        conf = OmegaConf.load(filename)
        metadata_dict = {
            key: val for key, val in dict(conf).items() if key != "sample_shapes"
        }
        metadata = WellMetadata(metadata_dict)
        return cls(metadata)

    def pipe_one_step_input(self, data):
        """Prepare data input for one step prediction."""
        return data

    def pipe_rollout_input(self, data):
        """Prepare data input for rollout prediction."""
        return data

    def pipe_one_step_output(self, data):
        """Process one step prediction for metrics computation."""
        return data

    def pipe_rollout_output(self, data):
        """Process rollout prediction for metrics computation."""
        return data

    def check_one_step(
        self, model: torch.nn.Module, history: int, horizon: int
    ) -> bool:
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
        fake_input = self.pipe_one_step_input(fake_input)
        try:
            pred = model(fake_input)
        except RuntimeError as e:
            print(f"Model {model} cannot ingest input: {e}")
            return False
        else:
            pred = self.pipe_one_step_output(pred)
            return pred.shape == output_shape

    def check_rollout(self, model: torch.nn.Module, history: int, horizon: int) -> bool:
        raise NotImplementedError
