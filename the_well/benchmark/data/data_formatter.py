from abc import ABC, abstractmethod
from typing import Dict, Tuple

import torch
from einops import rearrange

from .datasets import GenericWellMetadata


class AbstractDataFormatter(ABC):
    def __init__(self, metadata: GenericWellMetadata):
        self.metadata = metadata

    @abstractmethod
    def process_input(self, data: Dict) -> Tuple:
        raise NotImplementedError

    def process_output(self, data: Dict, output) -> torch.tensor:
        raise NotImplementedError


class DefaultChannelsFirstFormatter(AbstractDataFormatter):
    """
    Default preprocessor for data in channels first format.

    Stacks time as individual channel.
    """

    def __init__(self, metadata: GenericWellMetadata):
        super().__init__(metadata)
        if metadata.n_spatial_dims == 2:
            self.rearrange_in = "b t h w c -> b (t c) h w"
            self.repeat_constant = "b h w c -> b t h w c"
            self.rearrange_out = "b c h w -> b 1 h w c"
        elif metadata.n_spatial_dims == 3:
            self.rearrange_in = "b t h w d c -> b (t c) h w d"
            self.repeat_constant = "b h w d c -> b t h w d c"
            self.rearrange_constants = "b h w d c -> b c h w d"
            self.rearrange_out = "b c h w d -> b 1 h w d c"

    def process_input(self, data: Dict):
        x = data["input_fields"]
        x = rearrange(x, self.rearrange_in)
        if "constant_fields" in data:
            flat_constants = rearrange(data["constant_fields"], self.rearrange_in)
            x = torch.cat(
                [
                    x,
                    flat_constants,
                ],
                dim=1,
            )
        y = data["output_fields"]
        # TODO - Add warning to output if nan has to be replaced
        # in some cases (staircase), its ok. In others, it's not.
        return (torch.nan_to_num(x),), torch.nan_to_num(y)

    def process_output(self, output):
        return rearrange(output, self.rearrange_out)


class DefaultChannelsLastFormatter(AbstractDataFormatter):
    """
    Default preprocessor for data in channels last format.

    Stacks time as individual channel.
    """

    def __init__(self, metadata: GenericWellMetadata):
        super().__init__(metadata)
        if metadata.n_spatial_dims == 2:
            self.rearrange_in = "b t h w c -> b h w (t c)"
            self.repeat_constant = "b h w c -> b t h w c"
            self.rearrange_out = "b h w c -> b 1 h w c"
        elif metadata.n_spatial_dims == 3:
            self.rearrange_in = "b t h w d c -> b h w d (t c)"
            self.repeat_constant = "b h w d c -> b t h w d c"
            self.rearrange_out = "b h w d c -> b 1 h w d c"

    def process_input(self, data: Dict):
        x = data["input_fields"]
        x = rearrange(x, self.rearrange_in)
        if "constant_fields" in data:
            flat_constants = rearrange(data["constant_fields"], self.rearrange_in)
            x = torch.cat(
                [
                    x,
                    flat_constants,
                ],
                dim=-1,
            )
        y = data["output_fields"]
        # TODO - Add warning to output if nan has to be replaced
        # in some cases (staircase), its ok. In others, it's not.
        return (torch.nan_to_num(x),), torch.nan_to_num(y)

    def process_output(self, output):
        return rearrange(output, self.rearrange_out)
