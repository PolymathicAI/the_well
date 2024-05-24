from typing import Dict, Tuple
from .datasets import GenericWellMetadata

import torch
from abc import ABC, abstractmethod
from einops import rearrange, repeat

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
        # print(list(data.keys()))
        x = data["input_fields"]
        if "constant_fields" in data:
            x = torch.cat([x, 
                           repeat(data["constant_fields"], 
                                  self.repeat_constant, t=x.shape[1]
                           )
                                  ], 
                           dim=-1)
        y = data["output_fields"]
        return (rearrange(x, self.rearrange_in),), y
    
    def process_output(self, output):
        return rearrange(output, self.rearrange_out)