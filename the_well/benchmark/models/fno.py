from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from pdearena.models.cond_pdemodel import instantiate_class
from pdearena.modules.conditioned.condition_utils import fourier_embedding
from pdearena.modules.conditioned.twod_resnet import FourierBasicBlock, ResNet
from pdearena.utils import partialclass


@dataclass
class PDEModelConfig:
    """
    Original code: https://github.com/pdearena/pdearena/blob/main/pdearena/data/utils.py#L9-L14
    """

    n_scalar_components: int
    n_vector_components: int
    n_spatial_dim: int
    n_param_conditioning: int = 1


COND_MODEL_REGISTRY = {
    "FNO-128-16m": {
        "class_path": "the_well.benchmark.models.FNO",
        "init_args": {
            "hidden_channels": 128,
            "num_blocks": [1, 1, 1, 1],
            "block": partialclass(
                "CustomFourierBasicBlock", FourierBasicBlock, modes1=16, modes2=16
            ),
        },
    }
}


class FNO(ResNet):
    """Reimplementation of the base class from PDEArena to include multi-parameters inputs.

    Original code: https://github.com/pdearena/pdearena/blob/main/pdearena/modules/conditioned/twod_resnet.py
    """

    def __init__(
        self,
        n_input_scalar_components: int,
        n_input_vector_components: int,
        n_output_scalar_components: int,
        n_output_vector_components: int,
        block: torch.nn.Module,
        num_blocks: list,
        hidden_channels: int = 64,
        activation: str = "gelu",
        n_param_conditioning: int = 1,
    ):
        super().__init__(
            n_input_scalar_components,
            n_input_vector_components,
            n_output_scalar_components,
            n_output_vector_components,
            block=block,
            num_blocks=num_blocks,
            time_history=1,
            time_future=1,
            hidden_channels=hidden_channels,
            activation=activation,
            norm=False,
            diffmode=False,
            usegrid=False,
            param_conditioning="scalar",
        )
        self.n_param_conditioning = n_param_conditioning
        time_embed_dim = hidden_channels * 4
        self.pde_emb = torch.nn.ModuleList(
            [
                torch.nn.Sequential(
                    torch.nn.Linear(hidden_channels, time_embed_dim),
                    self.activation,
                    torch.nn.Linear(time_embed_dim, time_embed_dim),
                )
                for _ in range(self.n_param_conditioning)
            ]
        )

    def forward(self, x: torch.Tensor, time, z=Optional[torch.Tensor]) -> torch.Tensor:
        assert x.dim() == 5
        orig_shape = x.shape
        x = x.reshape(x.size(0), -1, *x.shape[3:])  # collapse T,C

        emb = self.time_embed(fourier_embedding(time, self.in_planes))
        if z is not None:
            if self.param_conditioning == "scalar":
                assert z.dim() == 2
                assert (
                    z.shape[-1] == self.n_param_conditioning
                ), f"Expected Nx{self.n_param_conditioning} parameters but received {z.shape}"
                for p_dim in range(z.shape[-1]):
                    emb = emb + self.pde_emb[p_dim](
                        fourier_embedding(z[:, p_dim], self.in_planes)
                    )
        x = self.activation(self.conv_in1(x.float()))
        x = self.activation(self.conv_in2(x.float()))

        if self.padding > 0:
            x = F.pad(x, [0, self.padding, 0, self.padding])

        for layer in self.layers:
            x = layer(x, emb)

        if self.padding > 0:
            x = x[..., : -self.padding, : -self.padding]

        x = self.activation(self.conv_out1(x))
        x = self.conv_out2(x)

        if self.diffmode:
            raise NotImplementedError("diffmode")
        return x.reshape(
            orig_shape[0],
            -1,
            (self.n_output_scalar_components + self.n_output_vector_components * 2),
            *orig_shape[3:],
        )


def get_model(args, pde):
    """Utility function to instanciate FNO model based on arguments and pde parameters.

    Original code: https://github.com/pdearena/pdearena/blob/main/pdearena/models/cond_pdemodel.py#L20-L40
    """
    if args.name in COND_MODEL_REGISTRY:
        _model = COND_MODEL_REGISTRY[args.name].copy()
        _model["init_args"].update(
            dict(
                n_input_scalar_components=pde.n_scalar_components,
                n_output_scalar_components=pde.n_scalar_components,
                n_input_vector_components=pde.n_vector_components,
                n_output_vector_components=pde.n_vector_components,
                n_param_conditioning=pde.n_param_conditioning,
            )
        )
        model = instantiate_class(tuple(), _model)
        return model
    else:
        raise NameError(f"Model {args.name} not found in registry.")
