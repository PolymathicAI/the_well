"""
Adapted from:

    Takamoto et al. 2022, PDEBENCH: An Extensive Benchmark for Scientific Machine Learning
    Source: https://github.com/pdebench/PDEBench/blob/main/pdebench/models/unet/unet.py

If you use this implementation, please cite original work above.
"""

from collections import OrderedDict

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from the_well.data.datasets import WellMetadata

conv_modules = {1: nn.Conv1d, 2: nn.Conv2d, 3: nn.Conv3d}
conv_transpose_modules = {
    1: nn.ConvTranspose1d,
    2: nn.ConvTranspose2d,
    3: nn.ConvTranspose3d,
}
pool_modules = {1: nn.MaxPool1d, 2: nn.MaxPool2d, 3: nn.MaxPool3d}
norm_modules = {1: nn.BatchNorm1d, 2: nn.BatchNorm2d, 3: nn.BatchNorm3d}


class UNetClassic(nn.Module):
    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        dset_metadata: WellMetadata,
        init_features: int = 32,
        gradient_checkpointing: bool = False,
    ):
        super(UNetClassic, self).__init__()
        self.dset_metadata = dset_metadata
        n_spatial_dims = dset_metadata.n_spatial_dims
        self.n_spatial_dims = n_spatial_dims
        self.gradient_checkpointing = gradient_checkpointing
        features = init_features
        self.encoder1 = self._block(dim_in, features, name="enc1")
        self.pool1 = pool_modules[n_spatial_dims](kernel_size=2, stride=2)
        self.encoder2 = self._block(features, features * 2, name="enc2")
        self.pool2 = pool_modules[n_spatial_dims](kernel_size=2, stride=2)
        self.encoder3 = self._block(features * 2, features * 4, name="enc3")
        self.pool3 = pool_modules[n_spatial_dims](kernel_size=2, stride=2)
        self.encoder4 = self._block(features * 4, features * 8, name="enc4")
        self.pool4 = pool_modules[n_spatial_dims](kernel_size=2, stride=2)

        self.bottleneck = self._block(features * 8, features * 16, name="bottleneck")

        self.upconv4 = conv_transpose_modules[n_spatial_dims](
            features * 16, features * 8, kernel_size=2, stride=2
        )
        self.decoder4 = self._block((features * 8) * 2, features * 8, name="dec4")
        self.upconv3 = conv_transpose_modules[n_spatial_dims](
            features * 8, features * 4, kernel_size=2, stride=2
        )
        self.decoder3 = self._block((features * 4) * 2, features * 4, name="dec3")
        self.upconv2 = conv_transpose_modules[n_spatial_dims](
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decoder2 = self._block((features * 2) * 2, features * 2, name="dec2")
        self.upconv1 = conv_transpose_modules[n_spatial_dims](
            features * 2, features, kernel_size=2, stride=2
        )
        self.decoder1 = self._block(features * 2, features, name="dec1")

        self.conv = conv_transpose_modules[n_spatial_dims](
            in_channels=features, out_channels=dim_out, kernel_size=1
        )

    def optional_checkpointing(self, layer, *inputs, **kwargs):
        if self.gradient_checkpointing:
            return checkpoint(layer, *inputs, use_reentrant=False, **kwargs)
        else:
            return layer(*inputs, **kwargs)

    def forward(self, x):
        enc1 = self.optional_checkpointing(self.encoder1, x)
        enc2 = self.optional_checkpointing(self.encoder2, self.pool1(enc1))
        enc3 = self.optional_checkpointing(self.encoder3, self.pool2(enc2))
        enc4 = self.optional_checkpointing(self.encoder4, self.pool3(enc3))

        bottleneck = self.optional_checkpointing(self.bottleneck, self.pool4(enc4))

        dec4 = self.optional_checkpointing(self.upconv4, bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.optional_checkpointing(self.decoder4, dec4)
        dec3 = self.optional_checkpointing(self.upconv3, dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.optional_checkpointing(self.decoder3, dec3)
        dec2 = self.optional_checkpointing(self.upconv2, dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.optional_checkpointing(self.decoder2, dec2)
        dec1 = self.optional_checkpointing(self.upconv1, dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.optional_checkpointing(self.decoder1, dec1)
        return self.conv(dec1)

    def _block(self, in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        conv_modules[self.n_spatial_dims](
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (
                        name + "norm1",
                        norm_modules[self.n_spatial_dims](num_features=features),
                    ),
                    (name + "tanh1", nn.Tanh()),
                    (
                        name + "conv2",
                        conv_modules[self.n_spatial_dims](
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (
                        name + "norm2",
                        norm_modules[self.n_spatial_dims](num_features=features),
                    ),
                    (name + "tanh2", nn.Tanh()),
                ]
            )
        )
