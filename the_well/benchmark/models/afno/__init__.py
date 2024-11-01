"""
reference: https://github.com/NVlabs/FourCastNet/

Simplified some unused parameters since they're
not used in original code.
"""

from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.models.layers import DropPath, trunc_normal_


class RealImagGELU(nn.Module):
    def forward(self, x):
        return torch.complex(F.gelu(x.real), F.gelu(x.imag))


class ComplexBlockLinear(nn.Module):
    def __init__(
        self,
        hidden_dim,
        bias=True,
        cmlp_diagonal_blocks=8,
    ):
        super().__init__()
        self.scale = 0.02  # Hardcoded in reference code
        self.hidden_dim = hidden_dim
        self.cmlp_diagonal_blocks = cmlp_diagonal_blocks
        self.block_size = self.hidden_dim // self.cmlp_diagonal_blocks
        self.weight = nn.Parameter(
            torch.view_as_real(
                self.scale
                * torch.randn(
                    cmlp_diagonal_blocks,
                    self.block_size,
                    self.block_size,
                    dtype=torch.cfloat,
                )
            )
        )

    def forward(self, x):
        x = x.reshape(*x.shape[:-1], self.cmlp_diagonal_blocks, self.block_size)
        x = torch.einsum("...bi,bio->...bo", x, torch.view_as_complex(self.weight))
        return x.reshape(*x.shape[:-2], -1)


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class AFNO_ND(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        resolution: tuple[int, ...],
        cmlp_diagonal_blocks=8,
        sparsity_threshold=0.01,
    ):
        super().__init__()
        assert (
            hidden_size % cmlp_diagonal_blocks == 0
        ), f"hidden_size {hidden_size} should be divisble by cmlp_diagonal_blocks {cmlp_diagonal_blocks}"

        self.resolution = resolution
        self.hidden_size = hidden_size
        self.sparsity_threshold = sparsity_threshold
        self.cmlp_diagonal_blocks = cmlp_diagonal_blocks
        self.scale = 0.02

        self.cmlp = nn.Sequential(
            ComplexBlockLinear(hidden_size, cmlp_diagonal_blocks=cmlp_diagonal_blocks),
            RealImagGELU(),
            ComplexBlockLinear(hidden_size, cmlp_diagonal_blocks=cmlp_diagonal_blocks),
        )

    def forward(self, x):
        dtype = x.dtype
        x = x.float()
        spatial_dims = tuple(range(1, len(x.shape) - 1))[::-1]

        x = torch.fft.rfftn(x, dim=spatial_dims, norm="ortho")
        x = self.cmlp(x)
        x = torch.view_as_real(x)
        x = F.softshrink(x, lambd=self.sparsity_threshold)
        x = torch.view_as_complex(x)
        x = torch.fft.irfftn(
            x, s=tuple(self.resolution), dim=spatial_dims, norm="ortho"
        )
        x = x.type(dtype)
        return x


class Block(nn.Module):
    def __init__(
        self,
        hidden_dim,
        resolution,
        mlp_ratio=4.0,
        drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        double_skip=True,
        cmlp_diagonal_blocks=8,
        sparsity_threshold=0.01,
    ):
        super().__init__()
        self.norm1 = norm_layer(hidden_dim)
        self.filter = AFNO_ND(
            hidden_dim, resolution, cmlp_diagonal_blocks, sparsity_threshold
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(hidden_dim)
        mlp_hidden_dim = int(hidden_dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=hidden_dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )
        self.double_skip = double_skip

    def forward(self, x):
        residual = x
        x = self.norm1(x)
        x = self.filter(x)

        if self.double_skip:
            x = x + residual
            residual = x

        x = self.norm2(x)
        x = self.mlp(x)
        x = self.drop_path(x)
        x = x + residual
        return x


class AFNO(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        dset_metadata,
        hidden_dim=768,
        n_blocks=12,  # Depth in original code - changing for consistency
        cmlp_diagonal_blocks=8,  # num_blocks in original
        patch_size=8,
        mlp_ratio=4.0,
        drop_rate=0.0,
        drop_path_rate=0.0,
        sparsity_threshold=0.01,
    ):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.resolution = dset_metadata.spatial_resolution
        self.n_spatial_dims = dset_metadata.n_spatial_dims
        self.n_blocks = n_blocks
        self.cmlp_diagonal_blocks = cmlp_diagonal_blocks
        norm_layer = partial(nn.LayerNorm, eps=1e-6)

        # Dimension dependent things
        if self.n_spatial_dims == 2:
            self.patch_embed = nn.Conv2d(
                dim_in, hidden_dim, kernel_size=patch_size, stride=patch_size
            )
            self.embed_permutation = [
                "b h w c -> b c h w",
                "b c h w -> b h w c",
            ]
            self.patch_debed = nn.ConvTranspose2d(
                hidden_dim, dim_out, kernel_size=patch_size, stride=patch_size
            )

        elif self.n_spatial_dims == 3:
            self.patch_embed = nn.Conv3d(
                dim_in, hidden_dim, kernel_size=patch_size, stride=patch_size
            )
            self.embed_permutation = [
                "b h w d c -> b c h w d",
                "b c h w d -> b h w d c",
            ]
            self.patch_debed = nn.ConvTranspose3d(
                hidden_dim, dim_out, kernel_size=patch_size, stride=patch_size
            )
        self.inner_size = [k // patch_size for k in self.resolution]
        pos_embed_size = [1] + self.inner_size + [hidden_dim]
        self.pos_embed = nn.Parameter(0.02 * torch.randn(pos_embed_size))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, n_blocks)]

        self.blocks = nn.ModuleList(
            [
                Block(
                    hidden_dim=hidden_dim,
                    resolution=self.inner_size,
                    mlp_ratio=mlp_ratio,
                    drop=drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    cmlp_diagonal_blocks=self.cmlp_diagonal_blocks,
                    sparsity_threshold=sparsity_threshold,
                )
                for i in range(n_blocks)
            ]
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token"}

    def forward_features(self, x):
        # Patch and bias
        x = rearrange(x, self.embed_permutation[0])
        x = self.patch_embed(x)
        x = rearrange(x, self.embed_permutation[1])
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        return x

    def forward(self, x):
        x = self.forward_features(x)
        # Debed
        x = rearrange(x, self.embed_permutation[0])
        x = self.patch_debed(x)
        x = rearrange(x, self.embed_permutation[1])
        return x
