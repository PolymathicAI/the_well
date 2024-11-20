import torch
import torch.nn as nn
import torch.nn.functional as F

from the_well.data.datasets import WellMetadata

from ..common import SN_MLP, SigmaNormLinear


def filter_reconstruction(x, mag_bias, phase_bias):
    mag, phase = x.abs(), x.angle()
    return torch.polar(torch.sigmoid(mag + mag_bias), phase + phase_bias)


def get_token_mask_from_resolution_rectangle(resolution, filter_ratio=1.0):
    max_res = int(max(resolution) / 2 * filter_ratio)
    fft_freqs = []
    for i, res in enumerate(resolution):
        if i == 0:
            fft_freqs.append(res * torch.fft.rfftfreq(res))
        else:
            fft_freqs.append(res * torch.fft.fftfreq(res))
    fft_freqs = torch.stack(torch.meshgrid(*fft_freqs), 0)
    fft_freqs = torch.sum(fft_freqs**2, 0)
    mask = fft_freqs <= max_res**2

    return mask


class ComplexLN(nn.Module):
    def __init__(self, channels, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.weights = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(
            torch.view_as_real(torch.zeros(channels, dtype=torch.cfloat))
        )

    def forward(self, x):
        std, mean = torch.std_mean(x, dim=-1, keepdim=True)
        x = (x - mean) / (self.eps + std)
        x = x * self.weights + torch.view_as_complex(self.bias)
        return x


class ComplexLinearDDP(nn.Module):
    def __init__(self, in_features, out_features, bias=True, device=None, dtype=None):
        super().__init__()
        temp = nn.Linear(in_features, out_features, bias=bias, dtype=torch.cfloat)
        self.weights = nn.Parameter(torch.view_as_real(temp.weight))
        self.bias = nn.Parameter(torch.view_as_real(temp.bias))

    def forward(self, input):
        return F.linear(
            input, torch.view_as_complex(self.weights), torch.view_as_complex(self.bias)
        )


class ModReLU(torch.nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.b = nn.Parameter(0.02 * torch.randn(channels))

    def forward(self, x):
        return torch.polar(F.relu(torch.abs(x) + self.b), x.angle())


class DSConvSpectralNd(nn.Module):
    def __init__(self, hidden_dim, resolution, ratio=1.0):
        super(DSConvSpectralNd, self).__init__()
        self.resolution = resolution
        self.register_buffer(
            "token_mask", get_token_mask_from_resolution_rectangle(resolution, ratio)
        )
        self.n_tokens = self.token_mask.sum()
        self.hidden_dim = hidden_dim

        self.filter_generator = nn.Sequential(
            ComplexLN(hidden_dim, self.n_tokens),
            ComplexLinearDDP(hidden_dim, hidden_dim),
            ModReLU(hidden_dim),
            ComplexLinearDDP(hidden_dim, hidden_dim),
        )

        temp_conv = torch.randn(1, self.n_tokens, hidden_dim, dtype=torch.cfloat)
        self.filter_mag_bias = nn.Parameter(torch.abs(temp_conv))
        self.filter_phase_bias = nn.Parameter(temp_conv.angle())
        # Output mixing
        self.output_mixer = SigmaNormLinear(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor):
        # Get shape and tpye info
        spatial_dims = tuple(range(1, len(x.shape) - 1))[::-1]  # Assuming B ... C
        # Convert to frequency and mask
        x_ft = torch.fft.rfftn(x, dim=spatial_dims, norm="ortho")
        broadcastable_mask = self.token_mask.unsqueeze(0).unsqueeze(-1)
        x_masked = torch.masked_select(x_ft, broadcastable_mask).reshape(
            -1, self.n_tokens, self.hidden_dim
        )
        # Generate and apply data-dependent filter
        x_linear = self.filter_generator(x_masked)
        conv = filter_reconstruction(
            x_linear, self.filter_mag_bias, self.filter_phase_bias
        )
        x_masked = x_masked * conv
        # Reconstruct original input
        x_ft = torch.zeros_like(x_ft)
        x_ft[:, self.token_mask, :] = x_masked
        x = torch.fft.irfftn(x_ft, dim=spatial_dims, norm="ortho")
        return self.output_mixer(x)


class ReFNOBlock(nn.Module):
    def __init__(self, dim, resolution, ratio=1.0):
        super(ReFNOBlock, self).__init__()
        self.mlp = SN_MLP(dim, exp_factor=1)
        self.spectral_conv = DSConvSpectralNd(dim, resolution, ratio)

    def forward(self, x):
        return self.spectral_conv(self.mlp(x))


class ReFNO(nn.Module):
    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        dset_metadata: WellMetadata,
        hidden_dim: int = 64,
        blocks: int = 4,
        ratio: float = 1.0,
    ):
        super(ReFNO, self).__init__()
        """

        """
        self.resolution = tuple(dset_metadata.spatial_resolution)
        self.encoder = SigmaNormLinear(dim_in, hidden_dim)
        self.pos_embedding = nn.Parameter(
            torch.randn(self.resolution + (hidden_dim,)) * 0.02
        )
        self.processor_blocks = nn.ModuleList(
            [ReFNOBlock(hidden_dim, self.resolution, ratio) for _ in range(blocks)]
        )
        self.decoder = SigmaNormLinear(hidden_dim, dim_out)

    def forward(self, x, *args, **kwargs):
        """
        (b,c,h,w) -> (b,1,h,w)
        """
        x = self.encoder(x)  # project
        x = x + self.pos_embedding
        for process in self.processor_blocks:
            x = x + process(x)

        x = self.decoder(x)
        return x
