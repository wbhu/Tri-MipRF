import math
from typing import Callable

import gin
import torch
from torch import Tensor, nn
import tinycudann as tcnn

from neural_field.encoding.tri_mip import TriMipEncoding
from neural_field.nn_utils.activations import trunc_exp


@gin.configurable()
class TriMipRF(nn.Module):
    def __init__(
        self,
        n_levels: int = 8,
        plane_size: int = 512,
        feature_dim: int = 16,
        geo_feat_dim: int = 15,
        net_depth_base: int = 2,
        net_depth_color: int = 4,
        net_width: int = 128,
        density_activation: Callable = lambda x: trunc_exp(x - 1),
    ) -> None:
        super().__init__()
        self.plane_size = plane_size
        self.log2_plane_size = math.log2(plane_size)
        self.geo_feat_dim = geo_feat_dim
        self.density_activation = density_activation

        self.encoding = TriMipEncoding(n_levels, plane_size, feature_dim)
        self.direction_encoding = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={
                "otype": "SphericalHarmonics",
                "degree": 4,
            },
        )
        self.mlp_base = tcnn.Network(
            n_input_dims=self.encoding.dim_out,
            n_output_dims=geo_feat_dim + 1,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": net_width,
                "n_hidden_layers": net_depth_base,
            },
        )
        self.mlp_head = tcnn.Network(
            n_input_dims=self.direction_encoding.n_output_dims + geo_feat_dim,
            n_output_dims=3,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "Sigmoid",
                "n_neurons": net_width,
                "n_hidden_layers": net_depth_color,
            },
        )

    def query_density(
        self, x: Tensor, level_vol: Tensor, return_feat: bool = False
    ):
        level = (
            level_vol if level_vol is None else level_vol + self.log2_plane_size
        )
        selector = ((x > 0.0) & (x < 1.0)).all(dim=-1)
        enc = self.encoding(
            x.view(-1, 3),
            level=level.view(-1, 1),
        )
        x = (
            self.mlp_base(enc)
            .view(list(x.shape[:-1]) + [1 + self.geo_feat_dim])
            .to(x)
        )
        density_before_activation, base_mlp_out = torch.split(
            x, [1, self.geo_feat_dim], dim=-1
        )
        density = (
            self.density_activation(density_before_activation)
            * selector[..., None]
        )
        return {
            "density": density,
            "feature": base_mlp_out if return_feat else None,
        }

    def query_rgb(self, dir, embedding):
        # dir in [-1,1]
        dir = (dir + 1.0) / 2.0  # SH encoding must be in the range [0, 1]
        d = self.direction_encoding(dir.view(-1, dir.shape[-1]))
        h = torch.cat([d, embedding.view(-1, self.geo_feat_dim)], dim=-1)
        rgb = (
            self.mlp_head(h)
            .view(list(embedding.shape[:-1]) + [3])
            .to(embedding)
        )
        return {"rgb": rgb}
