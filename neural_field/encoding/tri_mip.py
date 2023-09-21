import torch
from torch import nn

import nvdiffrast.torch


class TriMipEncoding(nn.Module):
    def __init__(
        self,
        n_levels: int,
        plane_size: int,
        feature_dim: int,
        include_xyz: bool = False,
    ):
        super(TriMipEncoding, self).__init__()
        self.n_levels = n_levels
        self.plane_size = plane_size
        self.feature_dim = feature_dim
        self.include_xyz = include_xyz

        self.register_parameter(
            "fm",
            nn.Parameter(torch.zeros(3, plane_size, plane_size, feature_dim)),
        )
        self.init_parameters()
        self.dim_out = (
            self.feature_dim * 3 + 3 if include_xyz else self.feature_dim * 3
        )

    def init_parameters(self) -> None:
        # Important for performance
        nn.init.uniform_(self.fm, -1e-2, 1e-2)

    def forward(self, x, level):
        # x in [0,1], level in [0,max_level]
        # x is Nx3, level is Nx1
        if 0 == x.shape[0]:
            return torch.zeros([x.shape[0], self.feature_dim * 3]).to(x)
        decomposed_x = torch.stack(
            [
                x[:, None, [1, 2]],
                x[:, None, [0, 2]],
                x[:, None, [0, 1]],
            ],
            dim=0,
        )  # 3xNx1x2
        if 0 == self.n_levels:
            level = None
        else:
            # assert level.shape[0] > 0, [level.shape, x.shape]
            torch.stack([level, level, level], dim=0)
            level = torch.broadcast_to(
                level, decomposed_x.shape[:3]
            ).contiguous()
        enc = nvdiffrast.torch.texture(
            self.fm,
            decomposed_x,
            mip_level_bias=level,
            boundary_mode="clamp",
            max_mip_level=self.n_levels - 1,
        )  # 3xNx1xC
        enc = (
            enc.permute(1, 2, 0, 3)
            .contiguous()
            .view(
                x.shape[0],
                self.feature_dim * 3,
            )
        )  # Nx(3C)
        if self.include_xyz:
            enc = torch.cat([x, enc], dim=-1)
        return enc
