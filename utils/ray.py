import torch
from dataclasses import dataclass
from typing import Optional

from utils.tensor_dataclass import TensorDataclass


@dataclass
class RayBundle(TensorDataclass):
    origins: Optional[torch.Tensor] = None
    """Ray origins (XYZ)"""

    directions: Optional[torch.Tensor] = None
    """Unit ray direction vector"""

    radiis: Optional[torch.Tensor] = None
    """Ray image plane intersection circle radii"""

    ray_cos: Optional[torch.Tensor] = None
    """Ray cos"""

    def __len__(self):
        num_rays = torch.numel(self.origins) // self.origins.shape[-1]
        return num_rays

    @property
    def shape(self):
        return list(super().shape)


@dataclass
class RayBundleExt(RayBundle):

    ray_depth: Optional[torch.Tensor] = None


@dataclass
class RayBundleRast(RayBundleExt):

    ray_uv: Optional[torch.Tensor] = None
    ray_mip_level: Optional[torch.Tensor] = None
