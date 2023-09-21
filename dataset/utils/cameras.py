import torch
import torch.nn.functional as F
import numpy as np

from utils.ray import RayBundle


class PinholeCamera:
    def __init__(
        self,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        width: int = None,
        height: int = None,
        coord_type: str = 'opengl',
        device: str = 'cuda:0',
        normalize_ray: bool = True,
    ):
        self.fx, self.fy, self.cx, self.cy = fx, fy, cx, cy
        self.width, self.height = width, height
        self.coord_type = coord_type
        self.K = torch.tensor(
            [
                [self.fx, 0, self.cx],
                [0, self.fy, self.cy],
                [0, 0, 1],
            ],
            dtype=torch.float32,
        )  # (3, 3)
        self.device = device
        self.normalize_ray = normalize_ray
        self.near = 0.1
        self.far = 100
        if self.coord_type == 'opencv':
            self.sign_z = 1.0
        elif self.coord_type == 'opengl':
            self.sign_z = -1.0
        else:
            raise ValueError
        self.ray_bundle = None

    def build(self, device):
        x, y = torch.meshgrid(
            torch.arange(self.width, device=device),
            torch.arange(self.height, device=device),
            indexing="xy",
        )
        directions = F.pad(
            torch.stack(
                [
                    (x - self.K[0, 2] + 0.5) / self.K[0, 0],
                    (y - self.K[1, 2] + 0.5) / self.K[1, 1] * self.sign_z,
                ],
                dim=-1,
            ),
            (0, 1),
            value=self.sign_z,
        )  # [H,W,3]
        # Distance from each unit-norm direction vector to its x-axis neighbor
        dx = torch.linalg.norm(
            (directions[:, :-1, :] - directions[:, 1:, :]),
            dim=-1,
            keepdims=True,
        )  # [H,W-1,1]
        dx = torch.cat([dx, dx[:, -2:-1, :]], 1)  # [H,W,1]
        dy = torch.linalg.norm(
            (directions[:-1, :, :] - directions[1:, :, :]),
            dim=-1,
            keepdims=True,
        )  # [H-1,W,1]
        dy = torch.cat([dy, dy[-2:-1, :, :]], 0)  # [H,W,1]
        # Cut the distance in half, and then round it out so that it's
        # halfway between inscribed by / circumscribed about the pixel.
        area = dx * dy
        radii = torch.sqrt(area / torch.pi)
        if self.normalize_ray:
            directions = directions / torch.linalg.norm(
                directions, dim=-1, keepdims=True
            )
        self.ray_bundle = RayBundle(
            origins=torch.zeros_like(directions),
            directions=directions,
            radiis=radii,
            ray_cos=torch.matmul(
                directions,
                torch.tensor([[0.0, 0.0, self.sign_z]], device=device).T,
            ),
        )
        return self.ray_bundle

    @property
    def fov_y(self):
        return np.degrees(2 * np.arctan(self.cy / self.fy))

    def get_proj(self):
        # projection
        proj = np.eye(4, dtype=np.float32)
        proj[0, 0] = 2 * self.fx / self.width
        proj[1, 1] = 2 * self.fy / self.height
        proj[0, 2] = 2 * self.cx / self.width - 1
        proj[1, 2] = 2 * self.cy / self.height - 1
        proj[2, 2] = -(self.far + self.near) / (self.far - self.near)
        proj[2, 3] = -2 * self.far * self.near / (self.far - self.near)
        proj[3, 2] = -1
        proj[3, 3] = 0
        return proj

    def get_PVM(self, c2w):
        c2w = c2w.copy()
        # to right up backward (opengl)
        c2w[:3, 1] *= -1
        c2w[:3, 2] *= -1
        w2c = np.linalg.inv(c2w)
        return np.matmul(self.get_proj(), w2c)
