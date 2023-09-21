from typing import Union, List, Optional, Callable

import gin
import torch
import nerfacc
from nerfacc import render_weight_from_density, accumulate_along_rays

from neural_field.model.RFModel import RFModel
from utils.ray import RayBundle
from utils.render_buffer import RenderBuffer
from neural_field.field.trimipRF import TriMipRF


@gin.configurable()
class TriMipRFModel(RFModel):
    def __init__(
        self,
        aabb: Union[torch.Tensor, List[float]],
        samples_per_ray: int = 1024,
        occ_grid_resolution: int = 128,
    ) -> None:
        super().__init__(aabb=aabb, samples_per_ray=samples_per_ray)
        self.field = TriMipRF()
        self.ray_sampler = nerfacc.OccupancyGrid(
            roi_aabb=self.aabb, resolution=occ_grid_resolution
        )

        self.feature_vol_radii = self.aabb_size[0] / 2.0
        self.register_buffer(
            "occ_level_vol",
            torch.log2(
                self.aabb_size[0]
                / occ_grid_resolution
                / 2.0
                / self.feature_vol_radii
            ),
        )

    def before_iter(self, step):
        # update_ray_sampler
        self.ray_sampler.every_n_step(
            step=step,
            occ_eval_fn=lambda x: self.field.query_density(
                x=self.contraction(x),
                level_vol=torch.empty_like(x[..., 0]).fill_(self.occ_level_vol),
            )['density']
            * self.render_step_size,
            occ_thre=5e-3,
        )

    @staticmethod
    def compute_ball_radii(distance, radiis, cos):
        inverse_cos = 1.0 / cos
        tmp = (inverse_cos * inverse_cos - 1).sqrt() - radiis
        sample_ball_radii = distance * radiis * cos / (tmp * tmp + 1.0).sqrt()
        return sample_ball_radii

    def forward(
        self,
        rays: RayBundle,
        background_color=None,
        alpha_thre=0.0,
        ray_marching_aabb=None,
    ):
        # Ray sampling with occupancy grid
        with torch.no_grad():

            def sigma_fn(t_starts, t_ends, ray_indices):
                ray_indices = ray_indices.long()
                t_origins = rays.origins[ray_indices]
                t_dirs = rays.directions[ray_indices]
                radiis = rays.radiis[ray_indices]
                cos = rays.ray_cos[ray_indices]
                distance = (t_starts + t_ends) / 2.0
                positions = t_origins + t_dirs * distance
                positions = self.contraction(positions)
                sample_ball_radii = self.compute_ball_radii(
                    distance, radiis, cos
                )
                level_vol = torch.log2(
                    sample_ball_radii / self.feature_vol_radii
                )  # real level should + log2(feature_resolution)
                return self.field.query_density(positions, level_vol)['density']

            ray_indices, t_starts, t_ends = nerfacc.ray_marching(
                rays.origins,
                rays.directions,
                scene_aabb=self.aabb,
                grid=self.ray_sampler,
                sigma_fn=sigma_fn,
                render_step_size=self.render_step_size,
                stratified=self.training,
                early_stop_eps=1e-4,
            )

        # Ray rendering
        def rgb_sigma_fn(t_starts, t_ends, ray_indices):
            t_origins = rays.origins[ray_indices]
            t_dirs = rays.directions[ray_indices]
            radiis = rays.radiis[ray_indices]
            cos = rays.ray_cos[ray_indices]
            distance = (t_starts + t_ends) / 2.0
            positions = t_origins + t_dirs * distance
            positions = self.contraction(positions)
            sample_ball_radii = self.compute_ball_radii(distance, radiis, cos)
            level_vol = torch.log2(
                sample_ball_radii / self.feature_vol_radii
            )  # real level should + log2(feature_resolution)
            res = self.field.query_density(
                x=positions,
                level_vol=level_vol,
                return_feat=True,
            )
            density, feature = res['density'], res['feature']
            rgb = self.field.query_rgb(dir=t_dirs, embedding=feature)['rgb']
            return rgb, density

        return self.rendering(
            t_starts,
            t_ends,
            ray_indices,
            rays,
            rgb_sigma_fn=rgb_sigma_fn,
            render_bkgd=background_color,
        )

    def rendering(
        self,
        # ray marching results
        t_starts: torch.Tensor,
        t_ends: torch.Tensor,
        ray_indices: torch.Tensor,
        rays: RayBundle,
        # radiance field
        rgb_sigma_fn: Callable = None,  # rendering options
        render_bkgd: Optional[torch.Tensor] = None,
    ) -> RenderBuffer:
        n_rays = rays.origins.shape[0]
        # Query sigma/alpha and color with gradients
        rgbs, sigmas = rgb_sigma_fn(t_starts, t_ends, ray_indices.long())

        # Rendering
        weights = render_weight_from_density(
            t_starts,
            t_ends,
            sigmas,
            ray_indices=ray_indices,
            n_rays=n_rays,
        )
        sample_buffer = {
            'num_samples': torch.as_tensor(
                [len(t_starts)], dtype=torch.int32, device=rgbs.device
            ),
        }

        # Rendering: accumulate rgbs, opacities, and depths along the rays.
        colors = accumulate_along_rays(
            weights, ray_indices=ray_indices, values=rgbs, n_rays=n_rays
        )
        opacities = accumulate_along_rays(
            weights, values=None, ray_indices=ray_indices, n_rays=n_rays
        )
        opacities.clamp_(
            0.0, 1.0
        )  # sometimes it may slightly bigger than 1.0, which will lead abnormal behaviours

        depths = accumulate_along_rays(
            weights,
            ray_indices=ray_indices,
            values=(t_starts + t_ends) / 2.0,
            n_rays=n_rays,
        )
        depths = (
            depths * rays.ray_cos
        )  # from distance to real depth (z value in camera space)

        # Background composition.
        if render_bkgd is not None:
            colors = colors + render_bkgd * (1.0 - opacities)

        return RenderBuffer(
            rgb=colors,
            alpha=opacities,
            depth=depths,
            **sample_buffer,
            _static_field=set(sample_buffer),
        )

    @gin.configurable()
    def get_optimizer(
        self, lr=2e-3, weight_decay=1e-5, feature_lr_scale=10.0, **kwargs
    ):
        params_list = []
        params_list.append(
            dict(
                params=self.field.encoding.parameters(),
                lr=lr * feature_lr_scale,
            )
        )
        params_list.append(
            dict(params=self.field.direction_encoding.parameters(), lr=lr)
        )
        params_list.append(dict(params=self.field.mlp_base.parameters(), lr=lr))
        params_list.append(dict(params=self.field.mlp_head.parameters(), lr=lr))

        optim = torch.optim.AdamW(
            params_list,
            weight_decay=weight_decay,
            **kwargs,
            eps=1e-15,
        )
        return optim
