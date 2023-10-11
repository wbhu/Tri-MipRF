import math
from typing import Union, List, Dict

import gin
import torch
from torch import nn
import torch.nn.functional as F
from torchmetrics.functional import peak_signal_noise_ratio, structural_similarity_index_measure  # For torchmetrics lower version 2.0
from torchmetrics.image import lpip, ssim, psnr  # For torchmetrics higher version 2.0

from utils.ray import RayBundle
from utils.render_buffer import RenderBuffer


# @gin.configurable()
class RFModel(nn.Module):
    def __init__(
        self,
        aabb: Union[torch.Tensor, List[float]],
        samples_per_ray: int = 1024,
    ) -> None:
        super().__init__()
        if not isinstance(aabb, torch.Tensor):
            aabb = torch.tensor(aabb, dtype=torch.float32)
        self.register_buffer("aabb", aabb)
        self.samples_per_ray = samples_per_ray
        self.render_step_size = (
            (self.aabb[3:] - self.aabb[:3]).max()
            * math.sqrt(3)
            / samples_per_ray
        ).item()
        aabb_min, aabb_max = torch.split(self.aabb, 3, dim=-1)
        self.aabb_size = aabb_max - aabb_min
        assert (
            self.aabb_size[0] == self.aabb_size[1] == self.aabb_size[2]
        ), "Current implementation only supports cube aabb"
        self.field = None
        self.ray_sampler = None

    def contraction(self, x):
        aabb_min, aabb_max = self.aabb[:3].unsqueeze(0), self.aabb[
            3:
        ].unsqueeze(0)
        x = (x - aabb_min) / (aabb_max - aabb_min)
        return x

    def before_iter(self, step):
        pass

    def after_iter(self, step):
        pass

    def forward(
        self,
        rays: RayBundle,
        background_color=None,
    ):
        raise NotImplementedError

    @gin.configurable()
    def get_optimizer(
        self, lr=1e-3, weight_decay=1e-5, feature_lr_scale=10.0, **kwargs
    ):
        raise NotImplementedError

    @gin.configurable()
    def compute_loss(
        self,
        rays: RayBundle,
        rb: RenderBuffer,
        target: RenderBuffer,
        # Configurable
        metric='smooth_l1',
        **kwargs
    ) -> Dict:
        if 'smooth_l1' == metric:
            loss_fn = F.smooth_l1_loss
        elif 'mse' == metric:
            loss_fn = F.mse_loss
        elif 'mae' == metric:
            loss_fn = F.l1_loss
        else:
            raise NotImplementedError

        alive_ray_mask = (rb.alpha.squeeze(-1) > 0).detach()
        loss = loss_fn(
            rb.rgb[alive_ray_mask], target.rgb[alive_ray_mask], reduction='none'
        )
        loss = (
            loss * target.loss_multi[alive_ray_mask]
        ).sum() / target.loss_multi[alive_ray_mask].sum()
        return {'total_loss': loss}

    @gin.configurable()
    def compute_metrics(
        self,
        rays: RayBundle,
        rb: RenderBuffer,
        target: RenderBuffer,
        # Configurable
        **kwargs
    ) -> Dict:
        # ray info
        alive_ray_mask = (rb.alpha.squeeze(-1) > 0).detach()
        rendering_samples_actual = rb.num_samples[0].item()
        ray_info = {
            'num_alive_ray': alive_ray_mask.long().sum().item(),
            'rendering_samples_actual': rendering_samples_actual,
            'num_rays': len(target),
        }
        # initialize the metrics model
        lpips = lpip.LearnedPerceptualImagePatchSimilarity(net_type='vgg')
        pred = rb.rgb.permute(2, 0, 1).unsqueeze(0).cpu()
        gt = target.rgb.permute(2, 0, 1).unsqueeze(0).cpu()
        # quality
        quality = {'PSNR': peak_signal_noise_ratio(rb.rgb, target.rgb).item(),
                   'SSIM': structural_similarity_index_measure(pred, gt).item(),
                   'LPIPS': lpips(pred, gt).item()}
        return {**ray_info, **quality}
