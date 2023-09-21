# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

from __future__ import annotations
from dataclasses import fields, dataclass, make_dataclass
from typing import Optional, List, Tuple, Set, Dict, Iterator
import torch
import types

from utils.tensor_dataclass import TensorDataclass


__TD_VARIANTS__ = dict()


@dataclass
class RenderBuffer(TensorDataclass):
    """
    A torch based, multi-channel, pixel buffer object.
    RenderBuffers are "smart" data buffers, used for accumulating tracing results, blending buffers of information,
    and providing discretized images.

    The spatial dimensions of RenderBuffer channels are flexible, see TensorDataclass.
    """

    rgb: Optional[torch.Tensor] = None
    """ rgb is a shaded RGB color. """

    alpha: Optional[torch.Tensor] = None
    """ alpha is the alpha component of RGB-A. """

    depth: Optional[torch.Tensor] = None
    """ depth is usually a distance to the surface hit point."""

    # Renderbuffer supports additional custom channels passed to the Renderbuffer constructor.
    # Some example of custom channels used throughout wisp:
    #     xyz=None,         # xyz is usually the xyz position for the surface hit point.
    #     hit=None,         # hit is usually a segmentation mask of hit points.
    #     normal=None,      # normal is usually the surface normal at the hit point.
    #     shadow =None,     # shadow is usually some additional buffer for shadowing.
    #     ao=None,          # ao is usually some addition buffer for ambient occlusion.
    #     ray_o=None,       # ray_o is usually the ray origin.
    #     ray_d=None,       # ray_d is usually the ray direction.
    #     err=None,         # err is usually some error metric against the ground truth.
    #     gts=None,         # gts is usually the ground truth image.

    def __new__(cls, *args, **kwargs):
        class_fields = [f.name for f in fields(RenderBuffer)]
        new_fields = [k for k in kwargs.keys() if k not in class_fields]
        if 0 < len(new_fields):
            class_key = frozenset(new_fields)
            rb_class = __TD_VARIANTS__.get(class_key)
            if rb_class is None:
                rb_class = make_dataclass(
                    f'RenderBuffer_{len(__TD_VARIANTS__)}',
                    fields=[
                        (
                            k,
                            Optional[torch.Tensor],
                            None,
                        )
                        for k in kwargs.keys()
                    ],
                    bases=(RenderBuffer,),
                )
                # Cache for future __new__ calls
                __TD_VARIANTS__[class_key] = rb_class
                setattr(types, rb_class.__name__, rb_class)
            return super(RenderBuffer, rb_class).__new__(rb_class)
        else:
            return super(TensorDataclass, cls).__new__(cls)

    @property
    def rgba(self) -> Optional[torch.Tensor]:
        """
        Returns:
            (Optional[torch.Tensor]) A concatenated rgba. If rgb or alpha are none, this property will return None.
        """
        if self.alpha is None or self.rgb is None:
            return None
        else:
            return torch.cat((self.rgb, self.alpha), dim=-1)

    @rgba.setter
    def rgba(self, val: Optional[torch.Tensor]) -> None:
        """
        Args:
            val (Optional[torch.Tensor]) A concatenated rgba channel value, which sets values for the rgb and alpha
            internal channels simultaneously.
        """
        self.rgb = val[..., 0:-1]
        self.alpha = val[..., -1:]

    @property
    def channels(self) -> Set[str]:
        """Returns a set of channels supported by this RenderBuffer"""
        all_channels = self.fields
        static_channels = self._static_field
        return all_channels.difference(static_channels)

    def has_channel(self, name: str) -> bool:
        """Returns whether the RenderBuffer supports the specified channel"""
        return name in self.channels

    def get_channel(self, name: str) -> Optional[torch.Tensor]:
        """Returns the pixels value of the specified channel,
        assuming this RenderBuffer supports the specified channel.
        """
        return getattr(self, name)

    def transpose(self) -> RenderBuffer:
        """Permutes dimensions 0 and 1 of each channel.
        The rest of the channel dimensions will remain in the same order.
        """
        fn = lambda x: x.permute(1, 0, *tuple(range(2, x.ndim)))
        return self._apply(fn)

    def scale(self, size: Tuple, interpolation='bilinear') -> RenderBuffer:
        """Upsamples or downsamples the renderbuffer pixels using the specified interpolation.
        Scaling assumes renderbuffers with 2 spatial dimensions, e.g. (H, W, C) or (W, H, C).

        Warning: for non-floating point channels, this function will upcast to floating point dtype
        to perform interpolation, and will then re-cast back to the original dtype.
        Hence truncations due to rounding may occur.

        Args:
            size (Tuple): The new spatial dimensions of the renderbuffer.
            interpolation (str): Interpolation method applied to cope with missing or decimated pixels due to
            up / downsampling. The interpolation methods supported are aligned with
            :func:`torch.nn.functional.interpolate`.

        Returns:
            (RenderBuffer): A new RenderBuffer object with rescaled channels.
        """

        def _scale(x):
            assert (
                x.ndim == 3
            ), 'RenderBuffer scale() assumes channels have 2D spatial dimensions.'
            # Some versions of torch don't support direct interpolation of non-fp tensors
            dtype = x.dtype
            if not torch.is_floating_point(x):
                x = x.float()
            x = x.permute(2, 0, 1)[None]
            x = torch.nn.functional.interpolate(
                x, size=size, mode=interpolation
            )
            x = x[0].permute(1, 2, 0)
            if x.dtype != dtype:
                x = torch.round(x).to(dtype)
            return x

        return self._apply(_scale)

    def exr_dict(self) -> Dict[str, torch.Tensor]:
        """This function returns an EXR format compatible dictionary.

        Returns:
            (Dict[str, torch.Tensor])
                a dictionary suitable for use with `pyexr` to output multi-channel EXR images which can be
                viewed interactively with software like `tev`.
                This is suitable for debugging geometric quantities like ray origins and ray directions.
        """
        _dict = self.numpy_dict()
        if 'rgb' in _dict:
            _dict['default'] = _dict['rgb']
            del _dict['rgb']
        return _dict

    def image(self) -> RenderBuffer:
        """This function will return a copy of the RenderBuffer which will contain 8-bit [0,255] images.

        This function is used to output a RenderBuffer suitable for saving as a 8-bit RGB image (e.g. with
        Pillow). Since this quantization operation permanently loses information, this is not an inplace
        operation and will return a copy of the RenderBuffer. Currently this function will only return
        the hit segmentation mask, normalized depth, RGB, and the surface normals.

        If users want custom behaviour, users can implement their own conversion function which takes a
        RenderBuffer as input.
        """
        norm = lambda arr: ((arr + 1.0) / 2.0) if arr is not None else None
        bwrgb = (
            lambda arr: torch.cat([arr] * 3, dim=-1)
            if arr is not None
            else None
        )
        rgb8 = lambda arr: (arr * 255.0) if arr is not None else None

        channels = dict()
        if self.rgb is not None:
            channels['rgb'] = rgb8(self.rgb)
        if self.alpha is not None:
            channels['alpha'] = rgb8(self.alpha)
        if self.depth is not None:
            # If the relative depth is respect to some camera clipping plane, the depth should
            # be clipped in advance.
            relative_depth = self.depth / (torch.max(self.depth) + 1e-8)
            channels['depth'] = rgb8(bwrgb(relative_depth))

        if hasattr(self, 'hit') and self.hit is not None:
            channels['hit'] = rgb8(bwrgb(self.hit))
        else:
            channels['hit'] = None
        if hasattr(self, 'normal') and self.normal is not None:
            channels['normal'] = rgb8(norm(self.normal))
        else:
            channels['normal'] = None

        return RenderBuffer(**channels)
