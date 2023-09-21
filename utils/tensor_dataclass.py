# Copyright 2022 The Nerfstudio Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tensor dataclass"""

import dataclasses
from dataclasses import dataclass
from typing import (
    Dict,
    Set,
    List,
    NoReturn,
    Optional,
    Tuple,
    TypeVar,
    Union,
    Iterator,
)

import numpy as np
import torch

TensorDataclassT = TypeVar("TensorDataclassT", bound="TensorDataclass")


@dataclass
class TensorDataclass:
    """@dataclass of tensors with the same size batch. Allows indexing and standard tensor ops.
    Fields that are not Tensors will not be batched unless they are also a TensorDataclass.

    Example:

    .. code-block:: python

        @dataclass
        class TestTensorDataclass(TensorDataclass):
            a: torch.Tensor
            b: torch.Tensor
            c: torch.Tensor = None

        # Create a new tensor dataclass with batch size of [2,3,4]
        test = TestTensorDataclass(a=torch.ones((2, 3, 4, 2)), b=torch.ones((4, 3)))

        test.shape  # [2, 3, 4]
        test.a.shape  # [2, 3, 4, 2]
        test.b.shape  # [2, 3, 4, 3]

        test.reshape((6,4)).shape  # [6, 4]
        test.flatten().shape  # [24,]

        test[..., 0].shape  # [2, 3]
        test[:, 0, :].shape  # [2, 4]
    """

    _shape: tuple = torch.Size([])
    _static_field: set = dataclasses.field(default_factory=set)

    def __post_init__(self) -> None:
        """Finishes setting up the TensorDataclass

        This will 1) find the broadcasted shape and 2) broadcast all fields to this shape 3)
        set _shape to be the broadcasted shape.
        """
        if not dataclasses.is_dataclass(self):
            raise TypeError("TensorDataclass must be a dataclass")

        batch_shapes = self._get_dict_batch_shapes(
            {
                f.name: self.__getattribute__(f.name)
                for f in dataclasses.fields(self)
            }
        )
        if len(batch_shapes) == 0:
            raise ValueError("TensorDataclass must have at least one tensor")
        try:
            batch_shape = torch.broadcast_shapes(*batch_shapes)

            broadcasted_fields = self._broadcast_dict_fields(
                {
                    f.name: self.__getattribute__(f.name)
                    for f in dataclasses.fields(self)
                },
                batch_shape,
            )
            for f, v in broadcasted_fields.items():
                self.__setattr__(f, v)

            self.__setattr__("_shape", batch_shape)
        except RuntimeError:
            pass
        except IndexError:
            # import ipdb;ipdb.set_trace()
            pass

    def _get_dict_batch_shapes(self, dict_: Dict) -> List:
        """Returns batch shapes of all tensors in a dictionary

        Args:
            dict_: The dictionary to get the batch shapes of.

        Returns:
            The batch shapes of all tensors in the dictionary.
        """
        batch_shapes = []
        for k, v in dict_.items():
            if k in self._static_field:
                continue
            if isinstance(v, torch.Tensor):
                batch_shapes.append(v.shape[:-1])
            elif isinstance(v, TensorDataclass):
                batch_shapes.append(v.shape)
        return batch_shapes

    def _broadcast_dict_fields(self, dict_: Dict, batch_shape) -> Dict:
        """Broadcasts all tensors in a dictionary according to batch_shape

        Args:
            dict_: The dictionary to broadcast.

        Returns:
            The broadcasted dictionary.
        """
        new_dict = {}
        for k, v in dict_.items():
            if k in self._static_field:
                continue
            if isinstance(v, torch.Tensor):
                new_dict[k] = v.broadcast_to((*batch_shape, v.shape[-1]))
            elif isinstance(v, TensorDataclass):
                new_dict[k] = v.broadcast_to(batch_shape)
        return new_dict

    def __getitem__(self: TensorDataclassT, indices) -> TensorDataclassT:
        if isinstance(indices, torch.Tensor):
            return self._apply_exclude_static(lambda x: x[indices])
        if isinstance(indices, (int, slice)):
            indices = (indices,)
        return self._apply_exclude_static(lambda x: x[indices + (slice(None),)])

    def __setitem__(self, indices, value) -> NoReturn:
        raise RuntimeError(
            "Index assignment is not supported for TensorDataclass"
        )

    def __len__(self) -> int:
        return self.shape[0]

    def __bool__(self) -> bool:
        if len(self) == 0:
            raise ValueError(
                f"The truth value of {self.__class__.__name__} when `len(x) == 0` "
                "is ambiguous. Use `len(x)` or `x is not None`."
            )
        return True

    def __iter__(self) -> Iterator[Tuple[str, Optional[torch.Tensor]]]:
        return iter(
            (f.name, getattr(self, f.name))
            for f in dataclasses.fields(self)
            if f.name not in ('_shape', '_static_field')
        )

    @property
    def shape(self) -> Tuple[int, ...]:
        """Returns the batch shape of the tensor dataclass."""
        return self._shape

    @property
    def size(self) -> int:
        """Returns the number of elements in the tensor dataclass batch dimension."""
        if len(self._shape) == 0:
            return 1
        return int(np.prod(self._shape))

    @property
    def ndim(self) -> int:
        """Returns the number of dimensions of the tensor dataclass."""
        return len(self._shape)

    @property
    def fields(self) -> Set[str]:
        return set([f[0] for f in self])

    def _apply(self, fn) -> TensorDataclassT:
        """Applies the function fn on each of the Renderbuffer channels, if not None.
        Returns a new instance with the processed channels.
        """
        data = {}
        for f in self:
            attr = f[1]
            data[f[0]] = None if attr is None else fn(attr)
        return dataclasses.replace(
            self,
            _static_field=self._static_field,
            **data,
        )
        # return TensorDataclass(**data)

    def _apply_exclude_static(self, fn) -> TensorDataclassT:
        data = {}
        for f in self:
            if f[0] in self._static_field:
                continue
            attr = f[1]
            data[f[0]] = None if attr is None else fn(attr)
        return dataclasses.replace(
            self,
            _static_field=self._static_field,
            **data,
        )

    @staticmethod
    def _apply_on_pair(td1, td2, fn) -> TensorDataclassT:
        """Applies the function fn on each of the Renderbuffer channels, if not None.
        Returns a new instance with the processed channels.
        """
        joint_fields = TensorDataclass._join_fields(
            td1, td2
        )  # Union of field names and tuples of values
        combined_channels = map(
            fn, joint_fields.values()
        )  # Invoke on pair per Renderbuffer field
        return dataclasses.replace(
            td1,
            _static_field=td1._static_field.union(td2._static_field),
            **dict(zip(joint_fields.keys(), combined_channels)),
        )
        # return TensorDataclass(**dict(zip(joint_fields.keys(), combined_channels)))    # Pack combined fields to a new rb

    @staticmethod
    def _apply_on_list(tds, fn) -> TensorDataclassT:
        joint_fields = set().union(*[td.fields for td in tds])
        joint_fields = {
            f: [getattr(td, f, None) for td in tds] for f in joint_fields
        }
        combined_channels = map(fn, joint_fields.values())

        return dataclasses.replace(
            tds[0],
            _static_field=tds[0]._static_field.union(
                *[td._static_field for td in tds[1:]]
            ),
            **dict(zip(joint_fields.keys(), combined_channels)),
        )

    @staticmethod
    def _join_fields(td1, td2):
        """Creates a joint mapping of renderbuffer fields in a format of
        {
            channel1_name: (rb1.c1, rb2.c1),
            channel2_name: (rb1.c2, rb2.cb),
            channel3_name: (rb1.c1, None),  # rb2 doesn't define channel3
        }
        If a renderbuffer does not have define a specific channel, None is returned.
        """
        joint_fields = td1.fields.union(td2.fields)
        return {
            f: (getattr(td1, f, None), getattr(td2, f, None))
            for f in joint_fields
        }

    def numpy_dict(self) -> Dict[str, np.array]:
        """This function returns a dictionary of numpy arrays containing the data of each channel.

        Returns:
            (Dict[str, numpy.Array])
                a dictionary with entries of (channel_name, channel_data)
        """
        _dict = dict(iter(self))
        _dict = {k: v.numpy() for k, v in _dict.items() if v is not None}
        return _dict

    def reshape(
            self: TensorDataclassT, shape: Tuple[int, ...]
    ) -> TensorDataclassT:
        """Returns a new TensorDataclass with the same data but with a new shape.

        This should deepcopy as well.

        Args:
            shape: The new shape of the tensor dataclass.

        Returns:
            A new TensorDataclass with the same data but with a new shape.
        """
        if isinstance(shape, int):
            shape = (shape,)
        return self._apply_exclude_static(
            lambda x: x.reshape((*shape, x.shape[-1]))
        )

    def flatten(self: TensorDataclassT) -> TensorDataclassT:
        """Returns a new TensorDataclass with flattened batch dimensions

        Returns:
            TensorDataclass: A new TensorDataclass with the same data but with a new shape.
        """
        return self.reshape((-1,))

    def broadcast_to(
            self: TensorDataclassT,
            shape: Union[torch.Size, Tuple[int, ...]],
    ) -> TensorDataclassT:
        """Returns a new TensorDataclass broadcast to new shape.

        Changes to the original tensor dataclass should effect the returned tensor dataclass,
        meaning it is NOT a deepcopy, and they are still linked.

        Args:
            shape: The new shape of the tensor dataclass.

        Returns:
            A new TensorDataclass with the same data but with a new shape.
        """
        return self._apply_exclude_static(
            lambda x: x.broadcast_to((*shape, x.shape[-1]))
        )

    def to(self: TensorDataclassT, device) -> TensorDataclassT:
        """Returns a new TensorDataclass with the same data but on the specified device.

        Args:
            device: The device to place the tensor dataclass.

        Returns:
            A new TensorDataclass with the same data but on the specified device.
        """
        return self._apply(lambda x: x.to(device))

    def cuda(self, non_blocking=False) -> TensorDataclassT:
        """Shifts the renderbuffer to the default torch cuda device"""
        fn = lambda x: x.cuda(non_blocking=non_blocking)
        return self._apply(fn)

    def cpu(self) -> TensorDataclassT:
        """Shifts the renderbuffer to the torch cpu device"""
        fn = lambda x: x.cpu()
        return self._apply(fn)

    def detach(self) -> TensorDataclassT:
        """Detaches the gradients of all channel tensors of the renderbuffer"""
        fn = lambda x: x.detach()
        return self._apply(fn)

    @staticmethod
    def direct_cat(
            tds: List[TensorDataclassT],
            dim: int = 0,
    ) -> TensorDataclassT:
        # cat_func = partial(torch.cat, dim=dim)
        def cat_func(arr):
            _arr = [ele for ele in arr if ele is not None]
            if 0 == len(_arr):
                return None
            return torch.cat(_arr, dim=dim)

        return TensorDataclass._apply_on_list(tds, cat_func)

    @staticmethod
    def direct_stack(
            tds: List[TensorDataclassT],
            dim: int = 0,
    ) -> TensorDataclassT:
        # cat_func = partial(torch.cat, dim=dim)
        def cat_func(arr):
            _arr = [ele for ele in arr if ele is not None]
            if 0 == len(_arr):
                return None
            return torch.stack(_arr, dim=dim)

        return TensorDataclass._apply_on_list(tds, cat_func)

    def cat(self, other: TensorDataclassT, dim: int = 0) -> TensorDataclassT:
        """Concatenates the channels of self and other RenderBuffers.
        If a channel only exists in one of the RBs, that channel will be returned as is.
        For channels that exists in both RBs, the spatial dimensions are assumed to be identical except for the
        concatenated dimension.

        Args:
            other (TensorDataclass) A second buffer to concatenate to the current buffer.
            dim (int): The index of spatial dimension used to concat the channels

        Returns:
            A new TensorDataclass with the concatenated channels.
        """

        def _cat(pair):
            if None not in pair:
                # Concatenating tensors of different dims where one is unsqueezed with dimensionality 1
                if (
                        pair[0].ndim == (pair[1].ndim + 1)
                        and pair[0].shape[-1] == 1
                ):
                    pair = (pair[0], pair[1].unsqueeze(-1))
                elif (
                        pair[1].ndim == (pair[0].ndim + 1)
                        and pair[1].shape[-1] == 1
                ):
                    pair = (pair[0].unsqueeze(-1), pair[1])
                return torch.cat(pair, dim=dim)
            elif (
                    pair[0] is not None and pair[1] is None
            ):  # Channel is None for other but not self
                return pair[0]
            elif (
                    pair[0] is None and pair[1] is not None
            ):  # Channel is None for self but not other
                return pair[1]
            else:
                return None

        return TensorDataclass._apply_on_pair(self, other, _cat)
