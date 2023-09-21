import datetime
from abc import abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Union
from collections import ChainMap
from loguru import logger
from termcolor import colored

import torch

# from torch.utils.tensorboard import SummaryWriter
from tensorboardX import SummaryWriter
from torchtyping import TensorType

to8b = lambda x: (255 * torch.clamp(x, min=0, max=1)).to(torch.uint8)


class Writer:
    """Writer class"""

    def __init__(self):
        self.std_logger = logger

    @abstractmethod
    def write_image(
            self,
            name: str,
            image: TensorType["H", "W", "C"],
            step: int,
    ) -> None:
        """method to write out image
        Args:
            name: data identifier
            image: rendered image to write
            step: the time step to log
        """
        raise NotImplementedError

    @abstractmethod
    def write_scalar(
            self,
            name: str,
            scalar: Union[float, torch.Tensor],
            step: int,
    ) -> None:
        """Required method to write a single scalar value to the logger
        Args:
            name: data identifier
            scalar: value to write out
            step: the time step to log
        """
        raise NotImplementedError

    def write_scalar_dict(
            self,
            name: str,
            scalar_dict: Dict[str, Any],
            step: int,
    ) -> None:
        """Function that writes out all scalars from a given dictionary to the logger
        Args:
            scalar_dict: dictionary containing all scalar values with key names and quantities
            step: the time step to log
        """
        for key, scalar in scalar_dict.items():
            try:
                float_scalar = float(scalar)
                self.write_scalar(name + "/" + key, float_scalar, step)
            except:
                pass

    def write_scalar_dicts(
            self,
            names: List[str],
            scalar_dicts: List[Dict[str, Any]],
            step: int,
    ) -> None:
        # self.std_logger.info(scalar_dicts)
        self.std_logger.info(
            ''.join(
                [
                    '{}{} '.format(
                        colored('{}:'.format(k), 'light_magenta'),
                        v
                        if k != 'ETA'
                        else str(datetime.timedelta(seconds=int(v))),
                    )
                    for k, v in dict(ChainMap(*scalar_dicts)).items()
                ]
            )
        )
        assert len(names) == len(scalar_dicts)
        for n, d in zip(names, scalar_dicts):
            self.write_scalar_dict(n, d, step)


class TensorboardWriter(Writer):
    """Tensorboard Writer Class"""

    def __init__(self, log_dir: Path):
        super(TensorboardWriter, self).__init__()
        self.tb_writer = SummaryWriter(log_dir=str(log_dir))

    def write_image(
            self,
            name: str,
            image: TensorType["H", "W", "C"],
            step: int,
    ) -> None:
        image = to8b(image)
        self.tb_writer.add_image(name, image, step, dataformats="HWC")

    def write_scalar(
            self,
            name: str,
            scalar: Union[float, torch.Tensor],
            step: int,
    ) -> None:
        self.tb_writer.add_scalar(name, scalar, step)

    def write_config(self, config: str):  # pylint: disable=unused-argument
        self.tb_writer.add_text("config", config)
