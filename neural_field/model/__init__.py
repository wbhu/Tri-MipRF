from .RFModel import RFModel
from .trimipRF import TriMipRFModel


def get_model(model_name: str = 'Tri-MipRF') -> RFModel:
    if 'Tri-MipRF' == model_name:
        return TriMipRFModel
    else:
        raise NotImplementedError
