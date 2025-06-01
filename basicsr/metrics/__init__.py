from copy import deepcopy

from basicsr.utils.registry import METRIC_REGISTRY
from .niqe import calculate_niqe
from .psnr_ssim import calculate_psnr, calculate_ssim
from .uciqe import calculate_uciqe
from .uiqm import calculate_uiqm
from .piqe import calculate_piqe

__all__ = ['calculate_psnr', 'calculate_ssim', 'calculate_niqe', 'calculate_uciqe', 'calculate_uiqm', 'calculate_piqe']


def calculate_metric(data, opt):
    """Calculate metric from data and options.

    Args:
        opt (dict): Configuration. It must contain:
            type (str): Model type.
    """
    opt = deepcopy(opt)
    metric_type = opt.pop('type')
    metric = METRIC_REGISTRY.get(metric_type)(**data, **opt)
    return metric
