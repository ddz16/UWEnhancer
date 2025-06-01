import torch
from torch import nn as nn
from torch.nn import functional as F
import math

from basicsr.utils.registry import LOSS_REGISTRY


def reduction_batch_based(image_loss, M):
    # average of all valid pixels of the batch

    # avoid division by 0 (if sum(M) = sum(sum(mask)) = 0: sum(image_loss) = 0)
    divisor = torch.sum(M)

    if divisor == 0:
        return 0
    else:
        return torch.sum(image_loss) / divisor

def reduction_image_based(image_loss, M):
    # mean of average of valid pixels of an image

    # avoid division by 0 (if M = sum(mask) = 0: image_loss = 0)
    valid = M.nonzero()

    image_loss[valid] = image_loss[valid] / M[valid]

    return torch.mean(image_loss)

def backscatter_mae_loss(prediction, target, mask, reduction=reduction_batch_based):

    M = torch.sum(mask, (1, 2, 3))
    res = torch.abs(prediction - target)
    image_loss = torch.sum(mask * res , (1, 2, 3))

    return reduction(image_loss, M)

def generate_mask(x, depth, groups=10):
    # x: (B, 3, H, W), depth: (B, H, W)
    B, _, H, W = x.shape
    
    min_depth = depth.min().item()  # 0.1
    max_depth = depth.max().item()  # 6

    depth_intervals = torch.linspace(min_depth, max_depth, steps=groups+1)
    depth_intervals[0] = 0

    mask = torch.zeros_like(depth)  # (B, H, W)

    # generate group mask according to metric depth
    for i in range(groups):
        lower_bound = depth_intervals[i]
        upper_bound = depth_intervals[i + 1]
        current_mask = torch.logical_and(torch.gt(depth, lower_bound), torch.le(depth, upper_bound)).float()  # (B, H, W)

        num_pixels = torch.sum(current_mask).float() / B
        num_pixels_1percent = math.ceil(num_pixels * 0.01) if math.ceil(num_pixels * 0.01) < 500 else 500
        print('num_pixels_1percent: ', num_pixels_1percent)

        current_mask = (1 - current_mask) * 999 + 1
        brightness = torch.mean(x, dim=1) * current_mask  # brightness: (B, H, W)
        brightness = brightness.reshape(B, -1)  # brightness: (B, H*W)
        _, min_indices = torch.topk(brightness, num_pixels_1percent, dim=1, largest=False)
        group_mask = torch.zeros_like(brightness)
        group_mask.scatter_(1, min_indices, 1)
        group_mask = group_mask.reshape(B, H, W)  # group_mask: (B, H, W)

        mask = torch.logical_or(group_mask, mask)
    
    return mask.float().unsqueeze(1)  # mask: (B, 1, H, W)


@LOSS_REGISTRY.register()
class BackscatterLoss(nn.Module):
    def __init__(self, loss_weight=1.0):
        super().__init__()
        self.loss_weight = loss_weight

    def forward(self, x, depth, B_c, exp_negative_beta_b):
        # x: (B, 3, H, W), depth: (B, H, W), B_c: (B, 3, H, W), exp_negative_beta_b: (B, 3, 1, 1)
        mask = generate_mask(x, depth)
        target = B_c * (1 - torch.pow(exp_negative_beta_b, depth.unsqueeze(1)))
        loss = backscatter_mae_loss(x, target, mask, reduction=reduction_batch_based)

        return self.loss_weight * loss


# a = torch.rand(8,256,256)
# b = torch.rand(8,256,256)
# ScaleAndShiftInvariantLoss()(a,b)