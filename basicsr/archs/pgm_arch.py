import torch
from torch import nn as nn
import torch.nn.functional as F

from basicsr.utils.registry import ARCH_REGISTRY


class BasicBlock(nn.Module):
    def __init__(self, c_in, c_out, k=3, s=1, p=1):
        super(BasicBlock, self).__init__()
        self.block = nn.Sequential([
            nn.Conv2d(c_in, c_out, kernel_size=k, stride=s, padding=p),
            nn.InstanceNorm2d(c_out),
            nn.ReLU(inplace=True),
        ])

    def forward(self, x):
        return self.block(x)


class BetaNet(nn.Module):
    def __init__(self, n_feat=64):
        super(BetaNet, self).__init__()
        self.encoder = nn.Sequential([
            BasicBlock(3, n_feat, k=3, s=1, p=1),
            BasicBlock(n_feat, n_feat, k=3, s=2, p=1),
            BasicBlock(n_feat, n_feat, k=3, s=2, p=1),
            BasicBlock(n_feat, n_feat, k=3, s=2, p=1),
        ])

        self.depth_conv = nn.Sequential([
            nn.Conv2d(1, 1, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(8)
        ])
        self.fusion_conv = BasicBlock(n_feat+1, n_feat, k=1, s=1, p=0)

        self.beat_d_head = nn.Sequential([
            nn.AdaptiveAvgPool2d(1),
            nn.Linear(n_feat, 3),
            nn.Sigmoid(),
        ])

        self.beta_b_head = nn.Sequential([
            nn.AdaptiveAvgPool2d(1),
            nn.Linear(n_feat, 3),
            nn.Sigmoid(),
        ])

    def forward(self, x, depth):
        feat = self.encoder(x)

        depth_feat = self.depth_conv(depth)
        fusion_feat = self.fusion_conv(torch.cat([feat, depth_feat], dim=1))
        exp_negative_beta_d = self.beat_d_head(fusion_feat)

        exp_negative_beta_b = self.beta_b_head(feat)
        return exp_negative_beta_d, exp_negative_beta_b


class LightNet(nn.Module):
    def __init__(self, n_feat=8):
        super(LightNet, self).__init__()
        self.conv = nn.Sequential([
            nn.Conv2d(3, n_feat, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_feat, 3, kernel_size=1)
        ])

    def low_pass_filter(self, images, cutoff_freq=3):

        fft_images = fft.fftn(images, dim=(-2, -1))

        _, _, H, W = images.size()

        lp_filter = torch.zeros((1, 1, H, W)).to(images.device)
        lp_filter[:, :, :cutoff_freq, :cutoff_freq] = 1
        lp_filter[:, :, :cutoff_freq, -cutoff_freq:] = 1
        lp_filter[:, :, -cutoff_freq:, :cutoff_freq] = 1
        lp_filter[:, :, -cutoff_freq:, -cutoff_freq:] = 1

        filtered_fft_images = fft_images * lp_filter.to(fft_images.device)

        filtered_images = fft.ifftn(filtered_fft_images, dim=(-2, -1)).abs()

        return filtered_images

    def forward(self, x):
        local_light = self.low_pass_filter(x)

        new_local_light = self.conv(local_light) + local_light

        # new_local_light = torch.sigmoid(new_local_light)
        new_local_light = torch.clamp(new_local_light, 0, 1)

        return new_local_light


@ARCH_REGISTRY.register()
class PhysicalGuidedModel(nn.Module):
    def __init__(self):
        super(PhysicalGuidedModel, self).__init__()

    def forward(self, x):
        return x
