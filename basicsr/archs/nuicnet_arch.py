import torch
from torch import nn as nn
import torch.nn.functional as F

from basicsr.utils.registry import ARCH_REGISTRY
from .vgg_arch import VGGFeatureExtractor


class InformationFusionBlock(nn.Module):
    def __init__(self, c_in, c_out):
        super(InformationFusionBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(c_in, c_out, kernel_size=1),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        y = self.block(x)
        return y


class FeatureExtractionBlock(nn.Module):
    def __init__(self, c_in=64, c_out=64, n=1):
        super(FeatureExtractionBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(c_in, c_out, kernel_size=3, stride=1, padding=n, dilation=n),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        y = self.block(x)
        return y


@ARCH_REGISTRY.register()
class NUICNet(nn.Module):
    def __init__(self, in_chans=1475):
        super(NUICNet, self).__init__()
        self.vgg_block = VGGFeatureExtractor(
            ["conv1_2", "conv2_2", "conv3_2", "conv4_2", "conv5_2"],
            requires_grad=False
            )
        self.blocks = nn.Sequential(
            InformationFusionBlock(in_chans, 64),
            FeatureExtractionBlock(n=1),
            FeatureExtractionBlock(n=2),
            FeatureExtractionBlock(n=4),
            FeatureExtractionBlock(n=8),
            FeatureExtractionBlock(n=16),
            FeatureExtractionBlock(n=32),
            FeatureExtractionBlock(n=64),
            FeatureExtractionBlock(n=128),
            InformationFusionBlock(64, 3),
        )

    def forward(self, input_x):
        _, _, H, W = input_x.shape
        x = input_x[:, :3, ...]
        vgg_features = self.vgg_block(x)
        vgg_features_list = list(vgg_features.values())
        vgg_features_list = [F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False) for x in vgg_features_list]
        vgg_feature_concat = torch.cat(vgg_features_list, dim=1)
        x_new = torch.cat([vgg_feature_concat, input_x], dim=1)
        y = self.blocks(x_new)
        return y
