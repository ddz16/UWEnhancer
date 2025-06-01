import torch
from torch import nn as nn
import torch.nn.functional as F

from basicsr.archs.arch_util import trunc_normal_, DropPath
from basicsr.utils.registry import ARCH_REGISTRY


class PixelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(PixelAttention, self).__init__()
        self.pa = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction_ratio, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        attn = self.pa(x)
        out = attn * x
        return out

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, kernel_size=1)
            )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        attn = self.sigmoid(self.conv(self.avg_pool(x)))
        out = attn * x
        return out

class NormGate(nn.Module):
    def __init__(self, channels, kernel_size):
        super(NormGate, self).__init__()
        self._norm_branch = nn.Sequential(
            nn.InstanceNorm2d(channels),
            nn.Conv2d(channels, channels, kernel_size, padding=(kernel_size // 2), padding_mode='reflect', bias=False)
        )
        self._sig_branch = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size, padding=(kernel_size // 2), padding_mode='reflect', bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        norm = self._norm_branch(x)
        sig = self._sig_branch(x)
        return norm * sig

class ResidualBlock(nn.Module):
    def __init__(self,in_channels, out_channels, kernel_size=3):
        super(ResidualBlock, self).__init__()
        self.residual = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), padding_mode='reflect'),
            nn.GELU(),
            NormGate(out_channels, kernel_size),
            # nn.Conv2d(out_channels, out_channels, kernel_size, padding=(kernel_size // 2), padding_mode='reflect'),
            # ChannelAttention(out_channels, reduction_ratio=8)
        )

    def forward(self, x):
        input = x
        x = self.residual(x)
        return input + x

# class ResidualBlock(nn.Module):
#     def __init__(self,in_channels, out_channels, kernel_size=3):
#         super(ResidualBlock, self).__init__()
#         self.residual = nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), padding_mode='reflect'),
#             nn.GELU(),
#             nn.InstanceNorm2d(out_channels),
#             nn.Conv2d(out_channels, out_channels, kernel_size, padding=(kernel_size // 2), padding_mode='reflect', bias=False),
#             ChannelAttention(out_channels, reduction_ratio=8),
#             PixelAttention(out_channels, reduction_ratio=8),
#         )

#     def forward(self, x):
#         input = x
#         x = self.residual(x)
#         return input + x
    

class SharedLayers(nn.Module):
    def __init__(self, layer, num_layers):
        super(SharedLayers, self).__init__()
        self.layer = layer
        self.num_layers = num_layers

    def forward(self, x):
        for _ in range(self.num_layers):
            x = self.layer(x)
        return x
    
class Block(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """
    def __init__(self, dim, drop_path=0.0, layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim, padding_mode='reflect') # depthwise conv
        self.norm = nn.InstanceNorm2d(dim)
        self.pwconv1 = nn.Linear(dim, 2 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(2 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        # x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x


@ARCH_REGISTRY.register()
class UConvNeXt(nn.Module):
    def __init__(self, in_chans=3, dim=64, res_dim=128, up_learnable=False, shared=False, layer_scale_init_value=0):
        super(UConvNeXt, self).__init__()

        depths = [1, 1, 1, 3, 1, 1, 1, 4]
        # depths = [1, 1, 1, 2, 1, 1, 1, 3]
        # depths = [1, 1, 1, 3, 1, 1, 1, 4]
        
        self.input_conv = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_chans, dim, kernel_size=3),
            nn.GELU()
        )
        self.down1 = nn.Sequential(
            nn.InstanceNorm2d(dim),
            nn.Conv2d(dim, dim, kernel_size=2, stride=2),
        )
        self.block1 = nn.Sequential(
            *[Block(dim=dim, layer_scale_init_value=layer_scale_init_value) for _ in range(depths[0])]
        )

        self.down2 = nn.Sequential(
            nn.InstanceNorm2d(dim),
            nn.Conv2d(dim, dim*2, kernel_size=2, stride=2),
        )
        self.block2 = nn.Sequential(
            *[Block(dim=dim*2, layer_scale_init_value=layer_scale_init_value) for _ in range(depths[1])]
        )

        self.down3 = nn.Sequential(
            nn.InstanceNorm2d(dim*2),
            nn.Conv2d(dim*2, dim*4, kernel_size=2, stride=2),
        )
        self.block3 = nn.Sequential(
            *[Block(dim=dim*4, layer_scale_init_value=layer_scale_init_value) for _ in range(depths[2])]
        )

        self.bottleneck = nn.Sequential(
            *[Block(dim=dim*4, layer_scale_init_value=layer_scale_init_value) for _ in range(depths[3])]
        )

        if up_learnable:
            self.up1 = nn.Sequential(
                nn.InstanceNorm2d(dim*4),
                nn.ConvTranspose2d(dim*4, dim*2, kernel_size=2, stride=2),
            )
        else:
            self.up1 = nn.Sequential(
                nn.InstanceNorm2d(dim*4),
                nn.Upsample(scale_factor=2, mode='bilinear'),
                nn.Conv2d(dim*4, dim*2, kernel_size=1),
            )
        self.block4 = nn.Sequential(
            *[Block(dim=dim*2, layer_scale_init_value=layer_scale_init_value) for _ in range(depths[4])]
        )

        if up_learnable:
            self.up2 = nn.Sequential(
                nn.InstanceNorm2d(dim*2),
                nn.ConvTranspose2d(dim*2, dim, kernel_size=2, stride=2),
            )
        else:
            self.up2 = nn.Sequential(
                nn.InstanceNorm2d(dim*2),
                nn.Upsample(scale_factor=2, mode='bilinear'),
                nn.Conv2d(dim*2, dim, kernel_size=1),
            )
        self.block5 = nn.Sequential(
            *[Block(dim=dim, layer_scale_init_value=layer_scale_init_value) for _ in range(depths[5])]
        )

        if up_learnable:
            self.up3 = nn.Sequential(
                nn.InstanceNorm2d(dim),
                nn.ConvTranspose2d(dim, dim, kernel_size=2, stride=2),
            )
        else:
            self.up3 = nn.Sequential(
                nn.InstanceNorm2d(dim),
                nn.Upsample(scale_factor=2, mode='bilinear'),
                nn.Conv2d(dim, dim, kernel_size=1),
            )
        self.block6 = nn.Sequential(
            *[Block(dim=dim, layer_scale_init_value=layer_scale_init_value) for _ in range(depths[6])]
        )

        self.out_conv = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, 3, kernel_size=3),
            # nn.Sigmoid()
        )
        
        self.shared = shared
        self.input_conv2 = nn.Conv2d(in_channels=in_chans, out_channels=res_dim, kernel_size=3, padding=1, bias=True, padding_mode='reflect')
        if shared:
            self.residual_blocks = SharedLayers(
                layer=ResidualBlock(in_channels=res_dim, out_channels=res_dim, kernel_size=3),
                num_layers=depths[7]
            )
        else:
            self.residual_blocks = nn.Sequential(
                *[ResidualBlock(in_channels=res_dim, out_channels=res_dim, kernel_size=3) for _ in range(depths[7])]
            )
            # self.residual_blocks = nn.Sequential(
            #     *[Block(dim=res_dim, layer_scale_init_value=layer_scale_init_value) for _ in range(depths[7])]
            # )
        self.out_conv2 = nn.Sequential(
            nn.Conv2d(in_channels=res_dim, out_channels=3, kernel_size=3, padding=1, padding_mode='reflect'),
            # nn.Sigmoid()
        )
        
        # self.apply(self._init_weights)

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         torch.nn.init.normal_(m.weight.data, 0.0, 0.02)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # global branch
        x_global = self.input_conv(x)        # [B,3,H,W] -> [B,C,H,W]
        x_down1 = self.down1(x_global)       # [B,C,H,W] -> [B,C,H/2,W/2]
        x_down1 = self.block1(x_down1)       # [B,C,H/2,W/2] -> [B,C,H/2,W/2]
        x_down2 = self.down2(x_down1)        # [B,C,H/2,W/2] -> [B,2C,H/4,W/4]
        x_down2 = self.block2(x_down2)       # [B,2C,H/4,W/4] -> [B,2C,H/4,W/4]
        x_down3 = self.down3(x_down2)        # [B,2C,H/4,W/4] -> [B,4C,H/8,W/8]
        x_down3 = self.block3(x_down3)       # [B,4C,H/8,W/8] -> [B,4C,H/8,W/8]
        x_feature = self.bottleneck(x_down3) # [B,4C,H/8,W/8] -> [B,4C,H/8,W/8]
        x_up1 = self.up1(x_feature)  # [B,4C,H/8,W/8] -> [B,2C,H/4,W/4]
        x_up1 = self.block4(x_up1)           # [B,2C,H/4,W/4] -> [B,2C,H/4,W/4]
        x_up2 = self.up2(x_up1 + x_down2)    # [B,2C,H/4,W/4] -> [B,C,H/2,W/2]
        x_up2 = self.block5(x_up2)           # [B,C,H/2,W/2] -> [B,C,H/2,W/2]
        x_up3 = self.up3(x_up2 + x_down1)    # [B,C,H/2,W/2] -> [B,C,H,W]
        x_up3 = self.block6(x_up3)           # [B,C,H,W] -> [B,C,H,W]
        out_global = x_up3 + x_global        # [B,C,H,W] -> [B,C,H,W]
        result1 = self.out_conv(out_global)  # [B,C,H,W] -> [B,3,H,W]
        
        # local branch
        x_local = self.input_conv2(x)
        out_local = x_local + self.residual_blocks(x_local)
        result2 = self.out_conv2(out_local)

        return torch.sigmoid(result1 + result2)