import torch
from torch import nn as nn
import torch.nn.functional as F
import torch.fft as fft
from torchvision.transforms.functional import normalize, resize

from basicsr.utils.registry import ARCH_REGISTRY
from basicsr.archs.arch_util import trunc_normal_, DropPath
from basicsr.archs.uiec2net_arch import UIEC2Net


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

class ConvNeXtEncoder(nn.Module):
    def __init__(self, in_chans=3, dim=64, layer_scale_init_value=0):
        super(ConvNeXtEncoder, self).__init__()

        depths = [1, 1, 1, 3]
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

    def forward(self, x):
        x_global = self.input_conv(x)        # [B,3,H,W] -> [B,C,H,W]
        x_down1 = self.down1(x_global)       # [B,C,H,W] -> [B,C,H/2,W/2]
        x_down1 = self.block1(x_down1)       # [B,C,H/2,W/2] -> [B,C,H/2,W/2]
        x_down2 = self.down2(x_down1)        # [B,C,H/2,W/2] -> [B,2C,H/4,W/4]
        x_down2 = self.block2(x_down2)       # [B,2C,H/4,W/4] -> [B,2C,H/4,W/4]
        x_down3 = self.down3(x_down2)        # [B,2C,H/4,W/4] -> [B,4C,H/8,W/8]
        x_down3 = self.block3(x_down3)       # [B,4C,H/8,W/8] -> [B,4C,H/8,W/8]
        x_feature = self.bottleneck(x_down3) # [B,4C,H/8,W/8] -> [B,4C,H/8,W/8]

        return x_feature
    

def _make_scratch(in_shape, out_shape, groups=1, expand=False):
    scratch = nn.Module()

    out_shape1 = out_shape
    out_shape2 = out_shape
    out_shape3 = out_shape
    if len(in_shape) >= 4:
        out_shape4 = out_shape

    if expand:
        out_shape1 = out_shape
        out_shape2 = out_shape*2
        out_shape3 = out_shape*4
        if len(in_shape) >= 4:
            out_shape4 = out_shape*8

    scratch.layer1_rn = nn.Conv2d(
        in_shape[0], out_shape1, kernel_size=3, stride=1, padding=1, bias=False, groups=groups
    )
    scratch.layer2_rn = nn.Conv2d(
        in_shape[1], out_shape2, kernel_size=3, stride=1, padding=1, bias=False, groups=groups
    )
    scratch.layer3_rn = nn.Conv2d(
        in_shape[2], out_shape3, kernel_size=3, stride=1, padding=1, bias=False, groups=groups
    )
    if len(in_shape) >= 4:
        scratch.layer4_rn = nn.Conv2d(
            in_shape[3], out_shape4, kernel_size=3, stride=1, padding=1, bias=False, groups=groups
        )

    return scratch

class ResidualConvUnit(nn.Module):
    """Residual convolution module.
    """

    def __init__(self, features, activation, bn):
        """Init.

        Args:
            features (int): number of features
        """
        super().__init__()

        self.bn = bn

        self.groups=1

        self.conv1 = nn.Conv2d(
            features, features, kernel_size=3, stride=1, padding=1, bias=True, groups=self.groups
        )
        
        self.conv2 = nn.Conv2d(
            features, features, kernel_size=3, stride=1, padding=1, bias=True, groups=self.groups
        )

        if self.bn==True:
            self.bn1 = nn.BatchNorm2d(features)
            self.bn2 = nn.BatchNorm2d(features)

        self.activation = activation

        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x):
        """Forward pass.

        Args:
            x (tensor): input

        Returns:
            tensor: output
        """
        
        out = self.activation(x)
        out = self.conv1(out)
        if self.bn==True:
            out = self.bn1(out)
       
        out = self.activation(out)
        out = self.conv2(out)
        if self.bn==True:
            out = self.bn2(out)

        if self.groups > 1:
            out = self.conv_merge(out)

        return self.skip_add.add(out, x)
    
class FeatureFusionBlock(nn.Module):
    """Feature fusion block.
    """

    def __init__(self, features, activation, deconv=False, bn=False, expand=False, align_corners=True, size=None):
        """Init.

        Args:
            features (int): number of features
        """
        super(FeatureFusionBlock, self).__init__()

        self.deconv = deconv
        self.align_corners = align_corners

        self.groups=1

        self.expand = expand
        out_features = features
        if self.expand==True:
            out_features = features//2
        
        self.out_conv = nn.Conv2d(features, out_features, kernel_size=1, stride=1, padding=0, bias=True, groups=1)

        self.resConfUnit1 = ResidualConvUnit(features, activation, bn)
        self.resConfUnit2 = ResidualConvUnit(features, activation, bn)
        
        self.skip_add = nn.quantized.FloatFunctional()

        self.size=size

    def forward(self, *xs, size=None):
        """Forward pass.

        Returns:
            tensor: output
        """
        output = xs[0]

        if len(xs) == 2:
            res = self.resConfUnit1(xs[1])
            output = self.skip_add.add(output, res)

        output = self.resConfUnit2(output)

        if (size is None) and (self.size is None):
            modifier = {"scale_factor": 2}
        elif size is None:
            modifier = {"size": self.size}
        else:
            modifier = {"size": size}

        output = nn.functional.interpolate(
            output, **modifier, mode="bilinear", align_corners=self.align_corners
        )

        output = self.out_conv(output)

        return output

def _make_fusion_block(features, use_bn, size = None):
    return FeatureFusionBlock(
        features,
        nn.ReLU(False),
        deconv=False,
        bn=use_bn,
        expand=False,
        align_corners=True,
        size=size,
    )

class DPTHead(nn.Module):
    def __init__(self, in_channels, features=256, use_bn=False, out_channels=[256, 512, 1024, 1024], use_clstoken=False):
        super(DPTHead, self).__init__()
        
        self.use_clstoken = use_clstoken
        
        # out_channels = [in_channels // 8, in_channels // 4, in_channels // 2, in_channels]
        # out_channels = [in_channels // 4, in_channels // 2, in_channels, in_channels]
        # out_channels = [in_channels, in_channels, in_channels, in_channels]
        
        self.projects = nn.ModuleList([
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channel,
                kernel_size=1,
                stride=1,
                padding=0,
            ) for out_channel in out_channels
        ])
        
        self.resize_layers = nn.ModuleList([
            nn.ConvTranspose2d(
                in_channels=out_channels[0],
                out_channels=out_channels[0],
                kernel_size=4,
                stride=4,
                padding=0),
            nn.ConvTranspose2d(
                in_channels=out_channels[1],
                out_channels=out_channels[1],
                kernel_size=2,
                stride=2,
                padding=0),
            nn.Identity(),
            nn.Conv2d(
                in_channels=out_channels[3],
                out_channels=out_channels[3],
                kernel_size=3,
                stride=2,
                padding=1)
        ])
        
        if use_clstoken:
            self.readout_projects = nn.ModuleList()
            for _ in range(len(self.projects)):
                self.readout_projects.append(
                    nn.Sequential(
                        nn.Linear(2 * in_channels, in_channels),
                        nn.GELU()))
        
        self.scratch = _make_scratch(
            out_channels,
            features,
            groups=1,
            expand=False,
        )

        self.scratch.stem_transpose = None
        
        self.scratch.refinenet1 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet2 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet3 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet4 = _make_fusion_block(features, use_bn)

        head_features_1 = features
        head_features_2 = 32
        
        self.scratch.output_conv1 = nn.Conv2d(head_features_1, head_features_1 // 2, kernel_size=3, stride=1, padding=1)
        
        self.scratch.output_conv2 = nn.Sequential(
            nn.Conv2d(head_features_1 // 2, head_features_2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(head_features_2, 1, kernel_size=1, stride=1, padding=0),
            nn.ReLU(True),
            nn.Identity(),
        )
    
    def forward(self, out_features, patch_h, patch_w):
        out = []
        for i, x in enumerate(out_features):
            if self.use_clstoken:
                x, cls_token = x[0], x[1]
                readout = cls_token.unsqueeze(1).expand_as(x)
                x = self.readout_projects[i](torch.cat((x, readout), -1))
            else:
                x = x[0]
            
            x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w))
            
            x = self.projects[i](x)
            x = self.resize_layers[i](x)
            
            out.append(x)
        
        layer_1, layer_2, layer_3, layer_4 = out
        
        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)
        
        path_4 = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn, size=layer_2_rn.shape[2:])
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn, size=layer_1_rn.shape[2:])
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)
        
        out = self.scratch.output_conv1(path_1)
        out = F.interpolate(out, (int(patch_h * 14), int(patch_w * 14)), mode="bilinear", align_corners=True)
        out = self.scratch.output_conv2(out)
            
        return out

class DPT_DINOv2(nn.Module):
    def __init__(self, encoder='vitl', features=256, use_bn=False, out_channels=[256, 512, 1024, 1024], use_clstoken=False):

        super(DPT_DINOv2, self).__init__()

        torch.manual_seed(1)
        
        self.pretrained = torch.hub.load('torchhub/facebookresearch_dinov2_main', 'dinov2_{:}14'.format(encoder), source='local', pretrained=False)
        
        dim = self.pretrained.blocks[0].attn.qkv.in_features
        
        self.depth_head = DPTHead(dim, features, use_bn, out_channels=out_channels, use_clstoken=use_clstoken)

    def forward(self, x):
        h, w = x.shape[-2:]

        features = self.pretrained.get_intermediate_layers(x, 4, return_class_token=True)
        
        patch_h, patch_w = h // 14, w // 14

        depth = self.depth_head(features, patch_h, patch_w)
        depth = F.interpolate(depth, size=(h, w), mode="bilinear", align_corners=True)
        depth = F.relu(depth)

        return depth.squeeze(1)


class BasicBlock(nn.Module):
    def __init__(self, c_in, c_out, k=3, s=1, p=1):
        super(BasicBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(c_in, c_out, kernel_size=k, stride=s, padding=p, padding_mode='reflect'),
            nn.InstanceNorm2d(c_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)

class BetaNet(nn.Module):
    def __init__(self, n_feat=64):
        super(BetaNet, self).__init__()
        # self.encoder = ConvNeXtEncoder(in_chans=7)
        self.encoder = nn.Sequential(
            BasicBlock(7, n_feat, k=3, s=1, p=1),
            BasicBlock(n_feat, n_feat, k=3, s=2, p=1),
            BasicBlock(n_feat, n_feat, k=3, s=2, p=1),
            BasicBlock(n_feat, n_feat, k=3, s=2, p=1),
        )

        self.beat_d_head = nn.Sequential(
            # nn.Upsample(scale_factor=8, mode='bilinear'),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(n_feat, n_feat, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_feat, 1, kernel_size=1),
            nn.Sigmoid(),
        )

        self.beta_b_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(n_feat, n_feat, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_feat, 1, kernel_size=1),
            nn.Sigmoid(),
        )

        self.scale_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(n_feat, n_feat, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(n_feat, 1, kernel_size=1),
            nn.Sigmoid(),
        )
        self.shift_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(n_feat, n_feat, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(n_feat, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x, depth, B_c):
        feat = self.encoder(torch.cat([x, depth, B_c], dim=1))
        exp_negative_beta_d = self.beat_d_head(feat)

        exp_negative_beta_b = self.beta_b_head(feat)
        depth_gap = self.scale_head(feat) * 2
        depth_min = self.shift_head(feat) + 0.1
        return exp_negative_beta_d, exp_negative_beta_b, depth_min, depth_gap
    
    # def forward(self, x, depth):
    #     feat = self.encoder(x)

    #     depth_feat = self.depth_conv(depth)
    #     fusion_feat = self.fusion_conv(torch.cat([feat, depth_feat], dim=1))
    #     exp_negative_beta_d = self.beat_d_head(fusion_feat)

    #     exp_negative_beta_b = self.beta_b_head(feat)
    #     depth_max = self.depth_max_head(fusion_feat) * 7 + 8
    #     depth_min = self.depth_min_head(fusion_feat)
    #     return exp_negative_beta_d, exp_negative_beta_b, depth_min, depth_max

class BetaNet1(nn.Module):
    def __init__(self, n_feat=64):
        super(BetaNet1, self).__init__()
        # self.encoder = nn.Sequential(
        #     BasicBlock(3, n_feat, k=3, s=1, p=1),
        #     BasicBlock(n_feat, n_feat, k=3, s=2, p=1),
        #     BasicBlock(n_feat, n_feat, k=3, s=2, p=1),
        #     BasicBlock(n_feat, n_feat, k=3, s=2, p=1),
        # )
        self.encoder = ConvNeXtEncoder()

        self.beat_d_head_a = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(n_feat, 3, kernel_size=1),
            nn.ReLU(inplace=True)
        )
        self.beat_d_head_b = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(n_feat, 3, kernel_size=1),
            nn.ReLU(inplace=True)
        )
        self.beat_d_head_c = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(n_feat, 3, kernel_size=1),
            nn.ReLU(inplace=True)
        )
        self.beat_d_head_d = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(n_feat, 3, kernel_size=1),
            nn.ReLU(inplace=True)
        )

        self.beta_b_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(n_feat, 3, kernel_size=1),
            nn.Sigmoid(),
        )

        self.depth_max_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(n_feat, 1, kernel_size=1),
            nn.Sigmoid(),
        )
        self.depth_min_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(n_feat, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def postprocess_depth(self, depth, depth_min, depth_max): 
        metric_depth = 1.0 / (depth + 1e-6)
        depth_min_per_sample, _ = torch.min(metric_depth.view(metric_depth.shape[0], -1), dim=1, keepdim=True)
        depth_max_per_sample, _ = torch.max(metric_depth.view(metric_depth.shape[0], -1), dim=1, keepdim=True)
        new_metric_depth = (metric_depth - depth_min_per_sample[:, :, None, None]) / (depth_max_per_sample[:, :, None, None] - depth_min_per_sample[:, :, None, None]) * (depth_max - depth_min) + depth_min
        return new_metric_depth
    
    def forward(self, x, depth):
        # depth: [B,1,H,W]
        feat = self.encoder(x)

        a = self.beat_d_head_a(feat)  # [B,3,1,1]
        b = self.beat_d_head_b(feat)  # [B,3,1,1]
        c = self.beat_d_head_c(feat)  # [B,3,1,1]
        d = self.beat_d_head_d(feat)  # [B,3,1,1]

        depth_max = self.depth_max_head(feat) * 7 + 8
        depth_min = self.depth_min_head(feat)
        metric_depth = self.postprocess_depth(depth, depth_min, depth_max)
        
        beta_d = a * torch.exp(-b * metric_depth) + c * torch.exp(-d * metric_depth)
        exp_negative_beta_d = torch.exp(-beta_d)  # [B,3,H,W]
        
        exp_negative_beta_b = self.beta_b_head(feat)
        return exp_negative_beta_d, exp_negative_beta_b, depth_min, depth_max

class LightNet(nn.Module):
    def __init__(self, n_feat=64):
        super(LightNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, n_feat, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_feat, 3, kernel_size=3, padding=1)
        )

    def low_pass_filter(self, images, cutoff_freq=1):

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

        local_light = self.conv(local_light) + local_light

        # new_local_light = torch.sigmoid(new_local_light)
        new_local_light = torch.clamp(local_light, 0, 1)

        return new_local_light  


@ARCH_REGISTRY.register()
class UWUIEC2Net(nn.Module):
    def __init__(self):
        super(UWUIEC2Net, self).__init__()
        self.light_net = LightNet(n_feat=64)
        self.beta_net = BetaNet(n_feat=256)
        self.depth_net = DPT_DINOv2(encoder='vits', features=64, out_channels=[48, 96, 192, 384], use_clstoken=False)
        state_dict = torch.load('experiments/pretrained_models/depth_anything_vits14.pth', map_location='cpu')
        self.depth_net.load_state_dict(state_dict)
        for param in self.depth_net.pretrained.parameters():
            param.requires_grad = False

        self.uconvnext = UIEC2Net(in_chans=7)

    def prepare_for_depthnet(self, x): 
        y = resize(x, (266, 266))
        y = normalize(y, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        return y
    
    def postprocess_depth(self, depth, depth_min, depth_gap): 
        metric_depth = 1.0 / (depth + 0.08)
        depth_min_per_sample, _ = torch.min(metric_depth.view(metric_depth.shape[0], -1), dim=1, keepdim=True)
        depth_max_per_sample, _ = torch.max(metric_depth.view(metric_depth.shape[0], -1), dim=1, keepdim=True)
        new_metric_depth = (metric_depth - depth_min_per_sample[:, :, None, None]) / (depth_max_per_sample[:, :, None, None] - depth_min_per_sample[:, :, None, None]) * depth_gap + depth_min
        return new_metric_depth
    
    def normalize_disparity(self, depth): 
        depth_min_per_sample, _ = torch.min(depth.view(depth.shape[0], -1), dim=1, keepdim=True)
        depth_max_per_sample, _ = torch.max(depth.view(depth.shape[0], -1), dim=1, keepdim=True)
        new_depth = (depth - depth_min_per_sample[:, :, None, None]) / (depth_max_per_sample[:, :, None, None] - depth_min_per_sample[:, :, None, None])
        return new_depth

    # def forward(self, x):
    #     x_for_depthnet = self.prepare_for_depthnet(x)
    #     B_c = self.light_net(x)               # [B,3,H,W]
    #     z = self.depth_net(x_for_depthnet)    # [B,H,W]
    #     z = resize(z.unsqueeze(1), (x.shape[-1], x.shape[-1]))
    #     z_norm = self.normalize_disparity(z)  # [B,1,H,W] 0-1
    #     exp_negative_beta_d, exp_negative_beta_b, depth_min, depth_gap = self.beta_net(x, z_norm, B_c)  # [B,3,H,W], [B,3,1,1]
    #     # metric_z = self.postprocess_depth(z_norm, depth_min, depth_gap)
    #     metric_z = 1 / (z_norm * depth_gap + depth_min)  # depth <= 10

    #     # print("B_c", B_c[0,:,100:110,100:110])
    #     # print("z", z[0,:,100:110,100:110])
    #     # print("exp_negative_beta_d", exp_negative_beta_d[0,:,100:110,100:110])
    #     # print("exp_negative_beta_b", exp_negative_beta_b[0, ...])
    #     # print("metric_z", metric_z[0, :, 100:110,100:110])
        
    #     J_c = (x - B_c * (1 - torch.pow(exp_negative_beta_b, metric_z))) * 1 / torch.pow(exp_negative_beta_d, metric_z)
    #     # J_c = torch.clamp(J_c, 0, 1)
    #     J_c_for_depthnet = self.prepare_for_depthnet(J_c)
    #     z_J_c = self.depth_net(J_c_for_depthnet)
    #     z_J_c = resize(z_J_c.unsqueeze(1), (x.shape[-1], x.shape[-1]))
    #     z_J_c_norm = self.normalize_disparity(z_J_c)  # [B,1,H,W] 0-1

    #     # result = self.uconvnext(torch.cat([J_c, z_J_c_norm, B_c], dim=1))
    #     result = self.uconvnext(J_c)

    #     return J_c, result, B_c, metric_z.squeeze(1), z_norm.squeeze(1), z_J_c_norm.squeeze(1), exp_negative_beta_b


    def forward(self, x):
        x_for_depthnet = self.prepare_for_depthnet(x)
        B_c = self.light_net(x)               # [B,3,H,W]
        z = self.depth_net(x_for_depthnet)    # [B,H,W]
        z = resize(z.unsqueeze(1), (x.shape[-1], x.shape[-1]))
        z_norm = self.normalize_disparity(z)  # [B,1,H,W] 0-1
        exp_negative_beta_d, exp_negative_beta_b, depth_min, depth_gap = self.beta_net(x, z_norm, B_c)  # [B,3,H,W], [B,3,1,1]
        # metric_z = self.postprocess_depth(z_norm, depth_min, depth_gap)
        metric_z = 1 / (z_norm * depth_gap + depth_min)  # depth <= 10

        # print("B_c", B_c[0,:,100:110,100:110])
        # print("z", z[0,:,100:110,100:110])
        # print("exp_negative_beta_d", exp_negative_beta_d[0,:,100:110,100:110])
        # print("exp_negative_beta_b", exp_negative_beta_b[0, ...])
        # print("metric_z", metric_z[0, :, 100:110,100:110])
        
        # J_c = (x - B_c * (1 - torch.pow(exp_negative_beta_b, metric_z))) * 1 / torch.pow(exp_negative_beta_d, metric_z)
        # J_c = torch.clamp(J_c, 0, 1)
        

        J_c = self.uconvnext(torch.cat([x, z_norm, B_c], dim=1))
        # J_c = self.uconvnext(x)

        lq = J_c * torch.pow(exp_negative_beta_d, metric_z) + B_c * (1 - torch.pow(exp_negative_beta_b, metric_z))

        # J_c_for_depthnet = self.prepare_for_depthnet(J_c)
        # z_J_c = self.depth_net(J_c_for_depthnet)
        # z_J_c = resize(z_J_c.unsqueeze(1), (x.shape[-1], x.shape[-1]))
        # z_J_c_norm = self.normalize_disparity(z_J_c)  # [B,1,H,W] 0-1

        return lq, J_c, B_c, metric_z.squeeze(1), z_norm.squeeze(1), z_norm.squeeze(1), exp_negative_beta_b