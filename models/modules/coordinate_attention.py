"""
Coordinate Attention (CA) Module
Paper: "Coordinate Attention for Efficient Mobile Network Design" (CVPR 2021)

Improved: added residual connection with learnable scale factor for
stable gradient flow in lightweight networks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CoordinateAttention(nn.Module):
    def __init__(self, in_channels, reduction=4):
        """
        Args:
            in_channels: Number of input channels.
            reduction: Channel reduction ratio for the bottleneck.
        """
        super().__init__()
        mid_channels = max(8, in_channels // reduction)

        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))  # (B, C, H, 1)
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))  # (B, C, 1, W)

        self.conv_reduce = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(mid_channels)
        self.act = nn.GELU()

        self.conv_h = nn.Conv2d(mid_channels, in_channels, kernel_size=1, bias=False)
        self.conv_w = nn.Conv2d(mid_channels, in_channels, kernel_size=1, bias=False)

        # Learnable residual scale, initialized small so the module
        # starts close to identity and gradually contributes
        self.scale = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        """
        Returns: x + scale * attention(x)  (residual form)
        """
        B, C, H, W = x.shape

        x_h = self.pool_h(x)                          # (B, C, H, 1)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)      # (B, C, W, 1)

        y = torch.cat([x_h, x_w], dim=2)               # (B, C, H+W, 1)
        y = self.act(self.bn(self.conv_reduce(y)))      # (B, mid, H+W, 1)

        y_h, y_w = torch.split(y, [H, W], dim=2)

        attn_h = torch.sigmoid(self.conv_h(y_h))        # (B, C, H, 1)
        attn_w = torch.sigmoid(self.conv_w(y_w.permute(0, 1, 3, 2)))  # (B, C, 1, W)

        # Residual: identity + learned attention delta
        return x + self.scale * (x * attn_h * attn_w - x)
