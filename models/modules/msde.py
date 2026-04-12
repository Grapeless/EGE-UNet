"""
Multi-Scale Depthwise Enhancement (MSDE) Module

Replaces Coordinate Attention (CA) on shallow encoders (Stage 1-3).

Motivation: Shallow encoders use single 3x3 convolutions with limited receptive
fields. Skin lesions vary greatly in size, so a single receptive field misses
features at different scales, causing under-segmentation (low sensitivity).

Design:
- Dual-path depthwise convolutions: local (dilation=1) + context (dilation=3)
- Pointwise fusion + GroupNorm (consistent with model, no BatchNorm)
- Channel gating for adaptive feature selection
- Residual connection with learnable scale (stable training)
"""

import torch
import torch.nn as nn


class MSDE(nn.Module):
    """Multi-Scale Depthwise Enhancement for shallow encoders."""

    def __init__(self, channels):
        super().__init__()
        # Local features: dilation=1, receptive field 3x3
        self.dw_local = nn.Conv2d(
            channels, channels, 3, padding=1,
            groups=channels, bias=False
        )
        # Context features: dilation=3, receptive field 7x7
        self.dw_context = nn.Conv2d(
            channels, channels, 3, padding=3,
            dilation=3, groups=channels, bias=False
        )
        # Pointwise fusion
        self.pw = nn.Conv2d(channels, channels, 1, bias=False)
        self.norm = nn.GroupNorm(max(1, channels // 4), channels)
        self.act = nn.GELU()
        # Channel gating: adaptive feature selection
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels, 1, bias=True),
            nn.Sigmoid()
        )
        # Learnable residual scale, init=0 for stable start
        self.scale = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        local_feat = self.dw_local(x)
        context_feat = self.dw_context(x)
        enhanced = self.act(self.norm(self.pw(local_feat + context_feat)))
        gate = self.channel_gate(enhanced)
        return x + self.scale * enhanced * gate
