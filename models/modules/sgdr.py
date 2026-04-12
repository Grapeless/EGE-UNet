"""
Semantic-Guided Decoder Refinement (SGDR) Module

Replaces Boundary-Aware Supervision (BAS) at the decoder output.

Motivation: The original BAS operates at H/8 scale (32x32) where boundaries
are sub-pixel and unlearnable, and uses detach() which blocks gradient flow.
SGDR operates at H/2 scale (128x128) where boundaries are clearly visible,
and does NOT detach gradients so the segmentation loss can guide the boundary head.

Design:
- Feature Refinement: residual depthwise block enhances decoder features
- Boundary-Guided Attention: lightweight boundary head produces attention
  that enhances features near lesion boundaries (no detach!)
- Single-scale at H/2 only (no broken multi-scale)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureRefinement(nn.Module):
    """Residual feature refinement to enhance decoder output discriminability."""

    def __init__(self, channels):
        super().__init__()
        self.refine = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.GroupNorm(max(1, channels // 4), channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 1, bias=False),
        )

    def forward(self, x):
        return x + self.refine(x)


class BoundaryGuidedAttention(nn.Module):
    """Lightweight boundary head + boundary-to-feature attention (no detach!)."""

    def __init__(self, channels):
        super().__init__()
        # Boundary detection head (single H/2 scale)
        self.boundary_head = nn.Conv2d(channels, 1, 3, padding=1)
        # Boundary -> feature attention (NOT detached, gradients flow through)
        self.boundary_attn = nn.Sequential(
            nn.Conv2d(1, channels, 1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        bd_logits = self.boundary_head(x)             # (B, 1, H/2, W/2)
        attn = self.boundary_attn(bd_logits)           # (B, C, H/2, W/2)
        enhanced = x * (1.0 + 0.5 * attn)             # gentle enhancement
        return enhanced, bd_logits


class SGDR(nn.Module):
    """Semantic-Guided Decoder Refinement: feature refinement + boundary guidance."""

    def __init__(self, channels):
        super().__init__()
        self.feature_refine = FeatureRefinement(channels)
        self.boundary_guide = BoundaryGuidedAttention(channels)

    def forward(self, x):
        x = self.feature_refine(x)
        x, bd_logits = self.boundary_guide(x)
        return x, bd_logits


def generate_boundary_gt(mask, dilation_radius=2):
    """Generate boundary GT using Sobel + morphological dilation.

    Args:
        mask: (B, 1, H, W) binary mask, values in [0, 1].
        dilation_radius: Dilation kernel radius. Actual kernel = 2*r+1.
    Returns:
        boundary: (B, 1, H, W) dilated boundary map, values in {0, 1}.
    """
    device = mask.device

    sobel_x = torch.tensor([[-1, 0, 1],
                             [-2, 0, 2],
                             [-1, 0, 1]], dtype=torch.float32, device=device).reshape(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1],
                             [ 0,  0,  0],
                             [ 1,  2,  1]], dtype=torch.float32, device=device).reshape(1, 1, 3, 3)

    edge_x = F.conv2d(mask, sobel_x, padding=1)
    edge_y = F.conv2d(mask, sobel_y, padding=1)
    boundary = torch.sqrt(edge_x ** 2 + edge_y ** 2 + 1e-8)
    boundary = (boundary > 0.5).float()

    if dilation_radius > 0:
        k = 2 * dilation_radius + 1
        boundary = F.max_pool2d(boundary, kernel_size=k, stride=1, padding=dilation_radius)

    return boundary


class BoundaryLoss(nn.Module):
    """Combined BCE + Dice loss for boundary supervision."""

    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, pred_logits, target_boundary):
        if pred_logits.shape[-2:] != target_boundary.shape[-2:]:
            target_boundary = F.interpolate(
                target_boundary, size=pred_logits.shape[-2:], mode='nearest'
            )

        bce = F.binary_cross_entropy_with_logits(pred_logits, target_boundary)

        pred_prob = torch.sigmoid(pred_logits)
        smooth = 1.0
        intersection = (pred_prob * target_boundary).sum()
        dice = 1.0 - (2.0 * intersection + smooth) / (pred_prob.sum() + target_boundary.sum() + smooth)

        return self.bce_weight * bce + self.dice_weight * dice
