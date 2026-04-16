"""
Boundary-Aware Supervision (BAS) Module

Multi-scale boundary heads (H/8 + H/2) for coarse-to-fine supervision,
dilated boundary GT, combined BCE+Dice boundary loss,
and boundary feature feedback into decoder path.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BoundaryDetectionHead(nn.Module):
    """Lightweight boundary detection head."""

    def __init__(self, in_channels, mid_channels=None):
        super().__init__()
        if mid_channels is None:
            mid_channels = max(8, in_channels // 2)

        self.head = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(4, mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, 1, kernel_size=1),
        )

    def forward(self, x):
        return self.head(x)


class BoundaryFeedback(nn.Module):
    """Feeds boundary prediction back into the decoder feature map
    as an extra attention-like signal."""

    def __init__(self, feat_channels):
        super().__init__()
        # 1-ch boundary map -> feat_channels spatial attention
        self.gate = nn.Sequential(
            nn.Conv2d(1, feat_channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, feat, boundary_logits):
        """
        Args:
            feat: (B, C, H, W) decoder feature
            boundary_logits: (B, 1, H, W) raw boundary logits
        Returns:
            Enhanced feature map
        """
        attn = self.gate(boundary_logits.detach())  # detach to avoid loss interference
        return feat * (1.0 + attn)


def generate_boundary_gt(mask, dilation_radius=3):
    """
    Generate boundary GT using Sobel + morphological dilation.

    Args:
        mask: (B, 1, H, W) binary mask, values in [0, 1].
        dilation_radius: Dilation kernel radius. Actual kernel = 2*r+1.
    Returns:
        boundary: (B, 1, H, W) dilated boundary map, values in {0, 1}.
    """
    device = mask.device

    # Sobel edge detection
    sobel_x = torch.tensor([[-1, 0, 1],
                             [-2, 0, 2],
                             [-1, 0, 1]], dtype=torch.float32, device=device).reshape(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1],
                             [ 0,  0,  0],
                             [ 1,  2,  1]], dtype=torch.float32, device=device).reshape(1, 1, 3, 3)

    edge_x = F.conv2d(mask, sobel_x, padding=1)
    edge_y = F.conv2d(mask, sobel_y, padding=1)
    boundary = torch.sqrt(edge_x ** 2 + edge_y ** 2 + 1e-8)

    # Binarize
    boundary = (boundary > 0.5).float()

    # Morphological dilation using max-pool
    if dilation_radius > 0:
        k = 2 * dilation_radius + 1
        boundary = F.max_pool2d(boundary, kernel_size=k, stride=1, padding=dilation_radius)

    return boundary


class BoundaryLoss(nn.Module):
    """
    Combined BCE + Dice loss for boundary supervision.
    """

    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, pred_logits, target_boundary):
        """
        Args:
            pred_logits: (B, 1, H, W) raw logits
            target_boundary: (B, 1, H, W) boundary GT in [0, 1]
        """
        if pred_logits.shape[-2:] != target_boundary.shape[-2:]:
            target_boundary = F.interpolate(
                target_boundary, size=pred_logits.shape[-2:],
                mode='nearest'
            )

        # BCE component
        bce = F.binary_cross_entropy_with_logits(pred_logits, target_boundary)

        # Dice component
        pred_prob = torch.sigmoid(pred_logits)
        smooth = 1.0
        intersection = (pred_prob * target_boundary).sum()
        dice = 1.0 - (2.0 * intersection + smooth) / (pred_prob.sum() + target_boundary.sum() + smooth)

        return self.bce_weight * bce + self.dice_weight * dice
