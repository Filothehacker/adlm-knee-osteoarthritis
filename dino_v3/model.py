import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Make the dinov3 package importable (it lives at <repo>/dinov3/)
DINOV3_PKG = os.path.join(PROJECT_ROOT, "dinov3")
if DINOV3_PKG not in sys.path:
    sys.path.insert(0, DINOV3_PKG)

import torch
import torch.nn as nn


class DINOv3MRI(nn.Module):
    """
    Wraps a 2D DINOv3 ViT backbone to produce per-patient feature vectors
    from 3D MRI volumes of shape [1, D, H, W].

    Strategy:
      1. Repeat the single grayscale channel 3× -> [D, 3, H, W]
      2. Pass every slice through the ViT in one batched forward pass.
      3. Extract the [CLS] token for each slice -> [D, embed_dim]
      4. Mean-pool across slices -> [embed_dim] patient feature vector.

    This is equivalent to treating depth slices as an "ensemble" of 2D views,
    which is standard practice when applying 2D ViTs to 3D medical volumes.
    """

    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone

    def forward(self, vol: torch.Tensor) -> torch.Tensor:
        """
        Args:
            vol: [B, 1, D, H, W]  (B is typically 1 during inference)
        Returns:
            features: [B, embed_dim]
        """
        B, C, D, H, W = vol.shape
        assert C == 1, "Expected single-channel MRI volume"

        # [B, 1, D, H, W] -> [B*D, 1, H, W] -> [B*D, 3, H, W]
        slices = vol.permute(0, 2, 1, 3, 4).reshape(B * D, 1, H, W)
        slices = slices.expand(-1, 3, -1, -1)  # repeat channel

        # Get CLS token from backbone: [B*D, embed_dim]
        cls_tokens = self.backbone(slices)

        # [B*D, embed_dim] -> [B, D, embed_dim] -> [B, embed_dim]
        features = cls_tokens.reshape(B, D, -1).mean(dim=1)
        return features


def build_dino_model(weights_path: str, device: torch.device) -> DINOv3MRI:
    """
    Build and return a DINOv3MRI model loaded from local weights.

    The pretrained ViT-B/16 weights file is expected at weights_path
    (default: weights_dinov3/dinov3_vitb16_pretrain_lvd1689m.pth).
    """
    from dinov3.hub.backbones import dinov3_vitb16

    print(f"Loading DINOv3 ViT-B/16 weights from {weights_path} ...")
    backbone = dinov3_vitb16(pretrained=True, weights=weights_path)
    backbone = backbone.to(device)
    backbone.eval()

    model = DINOv3MRI(backbone).to(device)
    model.eval()
    return model
