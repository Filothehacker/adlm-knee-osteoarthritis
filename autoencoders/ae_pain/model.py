import torch
import torch.nn as nn
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from ae_filippo.model import Autoencoder3D


class PainHead(nn.Module):
    """
    Simple regression head on top of the bottleneck z.

    We do global average pooling over (D, H, W) and then a Linear layer.
    Input z: (B, C, D', H', W')
    Output:  (B, num_outputs) - KOOS Pain score
    """

    def __init__(self, latent_channels: int, num_outputs: int = 1):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool3d(1)  # -> (B, C, 1, 1, 1)
        self.fc = nn.Linear(latent_channels, num_outputs)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: (B, C, D', H', W')
        pooled = self.pool(z)  # (B, C, 1, 1, 1)
        pooled = pooled.view(pooled.size(0), -1)  # (B, C)
        out = self.fc(pooled)  # (B, num_outputs)
        return out


class AutoencoderWithPainHead(nn.Module):
    """
    - reconstructs the MRI volume
    - predicts pain from the bottleneck
    """

    def __init__(
        self,
        base_ae: Autoencoder3D,
        latent_channels: int = 64,
        num_pain_outputs: int = 1,
    ):
        super().__init__()
        self.encoder = base_ae.encoder
        self.decoder = base_ae.decoder
        self.pain_head = PainHead(latent_channels, num_pain_outputs)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (B, 1, D, 224, 224), normalized to [-1, 1]

        Returns:
            x_hat:     reconstructed volume, same shape as x
            pain_pred: (B, num_pain_outputs)
            z:         latent tensor (B, C, D', H', W')
        """
        z = self.encoder(x)
        x_hat = self.decoder(z)
        pain_pred = self.pain_head(z)
        return x_hat, pain_pred, z


def build_pain_ae(
    pretrained_ae_path: str | None,
    device: torch.device,
    in_channels: int = 1,
    latent_channels: int = 64,
    num_pain_outputs: int = 1,
) -> AutoencoderWithPainHead:

    base_ae = Autoencoder3D(in_channels=in_channels, latent_channels=latent_channels)

    if pretrained_ae_path is not None:
        state = torch.load(pretrained_ae_path, map_location=device)
        base_ae.load_state_dict(state)
        print(f"[build_pain_ae] Loaded pretrained AE from {pretrained_ae_path}")

    base_ae.to(device)

    model = AutoencoderWithPainHead(
        base_ae=base_ae,
        latent_channels=latent_channels,
        num_pain_outputs=num_pain_outputs,
    ).to(device)

    return model
