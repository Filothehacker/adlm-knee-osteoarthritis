import os
import sys
import torch
import torch.nn as nn

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from ae_filippo.model import Autoencoder3D


class WOMACHead(nn.Module):
    def __init__(self, latent_channels: int):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(latent_channels, 1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z = self.pool(z).flatten(1)
        return self.fc(z)


class JSNHead(nn.Module):
    def __init__(self, latent_channels: int, num_classes: int = 4):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(latent_channels, num_classes)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z = self.pool(z).flatten(1)
        return self.fc(z)


class SurgeryHead(nn.Module):
    def __init__(self, latent_channels: int):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(latent_channels, 1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z = self.pool(z).flatten(1)
        return self.fc(z)



class TabularEncoder(nn.Module):
    """

    Input:
        tab_x : (B, T)

    Output:
        (B, tabular_latent_dim)
    """

    def __init__(
        self,
        num_tabular_inputs: int,
        tabular_latent_dim: int = 4,
        hidden_dim: int = 32,
    ):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(num_tabular_inputs, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, tabular_latent_dim),
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, tab_x: torch.Tensor) -> torch.Tensor:
        if tab_x.dim() == 1:
            tab_x = tab_x.unsqueeze(0)
        return self.encoder(tab_x)



class AutoencoderWithTabularInput(nn.Module):
    def __init__(
        self,
        base_ae: Autoencoder3D,
        latent_channels: int = 64,
        num_tabular_inputs: int = 1,
        tabular_latent_dim: int = 4,
    ):
        super().__init__()

        self.encoder = base_ae.encoder
        self.decoder = base_ae.decoder

        self.tabular_encoder = TabularEncoder(
            num_tabular_inputs=num_tabular_inputs,
            tabular_latent_dim=tabular_latent_dim,
        )

        self.womac_predictor = WOMACHead(latent_channels)
        self.jsn_predictor = JSNHead(latent_channels)
        self.surgery_predictor = SurgeryHead(latent_channels)

        self.channel_projection = nn.Conv3d(
            latent_channels + tabular_latent_dim,
            latent_channels,
            kernel_size=1,
        )

    def forward(
        self,
        x: torch.Tensor,
        tab_x: torch.Tensor,
    ):
        """
        Args:
            x     : (B, 1, D, H, W)
            tab_x : (B, T)  (missing values are encoded as 0.0 by the dataset)

        Returns:
            x_hat, womac_pred, jsn_pred, surgery_pred, z
        """
        encoded_mri = self.encoder(x)            # (B,C,D',H',W')
        encoded_tab = self.tabular_encoder(tab_x)  # (B,T_lat)

        B, C, D, H, W = encoded_mri.shape

        encoded_tab = encoded_tab.view(B, -1, 1, 1, 1)
        encoded_tab = encoded_tab.expand(-1, -1, D, H, W)

        z = torch.cat([encoded_mri, encoded_tab], dim=1)
        z = self.channel_projection(z)

        x_hat = self.decoder(z)

        womac_pred = self.womac_predictor(z)
        jsn_pred = self.jsn_predictor(z)
        surgery_pred = self.surgery_predictor(z)

        return x_hat, womac_pred, jsn_pred, surgery_pred, z


def build_tabular_input_ae(
    pretrained_ae_path: str | None,
    device: torch.device,
    in_channels: int = 1,
    latent_channels: int = 64,
    num_tabular_inputs: int = 1,
    tabular_latent_dim: int = 4,
) -> AutoencoderWithTabularInput:

    base_ae = Autoencoder3D(
        in_channels=in_channels,
        latent_channels=latent_channels,
    )

    if pretrained_ae_path is not None:
        state = torch.load(pretrained_ae_path, map_location=device)
        base_ae.load_state_dict(state)
        print(f"[build_tabular_input_ae] Loaded pretrained AE from {pretrained_ae_path}")

    base_ae.to(device)

    model = AutoencoderWithTabularInput(
        base_ae=base_ae,
        latent_channels=latent_channels,
        num_tabular_inputs=num_tabular_inputs,
        tabular_latent_dim=tabular_latent_dim,
    ).to(device)

    return model
