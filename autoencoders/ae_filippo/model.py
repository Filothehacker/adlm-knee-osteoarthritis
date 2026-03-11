import torch
import torch.nn as nn


class Encoder3D(nn.Module):
    def __init__(self, in_channels=1, latent_channels=64):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv3d(in_channels, 32, 3, stride=2, padding=1),  # -> /2
            nn.InstanceNorm3d(32, affine=True),
            nn.ReLU(True),

            nn.Conv3d(32, 64, 3, stride=2, padding=1),
            nn.InstanceNorm3d(64, affine=True),
            nn.ReLU(True),

            nn.Conv3d(64, 128, 3, stride=2, padding=1), 
            nn.InstanceNorm3d(128, affine=True),
            nn.ReLU(True),

            nn.Conv3d(128, latent_channels, 3, stride=2, padding=1),  # -> /16
            nn.InstanceNorm3d(latent_channels, affine=True),

            nn.ReLU(True),
        )

    def forward(self, x):
        return self.enc(x)


class Decoder3D(nn.Module):
    def __init__(self, out_channels=1, latent_channels=64):
        super().__init__()
        self.dec = nn.Sequential(
            nn.ConvTranspose3d(latent_channels, 128, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm3d(128, affine=True),
            nn.ReLU(True),

            nn.ConvTranspose3d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm3d(64, affine=True),
            nn.ReLU(True),

            nn.ConvTranspose3d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm3d(32, affine=True),
            nn.ReLU(True),

            nn.ConvTranspose3d(32, out_channels, 3, stride=2, padding=1, output_padding=1),
            nn.Tanh(),   # now matches input range [-1, 1]
        )

    def forward(self, z):
        return self.dec(z)


class Autoencoder3D(nn.Module):
    def __init__(self, in_channels=1, latent_channels=64):
        super().__init__()
        self.encoder = Encoder3D(in_channels, latent_channels)
        self.decoder = Decoder3D(in_channels, latent_channels)

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z
