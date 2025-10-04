import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------- Residual Block -----------------
class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(x + self.block(x))

# ----------------- Dynamic Weighting Module -----------------
class DynamicWeighting(nn.Module):
    def __init__(self, channels):
        super(DynamicWeighting, self).__init__()
        # Learnable parameters for dynamic weights
        self.alpha = nn.Parameter(torch.ones(channels))
        self.beta = nn.Parameter(torch.zeros(channels))

    def forward(self, x):
        # Apply channel-wise affine transformation dynamically
        # x shape: (batch, channels, height, width)
        # Reshape alpha and beta to (1, C, 1, 1) for broadcasting
        alpha = self.alpha.view(1, -1, 1, 1)
        beta = self.beta.view(1, -1, 1, 1)
        return x * alpha + beta

# ----------------- Squeeze-and-Excitation (SE) Block with Dynamic Weighting -----------------
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.dynamic_weighting = DynamicWeighting(channels)

    def forward(self, x):
        w = self.global_pool(x)
        w = self.relu(self.fc1(w))
        w = self.sigmoid(self.fc2(w))
        scaled = x * w
        return self.dynamic_weighting(scaled)

# ----------------- Simple Transformer Encoder Block -----------------
class TransformerEncoderBlock(nn.Module):
    def __init__(self, channels, num_heads=4, dim_feedforward=2048, dropout=0.1):
        super(TransformerEncoderBlock, self).__init__()
        self.norm1 = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(embed_dim=channels, num_heads=num_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(channels)
        self.ffn = nn.Sequential(
            nn.Linear(channels, dim_feedforward),
            nn.ReLU(inplace=True),
            nn.Linear(dim_feedforward, channels),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        b, c, h, w = x.shape
        x_ = x.flatten(2).permute(2, 0, 1)  # (sequence_len, batch, channels)

        res = x_
        x_ = self.norm1(x_)
        x_, _ = self.attn(x_, x_, x_)
        x_ = x_ + res

        res = x_
        x_ = self.norm2(x_)
        x_ = self.ffn(x_) + res

        x_ = x_.permute(1, 2, 0).contiguous().view(b, c, h, w)
        return x_

# ----------------- Updated UNet with more Residual Blocks and Dynamic Weighting -----------------
class UNet(nn.Module):
    def __init__(self, use_transformer=False):
        super(UNet, self).__init__()

        # Encoder (Downsampling path) with added Residual Blocks
        self.enc1_conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            SEBlock(64),
            ResidualBlock(64),
            ResidualBlock(64),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.enc2_conv = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            SEBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.enc3_conv = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            SEBlock(256),
            ResidualBlock(256),
            ResidualBlock(256),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Bottleneck with added Residual Blocks and optional Transformer
        self.bottleneck_conv = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            ResidualBlock(512),
            ResidualBlock(512),
            ResidualBlock(512),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.use_transformer = use_transformer
        if self.use_transformer:
            self.transformer = TransformerEncoderBlock(512, num_heads=8, dim_feedforward=2048, dropout=0.1)

        # Decoder (Upsampling path) with more Residual Blocks
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3_conv = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            SEBlock(256),
            ResidualBlock(256),
            ResidualBlock(256),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2_conv = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            SEBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1_conv = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            SEBlock(64),
            ResidualBlock(64),
            ResidualBlock(64),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        # Final output layer
        self.out_conv = nn.Conv2d(64, 3, kernel_size=1)

    def forward(self, x):
        # Encoder
        enc1 = self.enc1_conv(x)
        enc1_pool = self.maxpool(enc1)

        enc2 = self.enc2_conv(enc1_pool)
        enc2_pool = self.maxpool(enc2)

        enc3 = self.enc3_conv(enc2_pool)
        enc3_pool = self.maxpool(enc3)

        # Bottleneck
        bottleneck = self.bottleneck_conv(enc3_pool)
        if self.use_transformer:
            bottleneck = self.transformer(bottleneck)

        # Decoder with skip connections
        dec3_up = self.upconv3(bottleneck)
        dec3 = torch.cat((dec3_up, enc3), dim=1)
        dec3 = self.dec3_conv(dec3)

        dec2_up = self.upconv2(dec3)
        dec2 = torch.cat((dec2_up, enc2), dim=1)
        dec2 = self.dec2_conv(dec2)

        dec1_up = self.upconv1(dec2)
        dec1 = torch.cat((dec1_up, enc1), dim=1)
        dec1 = self.dec1_conv(dec1)

        return self.out_conv(dec1)