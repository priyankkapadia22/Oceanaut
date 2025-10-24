import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualDenseBlock(nn.Module):
    def __init__(self, in_channels, growth_channels=32, num_layers=5):
        super(ResidualDenseBlock, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(nn.Sequential(
                nn.Conv2d(in_channels + i * growth_channels, growth_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            ))
        self.lff = nn.Conv2d(in_channels + num_layers * growth_channels, in_channels, kernel_size=1)

    def forward(self, x):
        features = [x]
        for layer in self.layers:
            out = layer(torch.cat(features, dim=1))
            features.append(out)
        concat = torch.cat(features, dim=1)
        fused = self.lff(concat)
        return fused + x  # residual connection

class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        attention = self.sigmoid(self.conv(x_cat))
        return x * attention

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, kernel_size=1),
            nn.Sigmoid()
        )
    def forward(self, x):
        w = self.global_pool(x)
        w = self.fc(w)
        return x * w

class RDBWithAttention(nn.Module):
    def __init__(self, in_channels):
        super(RDBWithAttention, self).__init__()
        self.rdb = ResidualDenseBlock(in_channels)
        self.se = SEBlock(in_channels)
        self.sa = SpatialAttention()
    def forward(self, x):
        out = self.rdb(x)
        out = self.se(out)
        out = self.sa(out)
        return out

class EnhancedUNet(nn.Module):
    def __init__(self, use_transformer=False, base_channels=64):
        super(EnhancedUNet, self).__init__()
        # Encoder blocks
        self.enc1 = nn.Sequential(
            nn.Conv2d(3, base_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            RDBWithAttention(base_channels),
            RDBWithAttention(base_channels),
        )
        self.pool1 = nn.MaxPool2d(2,2)
        self.enc2 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels*2, 3, padding=1),
            nn.ReLU(inplace=True),
            RDBWithAttention(base_channels*2),
            RDBWithAttention(base_channels*2),
        )
        self.pool2 = nn.MaxPool2d(2,2)
        self.enc3 = nn.Sequential(
            nn.Conv2d(base_channels*2, base_channels*4, 3, padding=1),
            nn.ReLU(inplace=True),
            RDBWithAttention(base_channels*4),
            RDBWithAttention(base_channels*4),
        )
        self.pool3 = nn.MaxPool2d(2,2)

        # Bottleneck
        self.bottleneck_conv = nn.Sequential(
            nn.Conv2d(base_channels*4, base_channels*8, 3, padding=1),
            nn.ReLU(inplace=True),
            RDBWithAttention(base_channels*8),
            RDBWithAttention(base_channels*8)
        )
        self.use_transformer = use_transformer
        if self.use_transformer:
            self.transformer = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=base_channels*8,
                    nhead=8,
                    batch_first=True,
                    dim_feedforward=2048,
                    dropout=0.1
                ),
                num_layers=4
            )

        # Decoder blocks (+channel fixed for concat)
        self.up3 = nn.ConvTranspose2d(base_channels*8, base_channels*4, 2, 2)
        self.dec3 = nn.Sequential(
            nn.Conv2d(base_channels*8, base_channels*4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            RDBWithAttention(base_channels*4),
            RDBWithAttention(base_channels*4),
        )
        self.up2 = nn.ConvTranspose2d(base_channels*4, base_channels*2, 2, 2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(base_channels*4, base_channels*2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            RDBWithAttention(base_channels*2),
            RDBWithAttention(base_channels*2),
        )
        self.up1 = nn.ConvTranspose2d(base_channels*2, base_channels, 2, 2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(base_channels*2, base_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            RDBWithAttention(base_channels),
            RDBWithAttention(base_channels),
        )
        self.final_conv = nn.Conv2d(base_channels, 3, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        e3 = self.enc3(p2)
        p3 = self.pool3(e3)

        b = self.bottleneck_conv(p3)
        if self.use_transformer:
            b_shape = b.shape
            # Transformer expects [batch, seq_len, embedding]
            b_flat = b.flatten(2).permute(0, 2, 1)
            b_trans = self.transformer(b_flat)
            b = b_trans.permute(0, 2, 1).view(b_shape)

        u3 = self.up3(b)
        cat3 = torch.cat((u3, e3), dim=1)
        d3 = self.dec3(cat3)

        u2 = self.up2(d3)
        cat2 = torch.cat((u2, e2), dim=1)
        d2 = self.dec2(cat2)

        u1 = self.up1(d2)
        cat1 = torch.cat((u1, e1), dim=1)
        d1 = self.dec1(cat1)

        out = self.final_conv(d1)
        return out