import torch
import torch.nn as nn
import torch.nn.functional as F
import transformer


# Blocks

class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SEBlock, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(in_channels, in_channels // reduction, bias=False)
        self.fc2 = nn.Linear(in_channels // reduction, in_channels, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.global_avg_pool(x).view(b, c)
        y = F.relu(self.fc1(y), inplace=True)
        y = self.sigmoid(self.fc2(y)).view(b, c, 1, 1)
        return x * y


class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv1(x))


class SEWithSpatialAttention(nn.Module):
    def __init__(self, in_channels):
        super(SEWithSpatialAttention, self).__init__()
        self.se = SEBlock(in_channels)
        self.spatial = SpatialAttention()

    def forward(self, x):
        x = self.se(x)
        return x * self.spatial(x)


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None,
                 activation='leakyrelu'):
        super(DoubleConv, self).__init__()
        if not mid_channels:
            mid_channels = out_channels

        if activation == 'leakyrelu':
            act_layer = nn.LeakyReLU(0.1, inplace=True)
        elif activation == 'prelu':
            act_layer = nn.PReLU()
        else:
            act_layer = nn.ReLU(inplace=True)

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            act_layer,
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            act_layer,
        )

    def forward(self, x):
        return self.double_conv(x)


class dnn90(nn.Module):
    """Small CNN classifier: [B, 2, H, W] -> [B, num_classes]."""
    def __init__(self, input_channels=2, num_classes=512):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(128)

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc1 = nn.Linear(128, 256)
        # Wrapped in Sequential to match checkpoint keys (fc2.0.*)
        self.fc2 = nn.Sequential(
            nn.Linear(256, num_classes),
        )

        self.act = nn.SiLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.act(self.bn1(self.conv1(x)))
        x = self.act(self.bn2(self.conv2(x)))
        x = self.act(self.bn3(self.conv3(x)))
        x = self.act(self.bn4(self.conv4(x)))   # [B, 128, H/16, W/16]

        x = self.global_pool(x).flatten(1)      # [B, 128]
        x = self.dropout(self.act(self.fc1(x))) # [B, 256]
        return self.fc2(x)                      # [B, num_classes]


# Model

class LoRaSeekSF9Large(nn.Module):
    """
    Input:  noisy STFT, [B, 2, H, W] (real/imag)
    Output: denoised STFT [B, 2, H, W] + logits [B, 512]
    """
    # drop_rate is unused; kept so existing callers passing it still work.
    def __init__(self, drop_rate=0.1):
        super().__init__()

        # Encoder 0
        self.dc0 = DoubleConv(2, 32)
        self.se_spatial_skip0 = SEWithSpatialAttention(32)

        # Encoder 1
        self.downsample1 = nn.Conv2d(32, 64, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        self.dc1 = DoubleConv(64, 64)
        self.se_spatial_skip1 = SEWithSpatialAttention(64)

        # Encoder 2
        self.downsample2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(3, 1), stride=(2, 1)),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.transformer_downsample2 = transformer.TransformerNoCLS2DHigherLevel(
            dim=128, depth=1, num_heads=8, expansion=1.5, bias=False)
        self.se_spatial_skip2 = SEWithSpatialAttention(128)

        # Bottleneck
        self.downsample3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=(1, 0)),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.transformer_downsample3 = transformer.TransformerNoCLS2DHigherLevel(
            dim=256, depth=1, num_heads=16, expansion=2, bias=False)

        # Decoder 2
        self.upconv3_2 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 0)),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
        )

        self.reduce_chan_upconv3_2 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1, bias=False),
        )
        self.transformer_up3 = transformer.TransformerNoCLS2DHigherLevel(
            dim=128, depth=1, num_heads=8, expansion=1.5, bias=False)

        # Decoder 1
        self.upconv2_1 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=(4, 1), stride=(2, 1)),
        )
        self.dc21 = DoubleConv(128, 64)

        # Decoder 0
        self.upconv1_0 = nn.ConvTranspose2d(64, 32, kernel_size=(4, 3), stride=(2, 1), padding=(1, 1))
        self.dc10 = DoubleConv(64, 32)
        self.last = nn.Conv2d(32, 2, 3, 1, 1)

        self.classifier = dnn90()

    def forward(self, x):
        # Encoder
        x = self.dc0(x)                          # [B, 32, 512, 33]
        skip0 = self.se_spatial_skip0(x)

        x = self.downsample1(x)                  # [B, 64, 256, 33]
        x = self.dc1(x)
        skip1 = self.se_spatial_skip1(x)

        x = self.downsample2(x)                  # [B, 128, 127, 33]
        x = self.transformer_downsample2(x)
        skip2 = self.se_spatial_skip2(x)

        # Bottleneck
        x = self.downsample3(x)                  # [B, 256, 64, 16]
        x = self.transformer_downsample3(x)

        # Decoder
        x = self.upconv3_2(x)                    # [B, 128, 127, 33]
        x = self.reduce_chan_upconv3_2(torch.cat([x, skip2], dim=1))
        x = self.transformer_up3(x)

        x = self.upconv2_1(x)                    # [B, 64, 256, 33]
        x = self.dc21(torch.cat([x, skip1], dim=1))

        x = self.upconv1_0(x)                    # [B, 32, 512, 33]
        x = self.dc10(torch.cat([x, skip0], dim=1))

        denoise_stft = self.last(x)              # [B, 2, 512, 33]
        logits = self.classifier(denoise_stft)   # [B, 512]
        return denoise_stft, logits


if __name__ == "__main__":
    model = LoRaSeekSF9Large()
    x = torch.randn(1, 2, 512, 33)
    out, logits = model(x)
    print("Denoised STFT:", tuple(out.shape), "| logits:", tuple(logits.shape))

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    classifier_params = sum(p.numel() for p in model.classifier.parameters() if p.requires_grad)
    print(f"Parameters: {total_params - classifier_params} (denoiser) + "
          f"{classifier_params} (classifier) = {total_params}")
