import torch
import torch.nn as nn


class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=None, groups=1, bias=False,
                 activation=True):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding if padding is not None else kernel_size // 2,
            groups=groups, bias=bias
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU(inplace=True) if activation else nn.Identity()

    def forward(self, x):
        return self.activation(self.bn(self.conv(x)))
class CoordinateAttention(nn.Module):
    def __init__(self, in_channels, reduction=32):
        super().__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        reduced_channels = max(8, in_channels // reduction)
        self.conv1 = nn.Conv2d(in_channels * 2, reduced_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(reduced_channels)
        self.act = nn.ReLU(inplace=True)
        self.conv_h = nn.Conv2d(reduced_channels, in_channels, kernel_size=1, bias=False)
        self.conv_w = nn.Conv2d(reduced_channels, in_channels, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        identity = x
        B, C, H, W = x.shape

        h_avg = self.pool_h(x).permute(0, 1, 3, 2)  # (B, C, W, 1)
        w_avg = self.pool_w(x)  # (B, C, 1, H)

        x_cat = torch.cat([h_avg, w_avg], dim=1)  # 拼接在 `C` 维度，确保 `conv1` 处理正确
        x = self.act(self.bn1(self.conv1(x_cat)))

        h_weight = self.sigmoid(self.conv_h(x))
        w_weight = self.sigmoid(self.conv_w(x))

        return identity * h_weight * w_weight


class MobileViTBackbone(nn.Module):
    def __init__(self, out_channels=[256, 512, 1024], pretrained=False):
        super().__init__()
        self.pretrained = pretrained
        if self.pretrained:
            print("Loading MobileViT pretrained weights...")

        self.stem = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU6(inplace=True)
        )

        self.mv2_1 = Conv(16, 32, kernel_size=3, stride=1)
        self.mv2_2 = Conv(32, 64, kernel_size=3, stride=2)
        self.mv2_3 = Conv(64, 128, kernel_size=3, stride=2)

        self.ca1 = CoordinateAttention(128)
        self.mv2_4 = Conv(128, 256, kernel_size=3, stride=2)
        self.ca2 = CoordinateAttention(256)
        self.mv2_5 = Conv(256, 512, kernel_size=3, stride=2)
        self.ca3 = CoordinateAttention(512)

        self.sppf = Conv(512, 1024, kernel_size=1, stride=1)

    def forward(self, x):
        x = self.stem(x)
        x = self.mv2_1(x)
        x = self.mv2_2(x)
        feat1 = self.ca1(self.mv2_3(x))

        feat2 = self.ca2(self.mv2_4(feat1))
        feat3 = self.ca3(self.mv2_5(feat2))
        sppf_out = self.sppf(feat3)

        return feat1, feat2, feat3, sppf_out


__all__ = ['Conv', 'CoordinateAttention', 'MobileViTBackbone', ]
