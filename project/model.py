import torch
import torch.nn as nn
import torch.nn.functional as F


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, acf=F.relu):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.acf = acf
        self.init_weights()

    def init_weights(self):
        nn.init.kaiming_normal_(self.depthwise.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.pointwise.weight, nonlinearity='relu')
        if self.depthwise.bias is not None:
            nn.init.zeros_(self.depthwise.bias)
        if self.pointwise.bias is not None:
            nn.init.zeros_(self.pointwise.bias)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.acf(x)
        x = self.pointwise(x)
        x = self.acf(x)
        return x


class EfficientNetB0(nn.Module):
    def __init__(self, num_classes=100):
        super(EfficientNetB0, self).__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            Swish()
        )
        self.blocks = nn.Sequential(
            # Block 1
            DepthwiseSeparableConv(32, 16, kernel_size=3, padding=1, acf=Swish()),
            nn.BatchNorm2d(16),

            # Block 2
            DepthwiseSeparableConv(16, 24, kernel_size=3, stride=2, padding=1, acf=Swish()),
            nn.BatchNorm2d(24),

            # Block 3
            DepthwiseSeparableConv(24, 40, kernel_size=5, stride=2, padding=2, acf=Swish()),
            nn.BatchNorm2d(40),

            # Block 4
            DepthwiseSeparableConv(40, 80, kernel_size=3, stride=2, padding=1, acf=Swish()),
            nn.BatchNorm2d(80),

            # Block 5
            DepthwiseSeparableConv(80, 112, kernel_size=5, padding=2, acf=Swish()),
            nn.BatchNorm2d(112),

            # Block 6
            DepthwiseSeparableConv(112, 192, kernel_size=5, stride=2, padding=2, acf=Swish()),
            nn.BatchNorm2d(192),

            # Block 7
            DepthwiseSeparableConv(192, 320, kernel_size=3, padding=1, acf=Swish()),
            nn.BatchNorm2d(320),
        )
        self.head = nn.Sequential(
            nn.Conv2d(320, 1280, kernel_size=1),
            nn.BatchNorm2d(1280),
            Swish(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(1280, num_classes)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)
        return x
