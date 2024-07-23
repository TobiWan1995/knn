from utils.torch.torchlayers import *


class EfficientNetB0(nn.Module):
    def __init__(self, num_classes=100):
        super(EfficientNetB0, self).__init__()
        self.stem = nn.Sequential(
            ConvLayer(3, 32, kernel_size=3, stride=2, padding=1, acf=Swish()),
            BatchNormLayer(32, dim='2d')
        )
        self.blocks = nn.Sequential(
            # Block 1
            DepthwiseSeparableConv(32, 16, kernel_size=3, padding=1, acf=Swish()),
            BatchNormLayer(16, dim='2d'),

            # Block 2
            DepthwiseSeparableConv(16, 24, kernel_size=3, stride=2, padding=1, acf=Swish()),
            BatchNormLayer(24, dim='2d'),

            # Block 3
            DepthwiseSeparableConv(24, 40, kernel_size=5, stride=2, padding=2, acf=Swish()),
            BatchNormLayer(40, dim='2d'),

            # Block 4
            DepthwiseSeparableConv(40, 80, kernel_size=3, stride=2, padding=1, acf=Swish()),
            BatchNormLayer(80, dim='2d'),

            # Block 5
            DepthwiseSeparableConv(80, 112, kernel_size=5, padding=2, acf=Swish()),
            BatchNormLayer(112, dim='2d'),

            # Block 6
            DepthwiseSeparableConv(112, 192, kernel_size=5, stride=2, padding=2, acf=Swish()),
            BatchNormLayer(192, dim='2d'),

            # Block 7
            DepthwiseSeparableConv(192, 320, kernel_size=3, padding=1, acf=Swish()),
            BatchNormLayer(320, dim='2d'),
        )
        self.head = nn.Sequential(
            ConvLayer(320, 1280, kernel_size=1, acf=Swish()),
            BatchNormLayer(1280, dim='2d'),
            nn.AdaptiveAvgPool2d(1),
            FlattenLayer(),
            DropoutLayer(0.2),
            nn.Linear(1280, num_classes)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)
        return x


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, acf=f.relu, init_type='he'):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = ConvLayer(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels, acf=acf,
                                   init_type=init_type)
        self.pointwise = ConvLayer(in_channels, out_channels, kernel_size=1, acf=acf, init_type=init_type)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x
