import torch
import torch.nn as nn
import torch.nn.functional as f


# Generische LayerWrapper Klasse
class LayerWrapper(nn.Module):
    def __init__(self, layer, acf=None, init_type=None):
        super(LayerWrapper, self).__init__()
        self.layer = layer
        self.acf = acf
        self.init_type = init_type
        self.initialize_weights()

    def initialize_weights(self):
        if isinstance(self.layer, (nn.Linear, nn.Conv2d)):
            if self.init_type == 'he':
                nn.init.kaiming_normal_(self.layer.weight, nonlinearity='relu')
            elif self.init_type == 'xavier':
                nn.init.xavier_normal_(self.layer.weight)
            if self.layer.bias is not None:
                nn.init.zeros_(self.layer.bias)

    def forward(self, x):
        x = self.layer(x)
        if self.acf:
            if self.acf == f.softmax:
                x = self.acf(x, dim=1)  # Softmax benötigt die Dimension
            else:
                x = self.acf(x)
        return x


# Spezifische Layer Wrapper
class DenseLayer(LayerWrapper):
    def __init__(self, input_dim, output_dim, acf=f.relu, init_type='he'):
        layer = nn.Linear(input_dim, output_dim)
        super(DenseLayer, self).__init__(layer, acf, init_type)


class ConvLayer(LayerWrapper):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=None, acf=f.relu, init_type='he'):
        if groups is None:
            layer = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        else:
            layer = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=in_channels)
        super(ConvLayer, self).__init__(layer, acf, init_type)


class PoolingLayer(nn.Module):
    def __init__(self, kernel_size, stride=None, padding=0, mode='max'):
        super(PoolingLayer, self).__init__()
        if mode == 'max':
            self.layer = nn.MaxPool2d(kernel_size, stride, padding)
        elif mode == 'avg':
            self.layer = nn.AvgPool2d(kernel_size, stride, padding)
        else:
            raise ValueError("Unsupported pooling mode. Choose 'max' or 'avg'.")

    def forward(self, x):
        return self.layer(x)


class FlattenLayer(nn.Module):
    def forward(self, x):
        return torch.flatten(x, 1)


class DropoutLayer(nn.Module):
    def __init__(self, p=0.5):
        super(DropoutLayer, self).__init__()
        self.dropout = nn.Dropout(p)

    def forward(self, x):
        return self.dropout(x)


class BatchNormLayer(nn.Module):
    def __init__(self, num_features, dim='1d'):
        super(BatchNormLayer, self).__init__()
        if dim == '1d':
            self.batchnorm = nn.BatchNorm1d(num_features)
        elif dim == '2d':
            self.batchnorm = nn.BatchNorm2d(num_features)
        elif dim == '3d':
            self.batchnorm = nn.BatchNorm3d(num_features)
        else:
            raise ValueError("Unsupported dimension. Choose '1d', '2d', or '3d'.")

    def forward(self, x):
        return self.batchnorm(x)


# Definiere das MultiLayer-Modell
class MultiLayerModel(nn.Module):
    def __init__(self, layers, loss_fn, task='binary', num_classes=0):
        super(MultiLayerModel, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.loss_fn = loss_fn
        self.task = task
        self.num_classes = num_classes

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
