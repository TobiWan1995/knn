from torchvision import models
from utils.torch.torchlayers import *

# Funktion zum Erstellen von Modellen
def create_model(model_name, num_classes):
    if model_name == 'vgg16':
        model = models.vgg16(pretrained=True)
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs, num_classes)
    else:
        raise ValueError("Modellname nicht erkannt")
    return model

def create_custom_model(layers, loss_fn, task='binary', num_classes=0):
    return MultiLayerModel(layers, loss_fn, task, num_classes)

def create_custom_vgg16(num_classes):
    layers = [
        ConvLayer(3, 64, kernel_size=3, padding=1),
        ConvLayer(64, 64, kernel_size=3, padding=1),
        PoolingLayer(kernel_size=2, stride=2, mode='max'),

        ConvLayer(64, 128, kernel_size=3, padding=1),
        ConvLayer(128, 128, kernel_size=3, padding=1),
        PoolingLayer(kernel_size=2, stride=2, mode='max'),

        ConvLayer(128, 256, kernel_size=3, padding=1),
        ConvLayer(256, 256, kernel_size=3, padding=1),
        ConvLayer(256, 256, kernel_size=3, padding=1),
        PoolingLayer(kernel_size=2, stride=2, mode='max'),

        ConvLayer(256, 512, kernel_size=3, padding=1),
        ConvLayer(512, 512, kernel_size=3, padding=1),
        ConvLayer(512, 512, kernel_size=3, padding=1),
        PoolingLayer(kernel_size=2, stride=2, mode='max'),

        ConvLayer(512, 512, kernel_size=3, padding=1),
        ConvLayer(512, 512, kernel_size=3, padding=1),
        ConvLayer(512, 512, kernel_size=3, padding=1),
        PoolingLayer(kernel_size=2, stride=2, mode='max'),

        FlattenLayer(),
        DenseLayer(512 * 7 * 7, 4096),
        DropoutLayer(0.5),
        DenseLayer(4096, 4096),
        DropoutLayer(0.5),
        DenseLayer(4096, num_classes, acf=f.softmax)
    ]
    return MultiLayerModel(layers, nn.CrossEntropyLoss(), task='multiclass', num_classes=num_classes)