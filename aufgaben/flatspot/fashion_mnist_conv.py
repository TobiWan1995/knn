from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from utils.torch.torchlayers import *
from utils.torch.torchtrain import *


transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

valid_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)


# Initialisierung des Netzwerks mit Convolutional Layern
layers = [
    ConvLayer(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1, acf=f.relu, init_type='he'),
    PoolingLayer(kernel_size=2, stride=2, mode='max'),
    ConvLayer(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1, acf=f.relu, init_type='he'),
    PoolingLayer(kernel_size=2, stride=2, mode='max'),
    FlattenLayer(),
    DenseLayer(input_dim=64*7*7, output_dim=128, acf=f.relu, init_type='he'),
    DenseLayer(input_dim=128, output_dim=10, acf=f.softmax, init_type='he')  # Keine Aktivierungsfunktion für CrossEntropyLoss
]
criterion = nn.CrossEntropyLoss()  # Beispiel für Mehrklassenklassifikation

conv_model = MultiLayerModel(layers, loss_fn=criterion, task='multiclass', num_classes=10)

train_torch(conv_model, train_loader, valid_loader, epochs=100, lr=0.001, title="Fashion_MNIST_Conv_100")
