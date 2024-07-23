from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from utils.torch.torchlayers import *
from utils.torch.torchtrain import *


# Transformation, um die Daten zu flachen
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
    transforms.Lambda(lambda x: x.view(-1))  # Flacht die Bilder auf (28*28,) ab
])

# Laden des Fashion-MNIST-Datensatzes mit der neuen Transformation
train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

valid_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)

# Initialisierung des Netzwerks nur mit Linear-Layern
layers = [
    DenseLayer(input_dim=28*28, output_dim=128, acf=f.relu, init_type='he'),
    DenseLayer(input_dim=128, output_dim=64, acf=f.relu, init_type='he'),
    DenseLayer(input_dim=64, output_dim=10, init_type='he')
]
criterion = nn.CrossEntropyLoss()  # Beispiel für Mehrklassenklassifikation

linear_model = MultiLayerModel(layers, loss_fn=criterion, task='multiclass', num_classes=10)

train_torch(linear_model, train_loader, valid_loader, epochs=100, lr=0.001, title='Fashion_MNIST_Linear_100')