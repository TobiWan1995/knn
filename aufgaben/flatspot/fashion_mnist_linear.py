import torch.nn as nn
import torch.nn.functional as f

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from utils.torch.trainer import *


class SimpleDenseNet(nn.Module):
    def __init__(self):
        super(SimpleDenseNet, self).__init__()
        # Erste Dense-Schicht
        self.fc1 = nn.Linear(28 * 28, 128)
        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity='relu')
        if self.fc1.bias is not None:
            nn.init.zeros_(self.fc1.bias)

        # Zweite Dense-Schicht
        self.fc2 = nn.Linear(128, 64)
        nn.init.kaiming_normal_(self.fc2.weight, nonlinearity='relu')
        if self.fc2.bias is not None:
            nn.init.zeros_(self.fc2.bias)

        # Dritte Dense-Schicht
        self.fc3 = nn.Linear(64, 10)
        nn.init.kaiming_normal_(self.fc3.weight)
        if self.fc3.bias is not None:
            nn.init.zeros_(self.fc3.bias)

    def forward(self, x):
        x = torch.flatten(x, 1)  # Flatten der Eingabe
        x = f.relu(self.fc1(x))
        x = f.relu(self.fc2(x))
        x = self.fc3(x)  # Keine Aktivierungsfunktion für die letzte Schicht
        return x


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

trainer = Trainer(SimpleDenseNet(), nn.CrossEntropyLoss(), train_loader, valid_loader, num_classes=10, task="multiclass")
trainer.train(epochs=100, title='Fashion_MNIST_Linear_100')