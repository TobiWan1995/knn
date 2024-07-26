import torch.nn as nn
import torch.nn.functional as f

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from utils.torch.trainer import *


class SimpleCnnNet(nn.Module):
    def __init__(self):
        super(SimpleCnnNet, self).__init__()
        # Erste Convolutional-Schicht
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        nn.init.kaiming_normal_(self.conv1.weight, nonlinearity='relu')
        if self.conv1.bias is not None:
            nn.init.zeros_(self.conv1.bias)

        # Erste Pooling-Schicht
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Zweite Convolutional-Schicht
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        nn.init.kaiming_normal_(self.conv2.weight, nonlinearity='relu')
        if self.conv2.bias is not None:
            nn.init.zeros_(self.conv2.bias)

        # Zweite Pooling-Schicht
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Flatten-Schicht (als eigene Methode im forward-Method)

        # Erste Dense-Schicht
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity='relu')
        if self.fc1.bias is not None:
            nn.init.zeros_(self.fc1.bias)

        # Zweite Dense-Schicht
        self.fc2 = nn.Linear(128, 10)
        nn.init.kaiming_normal_(self.fc2.weight)
        if self.fc2.bias is not None:
            nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        x = f.relu(self.conv1(x))
        x = self.pool1(x)
        x = f.relu(self.conv2(x))
        x = self.pool2(x)
        x = torch.flatten(x, 1)
        x = f.relu(self.fc1(x))
        x = self.fc2(x)
        return x


transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

train_dataset = datasets.FashionMNIST(root='../../data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

valid_dataset = datasets.FashionMNIST(root='../../data', train=False, download=True, transform=transform)
valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)

trainer = Trainer(SimpleCnnNet, nn.CrossEntropyLoss(), train_loader, valid_loader, num_classes=10, task="multiclass")
trainer.train(epochs=100, title="Fashion_MNIST_Conv_100")