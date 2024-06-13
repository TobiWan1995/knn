from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from utils.torch.torchlayers import *
from utils.torch.torchtrain import *

train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
    transforms.Lambda(lambda x: x.view(-1))  # Flacht die Bilder auf (28*28,) ab
])

valid_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
    transforms.Lambda(lambda x: x.view(-1))  # Flacht die Bilder auf (28*28,) ab
])

train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=train_transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

valid_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=valid_transform)
valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)

# Initialisierung des Netzwerks mit Dropout und Batch Normalization
layers = [
    DenseLayer(input_dim=28*28, output_dim=128, acf=F.relu, init_type='he'),
    BatchNormLayer(num_features=128),
    DropoutLayer(p=0.5),
    DenseLayer(input_dim=128, output_dim=64, acf=F.relu, init_type='he'),
    BatchNormLayer(num_features=64),
    DropoutLayer(p=0.5),
    DenseLayer(input_dim=64, output_dim=10, acf=nn.Softmax(dim=1), init_type='he')  # Softmax für CrossEntropyLoss
]
criterion = nn.CrossEntropyLoss()  # Beispiel für Mehrklassenklassifikation

model = MultiLayerModel(layers, loss_fn=criterion, task='multiclass', num_classes=10)

train_torch(model, train_loader, valid_loader, epochs=50, lr=0.001, title="FashionMNIST_Linear_Regulated", warmup_steps=1000, weight_decay=0.01, early_stopping_patience=5)