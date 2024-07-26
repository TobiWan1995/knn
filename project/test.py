from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import torch.optim as optim

from model import *
from utils.torch.trainer import *

# Definiere die Transformationen für den Datensatz
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ]),
    'val': transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ]),
}

# Lade den CIFAR-100 Datensatz
train_dataset = datasets.CIFAR100(root='../data', train=True, download=True, transform=data_transforms['train'])
val_dataset = datasets.CIFAR100(root='../data', train=False, download=True, transform=data_transforms['val'])

train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True, num_workers=4)
valid_loader = DataLoader(val_dataset, batch_size=100, shuffle=False, num_workers=4)
class_names = train_dataset.classes


# Trainiere und evaluiere das Modell
effnet0b = EfficientNetB0(len(class_names))
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(effnet0b.parameters(), lr=0.01)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.1)

trainer = Trainer(effnet0b, criterion, train_loader, valid_loader, num_classes=len(class_names), task="multiclass", optimizer=optimizer, scheduler=scheduler)
trainer.train(epochs=10, title='EfficientNetB0-Cifar100')

# Speichere das trainierte Modell
# base_path = os.path.dirname(os.path.abspath(__file__))  # Absoluter Pfad zum Verzeichnis dieser Datei
# torch.save(effnet0b.state_dict(), os.path.join(base_path, 'EfficientNetB0-Cifar100.pth'))
