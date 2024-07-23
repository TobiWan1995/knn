from torch.utils.data import DataLoader
from torchvision import transforms, datasets

from model import *
from train import *

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
train_dataset = datasets.CIFAR100(root='./datasets', train=True, download=True, transform=data_transforms['train'])
val_dataset = datasets.CIFAR100(root='./datasets', train=False, download=True, transform=data_transforms['val'])

# Verwende eine kleinere Teilmenge des Datensatzes
train_subset, _ = torch.utils.data.random_split(train_dataset, [int(0.5 * len(train_dataset)), int(0.5 * len(train_dataset))])
val_subset, _ = torch.utils.data.random_split(val_dataset, [int(0.5 * len(val_dataset)), int(0.5 * len(val_dataset))])

train_loader = DataLoader(train_subset, batch_size=100, shuffle=True, num_workers=4)
val_loader = DataLoader(val_subset, batch_size=100, shuffle=False, num_workers=4)

dataloaders = {'train': train_loader, 'val': val_loader}
dataset_sizes = {'train': len(train_subset), 'val': len(val_subset)}
class_names = train_dataset.classes

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Trainiere und evaluiere das Modell
cnet = EfficientNetB0(len(class_names))
cnet = default_train_model(cnet, dataloaders, dataset_sizes, device, num_epochs=200, title='EfficientNetB0-Cifar100')
# Speichere das trainierte Modell
base_path = os.path.dirname(os.path.abspath(__file__))  # Absoluter Pfad zum Verzeichnis dieser Datei
torch.save(cnet.state_dict(), os.path.join(base_path, 'EfficientNetB0-Cifar100.pth'))
