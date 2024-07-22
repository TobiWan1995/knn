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
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ]),
}

# Lade den CIFAR-100 Datensatz
train_dataset = datasets.CIFAR100(root='./datasets', train=True, download=True, transform=data_transforms['train'])
val_dataset = datasets.CIFAR100(root='./datasets', train=False, download=True, transform=data_transforms['val'])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

dataloaders = {'train': train_loader, 'val': val_loader}
dataset_sizes = {'train': len(train_dataset), 'val': len(val_dataset)}
class_names = train_dataset.classes

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Trainiere und evaluiere das Modell
vgg16 = create_custom_vgg16(len(class_names))
vgg16 = default_train_model(vgg16, dataloaders, dataset_sizes, device, num_epochs=25, title='vgg16_cifar100')

# Speichere das trainierte Modell
# torch.save(model.state_dict(), os.path.join(base_path, 'vgg16_flickr30k.pth'))
