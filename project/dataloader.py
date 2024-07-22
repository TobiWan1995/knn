import os
from PIL import Image
from torch.utils.data import Dataset, random_split


class CustomFlickr30k(Dataset):
    def __init__(self, root, captions_file, transform=None, target_transform=None):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.image_captions = self._load_captions(captions_file)
        self.images = list(self.image_captions.keys())
        self.labels = list(self.image_captions.values())

    def _load_captions(self, captions_file):
        image_captions = {}
        with open(captions_file, 'r') as f:
            for line in f:
                img_name, caption = line.strip().split(',', 1)
                image_captions[img_name] = caption
        return image_captions

    def __len__(self):
        return len(self.image_captions)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.root, img_name)
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


def get_splits(dataset, train_split=0.8, val_split=0.1, test_split=0.1):
    assert train_split + val_split + test_split == 1, "Splits must sum to 1"

    total_size = len(dataset)
    train_size = int(train_split * total_size)
    val_size = int(val_split * total_size)
    test_size = total_size - train_size - val_size

    return random_split(dataset, [train_size, val_size, test_size])
