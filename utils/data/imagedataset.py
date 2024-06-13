import os
from PIL import Image
from utils.data.basedataset import BaseDataset
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF


class ImageDataset(BaseDataset):
    def __init__(self, image_folder, transform=None):
        super().__init__()
        self.images = []
        self.labels = []
        self.transform = transform or transforms.ToTensor()
        self.load_from_image_folder(image_folder)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

    def load_from_image_folder(self, folder_path):
        for root, _, files in os.walk(folder_path):
            for file in files:
                if file.lower().endswith(('png', 'jpg', 'jpeg')):
                    self.images.append(os.path.join(root, file))
                    self.labels.append(os.path.basename(root))

    def show_image(self, idx):
        image, label = self.__getitem__(idx)
        plt.imshow(TF.to_pil_image(image))
        plt.title(f'Label: {label}')
        plt.axis('off')
        plt.show()
