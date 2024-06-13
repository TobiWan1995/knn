import numpy as np
import torch
from torch import Generator
from torch.utils.data import Dataset, DataLoader, random_split
from utils.data.preprocess import Preprocessor


def custom_collate(batch):
    features, labels = zip(*batch)
    features = torch.tensor(np.array(features), dtype=torch.float32)
    labels = torch.tensor(np.array(labels), dtype=torch.long)
    return features, labels


class BaseDataset(Dataset):
    def __init__(self):
        super().__init__()

        self.labels = None
        self.features = None
        self.data = None

        self.preprocessor = Preprocessor(self)  # Initialisiert den Preprocessor mit dieser Instanz

    def __len__(self):
        return len(self.data) if self.data is not None else 0

    def __getitem__(self, idx):
        raise NotImplementedError("This method should be implemented by subclasses.")

    def get_splits(self, train_frac=0.7, test_frac=0.15, valid_frac=0.15, batch_size=32, randomSeed=42, shuffle=True):
        """
        Aufteilen des Datensatzes in Trainings-, Validierungs- und Testdatensätze.

        - train_frac: Anteil der Trainingsdaten
        - valid_frac: Anteil der Validierungsdaten
        - test_frac: Anteil der Testdaten wird berechnet aus Rest
        - batch_size: Größe der Batches für den DataLoader
        - randomSeed: Random Seed zum Reproduzieren der Datensatze
        - shuffle: Ob die Daten vor dem Erstellen der Batches gemischt werden sollen

        Die Methode berechnet die Anzahl der Beispiele für jeden Split, teilt den Datensatz mithilfe von
        random_split und erstellt DataLoader für jeden Split.
        """
        # Überprüfen, ob die Fraktionen insgesamt 1.0 ergeben
        if train_frac + valid_frac + test_frac != 1:
            raise ValueError("Die Summe der Fraktionen für Training, Test und Validierung muss 1 ergeben!")

        total_count = len(self)  # Gesamtzahl der Beispiele im Datensatz
        train_count = int(train_frac * total_count)  # Anzahl der Trainingsbeispiele
        valid_count = int(valid_frac * total_count)  # Anzahl der Validierungsbeispiele
        test_count = total_count - train_count - valid_count  # Anzahl der Testbeispiele

        generator = Generator().manual_seed(randomSeed)
        train_dataset, valid_dataset, test_dataset = random_split(self, [train_count, valid_count, test_count], generator=generator)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=custom_collate)
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=custom_collate)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=custom_collate)

        return train_loader, valid_loader, test_loader
