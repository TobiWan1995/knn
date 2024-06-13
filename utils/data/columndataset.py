import pandas as pd
from utils.data.basedataset import BaseDataset
from ucimlrepo import fetch_ucirepo


class ColumnDataset(BaseDataset):
    def __init__(self, ucimlid=None, csv_path=None, label_column=None):
        super().__init__()
        self.data = None  # Initialisierung des DataFrames korrigiert von pd.DataFrame auf None
        self.label_column = label_column

        if ucimlid is not None:
            self.load_from_ucimlrepo(ucimlid)
        elif csv_path is not None:
            self.load_from_csv(csv_path)
        else:
            raise ValueError("Either ucimlid or csv_path must be provided")

        # Setzt label_column als letzte Spalte, falls keine angegeben wurde
        if self.label_column is None and self.data is not None:
            self.label_column = self.data.columns[-1]

        self.setup_features_labels()

    def __getitem__(self, idx):
        # Sicherstellen, dass beim Zugriff die Daten korrekt konvertiert sind
        feature = self.features[idx].astype('float32')
        label = self.labels[idx].astype('long') if pd.api.types.is_numeric_dtype(self.labels) else self.labels[idx]
        return feature, label

    def __len__(self):
        return len(self.data) if self.data is not None else 0

    def load_from_ucimlrepo(self, ucimlid):
        data = fetch_ucirepo(id=ucimlid)
        self.data = pd.concat([data.data.features, data.data.targets], axis=1)

    def load_from_csv(self, csv_path):
        self.data = pd.read_csv(csv_path)

    def setup_features_labels(self):
        if self.label_column not in self.data.columns:
            raise ValueError(f"Label column '{self.label_column}' not found in data.")
        self.features = self.data.drop(columns=[self.label_column]).values
        self.labels = self.data[self.label_column].values

    def change_label_column(self, new_label_column):
        if new_label_column not in self.data.columns:
            raise ValueError(f"Label column '{new_label_column}' not found in data.")
        self.label_column = new_label_column
        self.setup_features_labels()
