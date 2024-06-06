import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder


class Preprocessor:
    def __init__(self, dataset):
        self.dataset = dataset
        self.scaler = StandardScaler()
        self.encoder = LabelEncoder()

    def fill_missing_values(self):
        numeric_cols = self.dataset.data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            self.dataset.data[col] = self.dataset.data[col].fillna(self.dataset.data[col].mean())
        self.update_features_labels()

    def encode_categorical(self):
        cat_cols = self.dataset.data.select_dtypes(include=['object']).columns
        for col in cat_cols:
            self.dataset.data[col] = self.encoder.fit_transform(self.dataset.data[col])
        self.update_features_labels()

    def normalize_columns(self):
        numeric_cols = self.dataset.data.select_dtypes(include=[np.number]).columns.tolist()
        self.dataset.data[numeric_cols] = self.scaler.fit_transform(self.dataset.data[numeric_cols])
        self.update_features_labels()

    def drop_features(self, columns):
        self.dataset.data.drop(columns=columns, inplace=True)
        self.update_features_labels()

    def drop_rows_by_values(self, values_to_remove):
        self.dataset.data = self.dataset.data[~self.dataset.data.isin(values_to_remove).any(axis=1)]
        self.update_features_labels()

    def update_features_labels(self):
        self.dataset.features = self.dataset.data.drop(columns=[self.dataset.label_column]).values.astype(np.float32)
        self.dataset.labels = self.dataset.data[self.dataset.label_column].values.astype(np.int32)
