from dataset.columndataset import ColumnDataset
from utils import acf as acf
from utils import knn_utils_02 as k
import autograd.numpy as anp

"""
Der 'adult' Datensatz, auch bekannt als 'Census Income' Datensatz, wurde aus dem 1994 Census 
Extraktionsprozess gewonnen. Er enthält Informationen über Individuen, einschließlich ihrer 
demografischen Daten und Arbeitsmerkmale. Das Ziel dieses Datensatzes ist es, vorherzusagen, 
ob das Einkommen einer Person über 50.000 USD pro Jahr liegt.

Die Features umfassen Attribute wie Alter, Arbeitsklasse, Bildung, Familienstand, Beruf, 
Geschlecht und viele mehr. Dieser Datensatz wird häufig für Klassifikationsprobleme verwendet, 
bei denen das Ziel darin besteht, das Einkommensniveau (über oder unter 50.000 USD) vorherzusagen.
"""

# Laden des adult Datensatzes aus dem UCI ML Repository
dataset = ColumnDataset(ucimlid=2)

# Darstellung der ersten paar Zeilen des Datensatzes zur Überprüfung
print(dataset.data.head())

# Vorverarbeitung der Daten
dataset.preprocessor.encode_categorical()
dataset.preprocessor.fill_missing_values()
dataset.preprocessor.normalize_columns()

# Nutzung der get_splits Methode zur Aufteilung in Trainings-, Validierungs- und Testdaten
train_loader, valid_loader, test_loader = dataset.get_splits(train_frac=0.7, valid_frac=0.15, batch_size=32)

# Konvertieren der DataLoader zu numpy arrays für die Verwendung im neuronalen Netz
X_train = anp.concatenate([x for x, _ in train_loader], axis=0)
Y_train = anp.concatenate([y for _, y in train_loader], axis=0)
X_valid = anp.concatenate([x for x, _ in valid_loader], axis=0)
Y_valid = anp.concatenate([y for _, y in valid_loader], axis=0)
X_test = anp.concatenate([x for x, _ in test_loader], axis=0)
Y_test = anp.concatenate([y for _, y in test_loader], axis=0)

# Netzwerk initialisieren
layers = [
    k.DenseLayer(X_train.shape[1], 64, acf=acf.tanh, init_type='he'),
    k.DenseLayer(64, 64, acf=acf.tanh, init_type='he'),
    k.DenseLayer(64, 1, acf=acf.idx, init_type='he')  # Lineare Ausgabe für binäre Klassifikation
]

mlp_adult = k.MLP(*layers, cost=acf.mse_cost)

# Training des neuronalen Netzes
k.train(mlp_adult, X_train.T, Y_train.T, epochs=1000, lr=0.01)


# Funktion zur Berechnung der Genauigkeit
def calculate_accuracy(y_true, y_pred):
    y_pred_rounded = anp.round(y_pred)
    accuracy = anp.mean(y_true == y_pred_rounded)
    return accuracy


# Evaluieren der Genauigkeit des neuronalen Netzes
def evaluate_accuracy(mlp, X, y):
    y_pred, _ = mlp.predict(X)
    accuracy = calculate_accuracy(y, y_pred)
    return accuracy


# Berechnung der Genauigkeit auf dem Trainings-, Validierungs- und Testdatensatz
train_accuracy = evaluate_accuracy(mlp_adult, X_train.T, Y_train.T)
valid_accuracy = evaluate_accuracy(mlp_adult, X_valid.T, Y_valid.T)
test_accuracy = evaluate_accuracy(mlp_adult, X_test.T, Y_test.T)

# Ausgabe der Ergebnisse
print(f'Train Accuracy: {train_accuracy}')
print(f'Validation Accuracy: {valid_accuracy}')
print(f'Test Accuracy: {test_accuracy}')
