import numpy as np
import matplotlib.pyplot as plt
from keras.src.datasets import mnist
from utils import knn_utils_02 as k

# MNIST-Daten laden (mnist dim 28x28 with 10 classes)
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Bilder normalisieren
x_train, x_test = x_train.reshape(-1, 784) / 255.0, x_test.reshape(-1, 784) / 255.0  # Flach und normalisiert

# One-Hot-Encoding der Labels
y_train_encoded, y_test_encoded = np.zeros((y_train.size, 10)), np.zeros((y_test.size, 10))
y_train_encoded[np.arange(y_train.size), y_train] = 1
y_test_encoded[np.arange(y_test.size), y_test] = 1

# Durchschnittsbilder für jede Klasse visualisieren
fig, axes = plt.subplots(1, 10, figsize=(10, 5))
for i in range(10):
    mean_img = np.mean(x_train[y_train == i], axis=0)
    axes[i].imshow(mean_img.reshape(28, 28), cmap='gray')
    axes[i].set_title(str(i))
    axes[i].axis('off')
# plt.show()

# Netzwerk initialisieren (Normal)
layers = [
    k.DenseLayer(784, 256, acf=k.sigmoid),
    k.DenseLayer(256, 256, acf=k.sigmoid),
    k.DenseLayer(256, 10, acf=k.sigmoid)
]

mlp = k.MLP(*layers)

# Training des Netzwerks
k.train(mlp, x_train.T, y_train_encoded.T, epochs=100, lr=0.01)

# Netzwerk initialisieren (klein)
layers_small = [
    k.DenseLayer(784, 128, acf=k.sigmoid),
    k.DenseLayer(128, 64, acf=k.sigmoid),
    k.DenseLayer(64, 10, acf=k.sigmoid)
]

mlp_small = k.MLP(*layers_small, cost=k.nll)

# Training des Netzwerks
# k.train(mlp_small, x_train.T, y_train_encoded.T, epochs=10, lr=0.000001)

# Netzwerk initialisieren (winzig)
layers_smaller = [
    k.DenseLayer(784, 64, acf=k.sigmoid),
    k.DenseLayer(64, 10, acf=k.sigmoid)
]

mlp_smaller = k.MLP(*layers_smaller)

# Training des Netzwerks
# k.train(mlp_smaller, x_train.T, y_train_encoded.T, epochs=100, lr=0.01)

def accuracy(predicted, actual):
    """
    Berechnet die Genauigkeit eines Klassifizierungsmodells.

    :param predicted: Vorhergesagte Wahrscheinlichkeiten [n_classes, n_samples]
    :param actual: Tatsächliche Labels in One-Hot-Encoded Form [n_classes, n_samples]
    :return: Accuracy als Skalar
    """
    # Bestimme die Klasse mit der höchsten Wahrscheinlichkeit für jede Vorhersage
    predicted_classes = np.argmax(predicted, axis=0)
    # Extrahiere die tatsächlichen Klassen aus den One-Hot-Encoded Labels
    actual_classes = np.argmax(actual, axis=0)
    # Berechne die Genauigkeit
    accuracy = np.mean(predicted_classes == actual_classes)
    return accuracy


# Angenommen, deine 'MLP' Klasse hat eine Methode namens 'predict', die die Wahrscheinlichkeiten zurückgibt
predicted = mlp.predict(x_test.T)  # Stelle sicher, dass x_test die richtige Form hat
accuracy_score = accuracy(predicted[0], y_test_encoded.T)  # y_test_encoded sollte auch die korrekte Form haben
print("Accuracy:", accuracy_score)
