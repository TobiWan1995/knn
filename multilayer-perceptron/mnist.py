import numpy as np
import matplotlib.pyplot as plt
from keras.src.datasets import mnist
from utils import knn_utils_02 as k
from utils import acf as acf

# MNIST-Daten laden (mnist dim 28x28 with 10 classes)
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Bilder normalisieren
x_train, x_test = x_train.reshape(-1, 784) / 255.0, x_test.reshape(-1, 784) / 255.0  # Flach und normalisiert

# One-Hot-Encoding der Labels
y_train_encoded, y_test_encoded = np.zeros((y_train.size, 10)), np.zeros((y_test.size, 10))
y_train_encoded[np.arange(y_train.size), y_train] = 1
y_test_encoded[np.arange(y_test.size), y_test] = 1

# Berechnung der Anzahl der Reihen (4 Reihen, da 12/3 = 4)
rows = 4

# Durchschnittsbilder für jede Klasse visualisieren
fig, axes = plt.subplots(rows, 3, figsize=(12, 12))
for i in range(12):  # Gehe von 0 bis 11, für 12 Bilder
    row = i // 3  # Bestimmt die Reihe
    col = i % 3   # Bestimmt die Spalte
    if i < 10:
        mean_img = np.mean(x_train[y_train == i], axis=0)
        ax = axes[row, col]
        ax.imshow(mean_img.reshape(28, 28), cmap='gray')
        ax.set_title(str(i))
    else:
        # wenn keine zusätzlichen Klassenbilder vorhanden sind
        ax = axes[row, col]
        ax.imshow(np.zeros((28, 28)), cmap='gray')  # Zeige ein leeres Bild
        ax.set_title("Empty")
    ax.axis('off')

plt.subplots_adjust(hspace=0.1, wspace=0.1)
plt.tight_layout(pad=2.0)

# plt.show()

# Netzwerk initialisieren (Normal)
layers = [
    k.DenseLayer(784, 256, acf=acf.relu),
    k.DenseLayer(256, 256, acf=acf.relu),
    k.DenseLayer(256, 10, acf=acf.relu)
]

mlp = k.MLP(*layers, cost=acf.bce)

# Training des Netzwerks
# k.train(mlp, x_train.T, y_train_encoded.T, epochs=1000, lr=0.001)

predicted = mlp.predict(x_test.T)
accuracy_score = k.accuracy(predicted[0], y_test_encoded.T)
print("Accuracy:", accuracy_score)

# Netzwerk initialisieren (big)
layers_big = [
    k.DenseLayer(784, 512, acf=acf.relu),
    k.DenseLayer(512, 256, acf=acf.relu),
    k.DenseLayer(256, 128, acf=acf.relu),
    k.DenseLayer(128, 64, acf=acf.relu),
    k.DenseLayer(64, 10, acf=acf.relu)
]

mlp_big = k.MLP(*layers_big, cost=acf.bce)

# Training des Netzwerks
k.train(mlp_big, x_train.T, y_train_encoded.T, epochs=1000, lr=0.001)

predicted = mlp_big.predict(x_test.T)
accuracy_score = k.accuracy(predicted[0], y_test_encoded.T)
print("Accuracy:", accuracy_score)

# Netzwerk initialisieren (winzig)
layers_smaller = [
    k.DenseLayer(784, 64, acf=acf.relu),
    k.DenseLayer(64, 10, acf=acf.relu)
]

mlp_smaller = k.MLP(*layers_smaller, cost=acf.bce)

# Training des Netzwerks
# k.train(mlp_smaller, x_train.T, y_train_encoded.T, epochs=1000, lr=0.01)

predicted = mlp_smaller.predict(x_test.T)
accuracy_score = k.accuracy(predicted[0], y_test_encoded.T)
print("Accuracy:", accuracy_score)
