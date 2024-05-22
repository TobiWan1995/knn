import numpy as np
import matplotlib.pyplot as plt
from keras.src.datasets import mnist
from utils import knn_utils_02 as k
from utils import  knn_helper as kh

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
    k.DenseLayer(784, 256, acf=kh.relu),
    k.DenseLayer(256, 256, acf=kh.relu),
    k.DenseLayer(256, 256, acf=kh.relu),
    k.DenseLayer(256, 10, acf=kh.softmax)
]

mlp = k.MLP(*layers, cost=kh.nll)

# Training des Netzwerks
# k.train(mlp, x_train.T, y_train_encoded.T, epochs=1000, lr=0.01)

# Netzwerk initialisieren (klein)
layers_small = [
    k.DenseLayer(784, 128, acf=kh.sigmoid),
    k.DenseLayer(128, 64, acf=kh.sigmoid),
    k.DenseLayer(64, 10, acf=kh.sigmoid)
]

mlp_small = k.MLP(*layers_small)

# Training des Netzwerks
# k.train(mlp_small, x_train.T, y_train_encoded.T, epochs=50, lr=0.00001, batch_size=50)

# Netzwerk initialisieren (winzig)
layers_smaller = [
    k.DenseLayer(784, 64, acf=kh.relu),
    k.DenseLayer(64, 10, acf=kh.relu)
]

mlp_smaller = k.MLP(*layers_smaller, cost=kh.bce)

# Training des Netzwerks
k.train(mlp_smaller, x_train.T, y_train_encoded.T, epochs=200, lr=0.01, batch_size=40000)

predicted = mlp_smaller.predict(x_test.T)
accuracy_score = k.accuracy(predicted[0], y_test_encoded.T)
print("Accuracy:", accuracy_score)
