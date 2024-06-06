import numpy as np
from utils import knn_utils_01 as k

# Angenommen, jede Form ist ein 5x5 Bild, flach gemacht zu einem 25-elementigen Vektor
plus = np.load("./shapes/plus.npy").flatten()
minus = np.load("./shapes/minus.npy").flatten()
times = np.load("./shapes/times.npy").flatten()
divide = np.load("./shapes/divide.npy").flatten()
ball = np.load("./shapes/ball.npy").flatten()



# Vorbereiten der Trainingsdaten als 2D-Array
xtrn = np.vstack((plus, minus, times, divide, ball))  # (5, 25)
print(xtrn)
xtrn = np.tile(xtrn, (2, 1))  # (10, 25)

# Trainingslabels
y_plus = [1., 0., 0., 0., 0.]
y_minus = [0., 1., 0., 0., 0.]
y_times = [0., 0., 1., 0., 0.]
y_divide = [0., 0., 0., 1., 0.]
y_ball = [0., 0., 0., 0., 1.]

# Vorbereiten der Trainingslabels
ytrn = np.vstack((y_plus, y_minus, y_times, y_divide, y_ball))  # (5, 5)
ytrn = np.tile(ytrn, (2, 1))  # (10, 5)

# Initialisierung der Gewichte (25,5)
w = np.random.rand(25, 5)

# Initialisierung des Bias
b = np.zeros(5)

# Definieren der Lernrate und der Epochen
learning_rate = 0.1
epochs = 500

# Training des Perceptrons
w, b = k.train_layer(xtrn, ytrn, w, learning_rate, b, k.sigmoid, epochs)

# Vorhersagen mit dem trainierten Perceptron machen
k.predict(xtrn, w, b, k.sigmoid)
