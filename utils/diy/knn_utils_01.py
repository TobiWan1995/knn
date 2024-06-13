import numpy as np
from utils.diy import acf as kh
import autograd.numpy as anp
from autograd import grad


def forward_layer(x, w, b, activation_function):
    z = anp.dot(x, w) + b
    return activation_function(z)


def train_layer(X, Y, w, n, b, activation_function, epochs):
    # Gradient der Kostenfunktion berechnen
    cost_function_gradient = grad(kh.mse_cost)

    for epoch in range(epochs):
        # Indizes der Trainingsdaten mischen, um Stichprobenunabhängigkeit zu gewährleisten
        indices = np.arange(len(X))
        np.random.shuffle(indices)
        X = X[indices]
        Y = Y[indices]

        # Online-Modus: Aktualisierung nach jedem Beispiel
        for i in range(len(X)):
            x = np.array(X[i], dtype=np.float64)
            y = np.array(Y[i], dtype=np.float64)

            # Vorwärtsdurchlauf
            o = forward_layer(x, w, b, activation_function)

            # Gradienten berechnen
            cost_grad_wrt_output = cost_function_gradient(o, y)
            dw = np.outer(x, cost_grad_wrt_output)
            db = cost_grad_wrt_output

            # Gewichte und Bias aktualisieren
            w -= n * dw
            b -= n * db

    return w, b


def predict_perceptron(X, w, b, activation_function):
    predictions = []

    # Verarbeiten aller Eingaben in einem Durchlauf
    O = forward_layer(np.array(X), w, b, activation_function)

    # Ausgabe der Vorhersagen
    print("-----------------Predictions-----------------")
    for i, (x, o) in enumerate(zip(X, O)):
        print(f"Pattern {i}: {o}")

    return predictions


