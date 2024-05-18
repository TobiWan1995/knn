import numpy as np
from utils import knn_helper as kh
import autograd.numpy as anp
from autograd import grad


def forward(x, w, b, activation_function):
    z = anp.dot(x, w) + b
    return activation_function(z)


def train_layer(X, Y, w, n, b, activation_function, epochs, mode='online', batch_size=1):
    # Gradient der Kostenfunktion berechnen
    cost_function_gradient = grad(kh.mse_cost)

    # Modus-Parameter prüfen
    if mode not in ['online', 'batch']:
        raise ValueError("Mode muss 'online' oder 'batch' sein")

    for epoch in range(epochs):
        # Indizes der Trainingsdaten mischen, um Stichprobenunabhängigkeit zu gewährleisten
        indices = np.arange(len(X))
        np.random.shuffle(indices)
        X = X[indices]
        Y = Y[indices]

        # Verarbeitung in Batches für den Batch-Modus
        if mode == 'batch':
            for start_idx in range(0, len(X), batch_size):
                end_idx = min(start_idx + batch_size, len(X))
                x_batch = np.array(X[start_idx:end_idx], dtype=np.float64)
                y_batch = np.array(Y[start_idx:end_idx], dtype=np.float64)

                # Vorwärtsdurchlauf für den ganzen Batch
                o_batch = forward(x_batch, w, b, activation_function)

                # Gradienten für den ganzen Batch berechnen
                cost_grad_wrt_output_batch = cost_function_gradient(o_batch, y_batch)
                dw = np.dot(x_batch.T, cost_grad_wrt_output_batch) / batch_size
                db = np.mean(cost_grad_wrt_output_batch, axis=0)

                # Gewichte und Bias aktualisieren
                w -= n * dw
                b -= n * db
        else:
            # Online-Modus: Aktualisierung nach jedem Beispiel
            for i in range(len(X)):
                x = np.array(X[i], dtype=np.float64)
                y = np.array(Y[i], dtype=np.float64)

                # Vorwärtsdurchlauf
                o = forward(x, w, b, activation_function)

                # Gradienten berechnen
                cost_grad_wrt_output = cost_function_gradient(o, y)
                dw = np.outer(x, cost_grad_wrt_output)
                db = cost_grad_wrt_output

                # Gewichte und Bias aktualisieren
                w -= n * dw
                b -= n * db

    return w, b


def predict(X, w, b, activation_function):
    predictions = []

    # Verarbeiten aller Eingaben in einem Durchlauf
    O = forward(np.array(X), w, b, activation_function)

    # Ausgabe der Vorhersagen
    print("-----------------Predictions-----------------")
    for i, (x, o) in enumerate(zip(X, O)):
        print(f"Pattern {i}: {o}")

    return predictions


