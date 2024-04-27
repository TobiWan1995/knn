import numpy as np
import autograd.numpy as anp
from autograd import grad


def relu(x):
    return anp.maximum(0, x)


def sigmoid(x):
    return 1 / (1 + anp.exp(-x))


def forward(x, w, b, activation_function):
    z = anp.dot(x, w) + b
    return activation_function(z)


def mse_cost(predicted, actual):
    return anp.mean((predicted - actual) ** 2)


def train(X, Y, w, n, b, activation_function, epochs):
    # Berechnen der Ableitung der Kostenfunktion
    cost_function_gradient = grad(mse_cost)

    for epoch in range(epochs):
        for i in range(len(X)):
            x = np.array(X[i], dtype=np.float64)
            y = np.array(Y[i], dtype=np.float64)

            # Vorwärtsdurchlauf
            o = forward(x, w, b, activation_function)

            # Berechnen der Gradienten für w und b
            cost_grad_wrt_output = cost_function_gradient(o, y)
            dw = np.outer(x, cost_grad_wrt_output)
            db = cost_grad_wrt_output

            # Gewichte und Bias aktualisieren
            w -= n * dw
            b -= n * db

    return w, b


def predict(X, w, b, activation_function):
    print("Prediction results:")
    print("-------------------------------------------------")
    print("{:<20} | {:<20}".format("Input Pattern", "Predicted Output"))
    print("-------------------------------------------------")
    predictions = []
    for i, x in enumerate(X):
        o = forward(x, w, b, activation_function)
        predictions.append(o)
        print("{:<20} | {:<20}".format(str(x), str(o)))
    print("-------------------------------------------------")
    print("Bias: ", b)
    return predictions

# Die Verwendung würde dann so aussehen:
# w, b = train(X, Y, w, n, b, sigmoid, 100)  # Trainieren für 100 Epochen mit sigmoid
# predictions = predict(X, w, b, sigmoid)    # Vorhersagen machen mit sigmoid
