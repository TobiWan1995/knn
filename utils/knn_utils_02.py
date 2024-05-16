import numpy as np
from autograd import grad, elementwise_grad
import autograd.numpy as anp


# Definition der Aktivierungsfunktion
def idx(x):
    return x


def relu(x):
    return np.maximum(x, 0)

def sigmoid(x):
    return 1 / (1 + anp.exp(-x / 0.1))


# Definition der Kostenfunktion
def mse_cost(predicted, actual):
    return anp.mean((predicted - actual) ** 2)


def nll(predicted, actual):
    """
    Berechnet die Negative Log-Likelihood zwischen den vorhergesagten Wahrscheinlichkeiten und den tatsächlichen Labels.

    :param predicted: Vorhergesagte Wahrscheinlichkeiten (Softmax-Ausgabe der MLP) [n_classes, n_samples]
    :param actual: Tatsächliche Labels in One-Hot-Encoded Form [n_classes, n_samples]
    :return: Negative Log-Likelihood Kosten
    """
    # Vermeidung von Log(0) durch Clippen der vorhergesagten Werte
    epsilon = 1e-10
    predicted = anp.clip(predicted, epsilon, 1 - epsilon)
    # Berechnung der Negative Log-Likelihood
    return -anp.sum(actual * anp.log(predicted))


# Definition der DenseLayer Klasse
class DenseLayer:
    def __init__(self, input_dim, output_dim, acf=relu):
        self.w = anp.random.randn(output_dim, input_dim) * anp.sqrt(2. / input_dim)
        self.b = anp.zeros((output_dim, 1))
        self.acf = acf
        self.acf_prime = elementwise_grad(acf)  # Berechnung der Ableitung der Aktivierungsfunktion

    def forward(self, x):
        z = anp.dot(self.w, x) + self.b
        a = self.acf(z)
        return z, a

    def __call__(self, x):
        _, a = self.forward(x)
        return a


class MLP:
    def __init__(self, *layers, cost=mse_cost):
        self.layers = layers
        self.cost = cost

    def predict(self, x):
        outputs = []
        for layer in self.layers:
            z, a = layer.forward(x)
            outputs.append((z, a))
            x = a
        return x, outputs

    def __call__(self, o, y):
        return self.cost(o, y)


# Beispiel zur Verwendung des MLP
# inputs, outputs = 10, 10 # Zeilen, Spalten (Gewichtsmatrix) oder Ausgans, Eingang (Schicht, Dense)

# Wenn input 10 und output 20 wird eine rand(out, in) Gewichtsmatrix erstellt
# layer1 = DenseLayer(inputs, 20)  # 10 Eingangsneuronen, 20 Ausgangsneuronen
# layer2 = DenseLayer(20,  outputs)  # 20 Eingangsneuronen, 10 Ausgangsneuron
# mlp = MLP(layer1, layer2)

# Zeile (Ein Eingabewert jedes Patterns) und Spalte (Daten eines Patterns (Anzahl Patterns))
# Vektor sodass w (horizontal = 1 Gewicht aller Neuronen) * x (vertikal = 1 Pattern)
# x = anp.random.randn(inputs, 1)

# w vertikal ist dann alle Gewichte eines Neurons!

# Die Ergebnis-Matrix hat dann Zeilen von W und Spalten von X und jede Spalte = Output p. Pattern
# print(layer1(x))

# y = anp.random.rand(outputs, 1) # Vertikaler Zielvektor v. 2 Pattern

# Ergibt Loss der letzten Schicht
# print(mlp(x, y))

def train(mlp, x, y, epochs=1, lr=0.001):
    def loss_fn(o, y):
        return mlp(o, y)  # ruft mlp.cost(o, y) auf

    # Berechnung des Gradienten der Verlustfunktion
    gradient_fn = grad(loss_fn, argnum=0)

    for epoch in range(epochs):
        output, layer_outputs = mlp.predict(x)  # Durchführen des Forward-Passes und Speichern der Zwischenergebnisse
        gradients_last = gradient_fn(output, y)  # Berechnen des Gradienten für den Output der letzten Schicht

        delta = gradients_last  # Initialisieren von delta mit dem Gradienten der letzten Schicht

        # Rückwärtspropagation durch die Schichten
        for i in reversed(range(len(mlp.layers))):
            layer = mlp.layers[i]
            z, a = layer_outputs[i]  # Zwischengespeicherte lineare Summe und Aktivierungen der aktuellen Schicht

            if i > 0:
                _, prev_a = layer_outputs[i - 1]  # Aktivierungen der vorherigen Schicht
            else:
                prev_a = x  # Wenn es die erste Schicht ist, verwende die Eingabe

            grad_w = np.dot(delta, prev_a.T)  # Gradienten für Gewichte berechnen
            grad_b = np.sum(delta, axis=1, keepdims=True)  # Gradienten für Bias berechnen
            layer.w -= lr * grad_w  # Aktualisieren der Gewichte
            layer.b -= lr * grad_b  # Aktualisieren des Bias

            if i > 0:  # Wenn nicht die erste Schicht, berechne neuen delta für die vorherige Schicht
                # Modifikation von delta durch die Ableitung der Aktivierungsfunktion
                delta = np.dot(layer.w.T, delta) * layer.acf_prime(layer_outputs[i - 1][0])

        current_loss = mlp.cost(output, y)
        print(f"Epoch {epoch + 1}: loss = {current_loss}")


# Trainieren des MLP
# train(mlp, x, y, epochs=1000, lr=0.001)

"""
# XOR-MLP
xtrn = np.array([[0., 0., 1., 1.],
                 [0., 1., 0., 1.]])
ytrn = np.array([0., 1., 1., 0.])

xor = MLP(DenseLayer(2, 20), DenseLayer(20, 1))

train(xor, xtrn, ytrn, epochs=1000, lr=0.001)

print(xor.predict(xtrn)[0])
"""