import numpy as np
from autograd import grad, elementwise_grad
import autograd.numpy as anp
from utils import knn_helper as kh


# Definition des neuronalen Netzes
class DenseLayer:
    def __init__(self, input_dim, output_dim, acf=kh.relu):
        # HE-Initialisierung
        self.w = anp.random.randn(output_dim, input_dim) * anp.sqrt(2. / input_dim)
        self.b = anp.zeros((output_dim, 1))
        self.acf = acf
        self.acf_prime = elementwise_grad(acf)  # Berechnung der Ableitung der Aktivierungsfunktion

    def forward(self, x):
        acf_in = anp.dot(self.w, x) + self.b
        acf_out = self.acf(acf_in)
        return acf_in, acf_out

    def __call__(self, x):
        return self.forward(x)


class MLP:
    def __init__(self, *layers, cost=kh.mse_cost):
        self.layers = layers
        self.cost = cost

    def predict(self, x):
        outputs = []
        for layer in self.layers:
            acf_in, acf_out = layer.forward(x)
            outputs.append((acf_in, acf_out))
            x = acf_out
        return x, outputs

    def __call__(self, o, y):
        return self.cost(o, y)


def batch_generator(x, y, batch_size):
    if batch_size is None:
        yield x, y
    else:
        num_batches = int(anp.ceil(x.shape[1] / batch_size))  # Anzahl der Batches basierend auf der Anzahl der Muster
        indices = anp.arange(x.shape[1])  # Indizes basierend auf der Anzahl der Muster
        anp.random.shuffle(indices)  # Zufällige Permutation der Indizes

        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, x.shape[1])  # Endindex begrenzen, um Überlauf zu vermeiden
            batch_indices = indices[start_idx:end_idx]
            yield x[:, batch_indices], y[:, batch_indices]  # Transponierte Daten verwenden, um Musterweise zu arbeiten


def update_params(layer, delta, layer_acf_out, lr):
    grad_w = anp.dot(delta, layer_acf_out.T)  # Gradienten für Gewichte berechnen
    grad_b = anp.sum(delta, axis=1, keepdims=True)  # Gradienten für Bias berechnen
    layer.w -= lr * grad_w  # Aktualisieren der Gewichte
    layer.b -= lr * grad_b  # Aktualisieren des Bias


# Definition der Trainingsfunktion mit Batch-Verarbeitung
def train(mlp, x, y, epochs=1, lr=0.001, batch_size=None):
    def loss_fn(o, y):
        return mlp(o, y)

    # Berechnung des Gradienten der Verlustfunktion
    gradient_fn = grad(loss_fn, argnum=0)

    for epoch in range(epochs):
        for batch_index, (x_batch, y_batch) in enumerate(batch_generator(x, y, batch_size)):
            output, layer_outputs = mlp.predict(x_batch)  # Forward-Pass und Speichern der Zwischenergebnisse
            gradients_last = gradient_fn(output, y_batch)  # Berechnen des Gradienten für den Output der letzten Schicht

            delta = gradients_last  # Initialisieren von delta mit dem Gradienten der letzten Schicht

            # Backpropagation durch die Schichten
            for layer_index in reversed(range(len(mlp.layers))):
                layer = mlp.layers[layer_index]
                layer_acf_in, layer_acf_out = layer_outputs[layer_index - 1]  # Aktivierungen der vorherigen Schicht

                if layer_index > 0:
                    # Alte Gewichte vor Aktualisierung zwischenspeichern (neues Delta)
                    layer_w_old = layer.w
                    # Alte Gewichte aktualisieren
                    update_params(layer, delta, layer_acf_out, lr)
                    # Berechnung der Gradienten der vorherigen Schicht
                    delta = anp.dot(layer_w_old.T, delta) * layer.acf_prime(layer_acf_in)
                else:
                    # Wenn es die erste Schicht ist, verwende die Eingabe
                    update_params(layer, delta, x_batch, lr)

            current_loss = mlp(output, y_batch)

            # Loss nur in der ersten Epoche und dann alle 10 Epochen ausgeben
            if (epoch == 0 or (epoch + 1) % 100 == 0) and (batch_size is None or batch_index == 1):
                print(f"Epoch {epoch + 1}: loss = {current_loss}")


def accuracy(predicted, actual):
    # Bestimme die Klasse mit der höchsten Wahrscheinlichkeit für jede Vorhersage
    predicted_classes = np.argmax(predicted, axis=0)
    # Extrahiere die tatsächlichen Klassen aus den One-Hot-Encoded Labels
    actual_classes = np.argmax(actual, axis=0)
    # Berechne die Genauigkeit
    accuracy = np.mean(predicted_classes == actual_classes)
    return accuracy


'''
# Beispiel zur Verwendung des MLP
inputs, outputs = 10, 10 # Zeilen, Spalten (Gewichtsmatrix) oder Ausgans, Eingang (Schicht, Dense)

# Wenn input 10 und output 20 wird eine rand(out, in) Gewichtsmatrix erstellt
layer1 = DenseLayer(inputs, 20)  # 10 Eingangsneuronen, 20 Ausgangsneuronen
layer2 = DenseLayer(20,  outputs)  # 20 Eingangsneuronen, 10 Ausgangsneuron
mlp = MLP(layer1, layer2)

# Zeile (Ein Eingabewert jedes Patterns) und Spalte (Daten eines Patterns (Anzahl Patterns))
# Vektor sodass w (horizontal = 1 Gewicht aller Neuronen) * x (vertikal = 1 Pattern)
x = anp.random.randn(inputs, 1)

# w vertikal sind alle Gewichte eines Neurons!

# Die Ergebnis-Matrix hat dann Zeilen von W und Spalten von X und jede Spalte = Output p. Pattern
print(layer1(x))

y = anp.random.rand(outputs, 1) # Vertikaler Zielvektor v. 2 Pattern

# Ergibt Loss der letzten Schicht
print(mlp(x, y))

# Trainieren des MLP
train(mlp, x, y, epochs=1000, lr=0.001)
'''

'''
# XOR-MLP
xtrn = np.array([[0., 0., 1., 1.],
                 [0., 1., 0., 1.]])
ytrn = np.array([0., 1., 1., 0.])

xor = MLP(DenseLayer(2, 20), DenseLayer(20, 1))

train(xor, xtrn, ytrn, epochs=1000, lr=0.01)

print(xor.predict(xtrn)[0])
'''