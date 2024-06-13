from autograd import grad, elementwise_grad
import autograd.numpy as anp
from utils.diy import acf as acf
import matplotlib.pyplot as plt


# Definition des neuronalen Netzes
class DenseLayer:
    def __init__(self, input_dim, output_dim, acf=acf.relu, init_type='he'):
        self.acf = acf
        self.acf_prime = elementwise_grad(self.acf)  # Berechnung der Ableitung der Aktivierungsfunktion
        self.b = anp.zeros((output_dim, 1))

        if init_type == 'he':
            # He-Initialisierung (auch Kaiming-Initialisierung genannt)
            self.w = anp.random.randn(output_dim, input_dim) * anp.sqrt(2. / input_dim)
        elif init_type == 'xavier':
            # Xavier-Initialisierung (auch Glorot-Initialisierung genannt)
            self.w = anp.random.randn(output_dim, input_dim) * anp.sqrt(1. / input_dim)
        else:
            raise ValueError("Unsupported initialization type. Choose 'he' or 'xavier'.")

    def forward(self, x):
        acf_in = anp.dot(self.w, x) + self.b
        acf_out = self.acf(acf_in)
        return acf_in, acf_out

    def __call__(self, x):
        return self.forward(x)


class MLP:
    def __init__(self, *layers, cost=acf.mse_cost):
        self.layers = layers
        self.cost = cost
        self.history = {
            'train_loss': [],
            'valid_loss': [],
            'train_accuracy': [],
            'valid_accuracy': []
        }

    def __call__(self, o, y):
        return self.cost(o, y)

    def predict(self, x):
        outputs = []
        for layer in self.layers:
            acf_in, acf_out = layer.forward(x)
            outputs.append((acf_in, acf_out))
            x = acf_out
        return x, outputs

    # Nur Binary Klassifikation (nicht Regression oder mult. Klassifikation)
    def accuracy(self, y_pred, y_true):
        y_pred_rounded = anp.round(y_pred)
        accuracy = anp.mean(y_true == y_pred_rounded)
        return accuracy


def train(mlp, x_train, y_train, epochs=1, lr=0.001, batch_size=None, x_valid=None, y_valid=None):
    def loss_fn(o, y):
        return mlp(o, y)

    # Berechnung des Gradienten der Verlustfunktion
    gradient_fn = grad(loss_fn, argnum=0)

    for epoch in range(epochs):
        # Training
        for batch_index, (x_batch, y_batch) in enumerate(batch_generator(x_train, y_train, batch_size)):
            output, layer_outputs = mlp.predict(x_batch)  # Forward-Pass und Speichern der Zwischenergebnisse
            gradients_last = gradient_fn(output, y_batch)  # Berechnen des Gradienten für den Output der letzten Schicht

            delta = gradients_last  # Initialisieren von delta mit dem Gradienten der letzten Schicht

            # Backpropagation durch die Schichten
            for layer_index in reversed(range(len(mlp.layers))):
                layer = mlp.layers[layer_index]
                if layer_index > 0:
                    layer_acf_in, layer_acf_out = layer_outputs[layer_index - 1]  # Aktivierungen der vorherigen Schicht
                    layer_w_old = layer.w  # Alte Gewichte vor Aktualisierung zwischenspeichern (neues Delta)
                    update_params(layer, delta, layer_acf_out, lr)  # Alte Gewichte aktualisieren
                    delta = anp.dot(layer_w_old.T, delta) * layer.acf_prime(
                        layer_acf_in)  # Berechnung der Gradienten der vorherigen Schicht
                else:
                    update_params(layer, delta, x_batch, lr)  # Wenn es die erste Schicht ist, verwende die Eingabe

        train_stats(mlp, x_train, y_train, epoch, x_valid=x_valid, y_valid=y_valid)

    plot_training_history(mlp.history)


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


def train_stats(mlp, x_train, y_train, epoch, x_valid=None, y_valid=None):
    # Verlust und Genauigkeit für die Trainingsdaten berechnen
    train_output, _ = mlp.predict(x_train)
    train_loss = mlp(train_output, y_train)
    train_accuracy = mlp.accuracy(train_output, y_train)

    mlp.history['train_loss'].append(train_loss)
    mlp.history['train_accuracy'].append(train_accuracy)

    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch + 1}: Train Loss = {train_loss}, Train Accuracy = {train_accuracy}")

    # Verlust und Genauigkeit für die Validierungsdaten berechnen
    if x_valid is not None and y_valid is not None:
        valid_output, _ = mlp.predict(x_valid)
        valid_loss = mlp(valid_output, y_valid)
        valid_accuracy = mlp.accuracy(valid_output, y_valid)

        mlp.history['valid_loss'].append(valid_loss)
        mlp.history['valid_accuracy'].append(valid_accuracy)

        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch + 1}: Valid Loss = {valid_loss}, Valid Accuracy = {valid_accuracy}")
    else:
        mlp.history['valid_loss'].append(0.0)
        mlp.history['valid_accuracy'].append(0.0)


def plot_training_history(history):
    epochs = range(1, len(history['train_loss']) + 1)

    # Plot für Trainings- und Validierungsverlust
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], label='Train Loss')
    plt.plot(epochs, history['valid_loss'], label='Valid Loss')
    plt.title('Train and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Plot für Trainings- und Validierungsgenauigkeit
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_accuracy'], label='Train Accuracy')
    plt.plot(epochs, history['valid_accuracy'], label='Valid Accuracy')
    plt.title('Train and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()


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
