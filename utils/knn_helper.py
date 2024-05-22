import autograd.numpy as anp # need to use autograd numpy to use grad

'''Aktivierungsfunktionen'''
def idx(x):
    return x

def relu(x):
    return anp.maximum(0, x)

def sigmoid(x):
    return 1 / (1 + anp.exp(-x))

def softmax(x):
    exp_x = anp.exp(x - anp.max(x, axis=0)) # subtract max to improve numerical stability
    return exp_x / anp.sum(exp_x, axis=0)

def tanh(x):
    return anp.tanh(x)

def tanh_prime(x):
    return 1 - anp.tanh(x)**2

'''Kostenfunktionen - Loss-Funktionen'''
def mse_cost(predicted, actual):
    return anp.mean((predicted - actual) ** 2)


def bce(o, t):
    # Clippen der Ausgabe, um numerische Instabilitäten zu verhindern:
    # Die Werte von o werden so beschränkt, dass sie nie ganz 0 oder ganz 1 sind,
    # um Division durch Null oder Logarithmus von 0 zu vermeiden.
    o = anp.clip(o, 1e-7, 1 - 1e-7)

    # Berechnung des binären Kreuzentropie-Loss
    # t * anp.log(o): Logarithmische Verlustbeiträge für die vorhergesagten positiven Klassen
    # (1 - t) * anp.log(1 - o): Logarithmische Verlustbeiträge für die vorhergesagten negativen Klassen
    return -anp.mean(t * anp.log(o) + (1 - t) * anp.log(1 - o))


def nll(p, y):
    # Multipliziere jede Vorhersage p mit dem entsprechenden One-Hot-kodierten Label y
    # Das Ergebnis ist ein Array, das an jeder Stelle, wo y '1' ist, den Wert von p hat, und sonst '0'
    masked_probabilities = p * y
    # Berechne den Logarithmus der gemaskten Wahrscheinlichkeiten, wobei kleine Werte durch einen kleinen positiven
    # Wert ersetzt werden, um numerische Probleme zu vermeiden
    log_probabilities = anp.log(masked_probabilities + 1e-9)
    # Summiere alle Logarithmen der Wahrscheinlichkeiten (der Wert ist nur an den '1' Stellen von y ungleich Null)
    sum_log_probabilities = anp.sum(log_probabilities)
    # Die Funktion soll den negativen Durchschnitt über alle Beispiele zurückgeben
    return -sum_log_probabilities / len(y.T)
