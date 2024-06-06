from ucimlrepo import fetch_ucirepo
from utils import acf as kh
from utils import knn_utils_02 as k
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import autograd.numpy as anp

'''Aufgabe 1'''
# fetch dataset
wine_quality = fetch_ucirepo(id=186)

# data (as pandas dataframes)
X = wine_quality.data.original
Y = wine_quality.data.targets

# metadata
print(wine_quality.metadata)

# variable information
print(wine_quality.variables)

'''Aufgabe 2'''


def filter_wine_color(x, y, color):
    # Filtern der Weißweine basierend auf der Spalte 'color'
    if 'color' in X.columns:
        x_c = x[x['color'] == color]
        y_c = y[x['color'] == color]
    else:
        raise ValueError("Es gibt keine Spalte 'color', um Weißweine zu filtern")

    # Entfernen Sie die Spalte 'color' aus den Features
    x_c = x_c.drop(columns=['color'])
    x_c = x_c.drop(columns=['quality'])
    return x_c, y_c


def normalize_wine_data(x, y):
    # Normalisierung der Features
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    return x_scaled, y.to_numpy()


X_white, Y_white = filter_wine_color(X, Y, 'white')

X_white_norm, Y_white_norm = normalize_wine_data(X_white, Y_white)

# Aufteilen in Trainings- und Testdaten
X_white_train, X_white_test, Y_white_train, Y_white_test = train_test_split(X_white_norm, Y_white_norm, test_size=0.2,
                                                                            random_state=42)
''''Aufgabe 3'''

# Netzwerk initialisieren
layers = [
    k.DenseLayer(11, 64, acf=kh.sigmoid),
    k.DenseLayer(64, 64, acf=kh.sigmoid),
    k.DenseLayer(64, 1, acf=kh.idx)  # Lineare Ausgabe
]

mlp_white = k.MLP(*layers, cost=kh.mse_cost)

k.train(mlp_white, X_white_train.T, Y_white_train.T, epochs=1000, lr=0.01)

'''1. Wie gut kann Ihr neuronales Netzwerk die Qualität von Weißweinen bestimmen?'''


def calculate_accuracy(y_true, y_pred):
    y_pred_rounded = anp.round(y_pred)
    accuracy = anp.mean(y_true == y_pred_rounded)
    return accuracy


def evaluate_accuracy(mlp, X, y):
    y_pred, _ = mlp.predict(X)
    accuracy = calculate_accuracy(y, y_pred)
    return accuracy


# Evaluieren der Accuracy auf dem Trainings- und Testdatensatz für Weißweine
train_accuracy_white = evaluate_accuracy(mlp_white, X_white_train.T, Y_white_train.T)
test_accuracy_white = evaluate_accuracy(mlp_white, X_white_test.T, Y_white_test.T)

print(f'Train Accuracy white: {train_accuracy_white}')
print(f'Test Accuracy white: {test_accuracy_white}')

'''2. Kann Ihr neuronales Netzwerk ohne weiteres Training auch die Qualität von Rotweinen bestimmen?'''

X_red, Y_red = filter_wine_color(X, Y, 'red')
X_red_norm, Y_red_norm = normalize_wine_data(X_red, Y_red)
X_red_train, X_red_test, Y_red_train, Y_red_test = train_test_split(X_red_norm, Y_red_norm, test_size=0.2,
                                                                            random_state=42)

# Evaluieren der Accuracy auf dem Testdatensatz für Rotweine
test_accuracy_red = evaluate_accuracy(mlp_white, X_red_test.T, Y_red_test.T)

print(f'Test Accuracy white on red (without training): {test_accuracy_red}')

'''Wie viele Trainingsschritte sind notwendig, um Rotweine ebenso gut
erkennen zu können wie Weißweine?'''

mlp_red = k.MLP(*layers, cost=kh.mse_cost)

k.train(mlp_red, X_red_train.T, Y_red_train.T, epochs=1000, lr=0.01)

train_accuracy_red = evaluate_accuracy(mlp_red, X_red_train.T, Y_red_train.T)
test_accuracy_red = evaluate_accuracy(mlp_red, X_red_test.T, Y_red_test.T)

print(f'Train Accuracy red: {train_accuracy_red}')
print(f'Test Accuracy red: {test_accuracy_red}')

'''4. Kann es hilfreich sein, dem MLP für Weißweine einige Rotweinbeispiele
zu zeigen (training), um eine gute Leistung für Rotweine zu erzielen?'''

k.train(mlp_white, X_red_train.T, Y_red_train.T, epochs=10, lr=0.01)

train_accuracy_white_red = evaluate_accuracy(mlp_white, X_red_train.T, Y_red_train.T)
test_accuracy_white_red = evaluate_accuracy(mlp_white, X_red_test.T, Y_red_test.T)

print(f'Train Accuracy white on red (with training): {train_accuracy_white_red}')
print(f'Test Accuracy white on red (with training): {test_accuracy_white_red}')