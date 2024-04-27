from math import exp
from numpy import *


def relu(x):
    return max(0, x)


def sigmoid(x):
    return 1 / (1 + exp(-x / 1))


def forward(x, w, b):
    # return sigmoid(x[0] * w[0] + x[1] * w[1] + b)
    return relu(x[0] * w[0] + x[1] * w[1] + b)


def predict(X, w, b):
    for i, item in enumerate(X):
        o = forward(X[i], w, b)
        print("Pattern ", i, ": ", o)
    print("Bias: ", b)
    return w


def train(X, Y, w, n, b):
    dw = [0.0, 0.0]
    loss = 0.0
    for i, item in enumerate(X):
        x = X[i]
        y = Y[i]

        o = forward(x, w, b)
        delta = y - o
        loss += delta * delta

        dw[0] = n * delta * x[0]
        dw[1] = n * delta * x[1]

        db = n * delta * b

        w += dw
        b += db

    return w, b


xtrn = [[0, 0], [0, 1], [1, 0], [1, 1]]
N = 0.1  # lernrate

## and-net
print("----------and-net---------")

ytrn = [0, 0, 0, 1]
W = random.rand(2)
bias = 0.5

for m in range(1, 50):
    W, bias = train(xtrn, ytrn, W, N, bias)

predict(xtrn, W, bias)

## not-net
print("----------not-net---------")

ytrn = [1, 0, 0, 0]
W = random.rand(2)
bias = 0.2

for m in range(1, 50):
    W, bias = train(xtrn, ytrn, W, N, bias)

predict(xtrn, W, bias)