Bias wird in der Forward-Funktion in die Summe gerechnet!

# def forward(x,w, b):
#  return sigm(x[0] * w[0] + x[1] * w[1] + b)

Bias wird benoetigt damit einige Neuronen trainiert werden koennen.
Beispielsweise laesst sich ohne Bias kein OR Perceptron trainieren.

# def train(X,Y, w, η, b):
#   Δw = [0, 0]  # ,0
#   loss = 0.0
#   for i, item in enumerate(X):
#     x = X[i]
#     y = Y[i]
#
#     o = forward(x,w)
#     δ = y - o
#     loss += δ * δ
#
#     Δw[0] =  η * δ * x[0]
#     Δw[1] =  η * δ * x[1]
#
#     Δb =  η * δ * 1
#
#     w += Δw
#     b += Δb
#
#   return w,b
