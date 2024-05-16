

Ein Bias-Neuron ist in neuronalen Netzen wichtig, weil es hilft, die Entscheidungsfunktion des Netzwerks flexibler zu gestalten und die Leistung zu verbessern. Hier sind die Hauptgründe, warum ein Bias-Neuron benötigt wird:

### Erhöhung der Modellflexibilität:

Ohne Bias wäre die Entscheidungsfunktion des Neurons stark eingeschränkt. Insbesondere könnten die Ausgabewerte der Neuronen nur entlang der Achse des Koordinatensystems verlaufen, was die Anpassungsfähigkeit des Modells verringert. Der Bias verschiebt die Entscheidungsgrenze, sodass sie nicht zwingend durch den Ursprung gehen muss, was die Modellflexibilität erhöht.

### Verbesserung der Lernfähigkeit:

Bias-Neuronen tragen dazu bei, dass das neuronale Netzwerk in der Lage ist, auch solche Muster zu lernen, die nicht linear durch den Ursprung verlaufen. Das bedeutet, dass das Netzwerk komplexere Zusammenhänge und Datenmuster besser erfassen und verarbeiten kann.

### Vermeidung von Null-Aktivierung:

Wenn die Gewichtungen aller Eingabeneuronen Null sind, kann das Neuron dennoch durch den Bias aktiviert werden. Dies verhindert, dass Neuronen inaktiv bleiben und dadurch das Lernen behindern.

### Mathematische Notwendigkeit:

In der linearen Algebra entspricht der Bias einem Verschiebungsterm in der Gleichung einer Hyperebene (z.B. der Geraden 
y=mx+b in 2D, wobei b der Bias ist). Ohne diesen Term könnte die Hyperebene nicht alle möglichen Positionen im Raum einnehmen.
Zusammengefasst ermöglicht der Bias, dass neuronale Netze flexibler und leistungsfähiger werden, indem sie komplexere und nicht-lineare Datenmuster besser modellieren können.

Bias wird in der Forward-Funktion in die Summe gerechnet!

```python
import numpy as np

def forward(x,w, b):
  return sigmoid(x[0] * w[0] + x[1] * w[1] + b)

def sigmoid(x):
    return 1 / (1 + np.exp(-x / 1))

def train(X,Y, w, η, b):
   Δw = [0, 0] 
   loss = 0.0
   for i, item in enumerate(X):
     x = X[i]
     y = Y[i]

     o = forward(x,w)
     δ = y - o
     loss += δ * δ

     Δw[0] =  η * δ * x[0]
     Δw[1] =  η * δ * x[1]

     Δb =  η * δ * 1

     w += Δw
     b += Δb

   return w,b
```