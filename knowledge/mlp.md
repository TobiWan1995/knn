# MLP (Multilayer Perceptron)

## Eingabe (Input):

Das MLP nimmt Eingabedaten auf, die typischerweise als Vektoren oder Matrizen (bei Batches) organisiert sind.

## Forward Pass:

Die Eingabedaten werden durch das Netzwerk geleitet, wobei jede Schicht lineare Transformationen (Gewichtsmultiplikation und Bias-Addition) und nichtlineare Aktivierungen (z.B. ReLU, Sigmoid) durchführt.

## Berechnung des Outputs:

Der endgültige Output des Netzwerks wird basierend auf den Gewichten und Aktivierungen der letzten Schicht berechnet. Dieser Output repräsentiert die Vorhersage des Netzwerks.

## Loss-Funktion:

Die Vorhersagen werden mit den tatsächlichen Zielwerten verglichen, um den Fehler zu messen. Eine Loss-Funktion (z.B. Mean Squared Error für Regression, Cross-Entropy für Klassifikation) quantifiziert, wie gut die Vorhersagen des Netzwerks mit den tatsächlichen Werten übereinstimmen.

## Backward Pass (Backpropagation):

Der Fehler wird vom Output zurück durch das Netzwerk propagiert, um zu bestimmen, wie sehr jedes Gewicht zum Gesamtfehler beigetragen hat. Dies geschieht durch die Berechnung der Gradienten der Loss-Funktion bezüglich jedes Gewichts und jedes Bias.

## Aktualisierung der Gewichte:

Die Gewichte und Biases werden aktualisiert, um den Fehler zu minimieren. Dies geschieht typischerweise mit einem Optimierungsverfahren wie dem Gradientenabstieg. Die Gewichte werden entsprechend der berechneten Gradienten und einer festgelegten Lernrate angepasst.

## Wiederholung:

Diese Schritte werden über mehrere Epochen hinweg wiederholt, wobei jede Epoche einen kompletten Durchlauf der Trainingsdaten durch das Netzwerk darstellt. Mit jeder Epoche sollte der Fehler verringert und die Vorhersagegenauigkeit verbessert werden.
