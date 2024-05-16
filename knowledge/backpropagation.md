# Ableitung der Loss-Funktion

Zunächst leiten Sie die Loss-Funktion bezüglich des Outputs der letzten Schicht ab. Dies gibt Ihnen den Gradienten des Fehlers am Output, also dort, wo die Vorhersagen des Netzwerks direkt mit den tatsächlichen Zielwerten verglichen werden.

## Gradient für die letzte Schicht

Dieser Gradient wird dann verwendet, um die partiellen Ableitungen (Gradienten) der Gewichte und Biases der letzten Schicht zu berechnen. Diese berechnen Sie, indem Sie den Output-Gradienten mit den Ausgaben der vorhergehenden Schicht (oder dem Input zur letzten Schicht) kombinieren.

## Propagation des Fehlers (Backpropagation)

Nachdem Sie den Gradienten für die letzte Schicht berechnet haben, „propagieren“ Sie diesen Fehler zurück durch das Netzwerk. Für jede vorige Schicht berechnen Sie den Fehlergradienten basierend auf den Gewichten der nachfolgenden Schicht und der Ableitung der Aktivierungsfunktion dieser Schicht.

## Aktualisierung der Gewichte

Schließlich nutzen Sie die berechneten Gradienten für jede Schicht, um die Gewichte und Biases mittels eines Optimierungsverfahrens (typischerweise Gradientenabstieg) zu aktualisieren. Dies reduziert schrittweise den Gesamtfehler des Netzwerks.
