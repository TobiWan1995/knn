# Ableitung der Loss-Funktion nach dem Output 𝑜o des Netzes

Dies ist der erste Schritt, bei dem der Gradient der Loss-Funktion direkt am Output des Netzes berechnet wird. Diese Ableitung bewertet, wie sich kleine Veränderungen im Output auf den berechneten Loss auswirken.

## Ableitung der Loss-Funktion nach den Gewichten 𝑊W und Biases 𝑏b

### Für Gewichte:

Nachdem Sie den Gradienten der Loss-Funktion bezüglich des Outputs haben (den Sie mit Ableitung der Kostenfunktion nach Output loss_grad(o, y) berechnen), müssen Sie als Nächstes herausfinden, wie sich Änderungen an jedem Gewicht auf den Output auswirken, und somit indirekt auf den Loss. Dies wird durch die Multiplikation des Eingabevektors x (fuer mehrere Schichten der Eingabevektor jeder Schicht) mit diesem Gradienten erreicht, was Ihnen den Gradienten jedes Gewichts gibt. Dies entspricht der Kettenregel in der Differentialrechnung, wobei Sie die Änderung des Outputs durch eine Änderung der Gewichte  und dann die Änderung des Loss durch eine Änderung des Outputs betrachten.

### Für Biases:

Da der Einfluss eines Bias auf den Output unabhängig vom Wert der Eingabe ist (jeder Bias wird direkt zum Ausgabewert des Neurons addiert), ist der Gradient des Loss bezüglich eines Bias gleich dem Gradienten des Loss bezüglich des Outputs des Neurons.

## Backpropagation für vorherige Schichten (wenn vorhanden)

Wenn das Netzwerk mehrere Schichten hat, wird der berechnete Gradient des Outputs verwendet, um rückwärts durch das Netzwerk zu gehen und die Gradienten für alle vorigen Schichten zu berechnen. Das beinhaltet die Berechnung der Ableitungen der Ausgaben der versteckten Schichten und die Aktualisierung ihrer Gewichte und Biases basierend auf ihren spezifischen Beiträgen zum Output und somit zum Fehler.

## Zusammenfassung

In einem trainierten neuronalen Netzwerk führt die Berechnung der Ableitungen durch die gesamte Architektur (alle Schichten). Die Backpropagation nutzt die Kettenregel, um den Einfluss jeder Gewichts- und Bias-Änderung in jeder Schicht auf den Endloss zu bestimmen. Die endgültige Aktualisierung der Gewichte und Biases mit ihren berechneten Gradienten verbessert sukzessive die Netzwerkperformance, indem sie den Gesamtloss minimiert. Dieser Prozess wird iterativ über viele Epochen wiederholt, um das Netzwerk optimal auf die Trainingsdaten abzustimmen.
