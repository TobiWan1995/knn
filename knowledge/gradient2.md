# Verständnis von x in Mehrschichtnetzen

In einem mehrschichtigen Perzeptron (MLP) oder jedem anderen tiefen neuronalen Netzwerk verändert sich der "Eingabevektor" x von Schicht zu Schicht. Hier die Details:

## Ursprünglicher Eingabevektor

Bei der ersten Schicht des Netzwerks ist der Eingabevektor 𝑥x die tatsächliche Eingabe in das Netzwerk, also die Trainingsdaten, die dem Netzwerk für die Vorwärtspropagation gegeben werden.

## Transformierte Eingabe für jede folgende Schicht

Für jede nachfolgende Schicht wird der "Eingabevektor" durch die Ausgabe der vorherigen Schicht definiert. Dies bedeutet, dass das, was als x für eine Schicht dient, tatsächlich die aktivierten Ausgaben der vorherigen Schicht sind.

Mathematisch ausgedrückt, wenn x(l−1) der Ausgabevektor der Schicht l−1 ist, dann wird dieser Vektor durch Gewichte W(l) und Biases b(l) der Schicht l transformiert, und durch eine Aktivierungsfunktion 𝜎σ geleitet, um x(l), die Eingabe für die Schicht l+1, zu erzeugen.

## Aktualisierung der Gewichte und Biases

Während des Backpropagation-Prozesses wird ∂L zu ∂x(l) berechnet, um die Änderung des Loss in Bezug auf die Ausgabe der Schicht 𝑙l zu verstehen. Diese Gradienteninformation wird dann verwendet, um die Gewichte W(l) und Biases b(l) zu aktualisieren, die die Transformation von x(l−1) zu x(l) verantworten.

## Beispiel

Wenn ein MLP aus drei Schichten besteht und x die initiale Eingabe ist, dann wäre:

x(1)=σ(W(1)x+b(1)) die Ausgabe der ersten Schicht und die Eingabe für die zweite Schicht.

x(2)=σ(W(2)x(1)+b(2)) die Ausgabe der zweiten Schicht und die Eingabe für die dritte Schicht.

x(3)=σ(W(3)x(2)+b(3)) die Ausgabe der dritten Schicht und der finale Output des Netzwerks für den gegebenen Eingabewert 𝑥x.

## Schlussfolgerung

In mehrschichtigen Netzen ist der Eingabevektor für jede Schicht das Ergebnis der Transformation der Eingabe durch die vorherige Schicht, angepasst durch Gewichte und Biases und modifiziert durch die Aktivierungsfunktion. Dieser Prozess erlaubt es dem Netzwerk, zunehmend komplexere Muster und Beziehungen in den Daten zu lernen und zu modellieren.
