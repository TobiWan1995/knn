# Unterschiede zwischen k-Means und Gaussian Mixture Models (GMM)

## k-Means Clustering

- **Zuordnungsart**: Deterministisch. Jeder Datenpunkt wird dem nächstgelegenen Clusterzentrum zugeordnet.
- **Clusterform**: Optimiert für sphärische Cluster, da es auf dem Abstand zum Clusterzentrum basiert.
- **Skalierbarkeit**: Sehr effizient bei großen Datensätzen.
- **Robustheit**: Empfindlich gegenüber Ausreißern und nicht optimal für Cluster mit variierenden Größen und Dichten.
- **Algorithmus**: Einfach und schnell, verwendet die Euclidean-Distanz für die Zuordnung der Datenpunkte zu den Clustern.

## Gaussian Mixture Models (GMM)

- **Zuordnungsart**: Probabilistisch. Jeder Datenpunkt erhält eine Wahrscheinlichkeit für die Zugehörigkeit zu jedem Cluster.
- **Clusterform**: Flexibel, da jede Komponente eine eigene Kovarianzstruktur haben kann; unterstützt elliptische Cluster.
- **Skalierbarkeit**: Weniger effizient als k-Means bei sehr großen Datensätzen, da es aufwendigere Berechnungen erfordert.
- **Robustheit**: Kann komplexe Clusterstrukturen besser modellieren, insbesondere bei überlappenden Clustern.
- **Algorithmus**: Komplexer, basiert auf der Maximierung der Log-Likelihood unter Annahme einer Gaußschen Verteilung der Daten.

## Probabilistisch vs. Deterministisch

- **Probabilistisch (GMM)**: Jeder Datenpunkt hat eine Reihe von Wahrscheinlichkeiten, die seine Zugehörigkeit zu jedem der möglichen Cluster repräsentieren. Dies erlaubt eine "weiche" Zuordnung, was besonders in unsicheren oder überlappenden Datenstrukturen nützlich ist.
- **Deterministisch (k-Means)**: Jeder Datenpunkt wird genau einem Cluster zugeordnet, basierend auf dem kürzesten Abstand zum Clusterzentrum. Dies führt zu einer "harten" Zuordnung.

Die Wahl der Methode hängt von der Natur der Daten und den spezifischen Anforderungen des Projekts ab. GMM ist besonders nützlich, wenn die Datenstruktur komplex ist und eine weichere Clusterzuordnung erfordert, während k-Means für große, gut separierbare Datenmengen effizienter sein kann.
