"""
DistillNLP – Ein Paket zur Implementierung von Knowledge Distillation für Textklassifikation.

Dieses Paket umfasst folgende Module:
- data_preprocessing: Funktionen für die Tokenisierung, den Vokabularaufbau, die Sequenzumwandlung und das Padding.
- models: Definitionen für das Teacher- und das Student-Modell.
- training: Trainingslogik, einschließlich Knowledge Distillation zwischen Teacher und Student.
- evaluation: Funktionen zur Evaluierung der Modellleistung anhand verschiedener Metriken.
- utils: Diverse Hilfsfunktionen für Logging, Konfigurationsmanagement und Modell-Speicherung.

Mit DistillNLP können leistungsfähige, aber rechenintensive Modelle (Teacher) verwendet werden, um kompakte Modelle (Student) zu trainieren, die ideal für den Einsatz auf Edge-Geräten und mobilen Plattformen sind.
"""


