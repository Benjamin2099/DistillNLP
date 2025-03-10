📖 Knowledge Distillation für Textklassifikation
🚀 Knowledge Distillation für effiziente Textklassifikation:
Dieses Projekt nutzt Knowledge Distillation, um ein leistungsfähiges Teacher-Modell zu trainieren und daraus ein kompaktes Student-Modell abzuleiten. Das Student-Modell ist für den Einsatz auf Edge-Geräten und für ressourcenarme Umgebungen optimiert.

🔹 Features
✅ Knowledge Distillation – Effiziente Modellkomprimierung
✅ Sentiment-Analyse für Textklassifikation – Positiv/Negativ-Vorhersage
✅ Modularer Machine Learning Workflow – Datenaufbereitung, Training, Evaluierung
✅ REST API mit Flask – Echtzeit-Modelleinsatz für Vorhersagen
✅ GPU-Unterstützung – Automatische Nutzung von CUDA falls verfügbar

🏗 Projektstruktur

knowledge_distillation_text/
├── README.md                          # Projektbeschreibung, Anleitung und Nutzungshinweise
├── requirements.txt                   # Liste der benötigten Python-Pakete (Dependencies)
├── setup.py                           # Setup-Skript zur Installation des Projekts als Python-Paket
├── .gitignore                         # Dateien/Ordner, die Git ignorieren soll
├── Dockerfile                         # Docker-Konfiguration für Container-Deployment
├── config/
│   ├── default.yaml                   # Standard-Konfiguration (Hyperparameter, Pfade, etc.)
│   └── logging.conf                   # Logging-Konfiguration
├── data/
│   ├── vocab.json                     # Vokabular-Datei (Mapping von Tokens zu Indizes)
│   ├── sample_dataset.csv             # Beispieldatensatz für Sentiment-Analyse
├── docs/
│   └── knowledge_distillation_text_documentation.json  # Ausführliche Projektdokumentation
├── src/
│   ├── __init__.py                    
│   ├── data_preprocessing.py          # Tokenisierung, Vokabularaufbau, Padding
│   ├── models.py                      # Definitionen von Teacher- und Student-Modellen
│   ├── training.py                    # Trainingsfunktionen für Teacher und Student
│   ├── evaluation.py                  # Evaluierungsmetriken (Accuracy, Precision, etc.)
│   ├── utils.py                       # Hilfsfunktionen (Logging, Modellmanagement)
├── tests/
│   ├── test_data_preprocessing.py     # Unit-Tests für Datenverarbeitung
│   ├── test_models.py                 # Tests für Modellarchitekturen
│   ├── test_training.py               # Tests für Trainings- und Evaluierungsfunktionen
├── distillNLP_main.py # Hauptskript für Training, Evaluierung & Distillation
└── app.py                             # Flask-API für Echtzeit-Inferenz
🚀 Installation
1️⃣ Klonen des Repositorys

git clone https://github.com/dein-benutzername/knowledge_distillation_text.git
cd knowledge_distillation_text
2️⃣ Erstellen einer virtuellen Umgebung

python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows
3️⃣ Installieren der Abhängigkeiten

pip install -r requirements.txt
🏋 Modelltraining & Evaluierung
1️⃣ Training des Teacher-Modells

python distillation_edge_text_extended.py --mode teacher
🔹 Speichert das Teacher-Modell unter models/teacher_model.pth.

2️⃣ Training des Student-Modells mit Knowledge Distillation

python distillation_edge_text_extended.py --mode student
🔹 Speichert das Student-Modell unter models/student_model.pth.

3️⃣ Evaluierung des Student-Modells

python distillation_edge_text_extended.py --mode evaluate
🔹 Zeigt Genauigkeit, Precision, Recall und F1-Score.

🌐 Bereitstellung der API
1️⃣ Starten der Flask-API

python app.py
🔹 Die API läuft unter http://localhost:5000.

2️⃣ Testen der API mit cURL

curl -X POST "http://localhost:5000/predict" -H "Content-Type: application/json" -d '{"text": "I love this movie"}'
🔹 Antwort:

  "text": "I love this movie",
  "prediction": 1,
  "sentiment": "positiv"
}
🐳 Docker Deployment
1️⃣ Erstellen des Docker-Containers

docker build -t knowledge_distillation_text .
2️⃣ Starten des Containers

docker run -p 5000:5000 knowledge_distillation_text
📊 Evaluierungsmethoden
Nach dem Training und der Distillation werden folgende Metriken zur Modellbewertung berechnet:

Metrik	Beschreibung
Accuracy	Gibt an, wie viele Vorhersagen korrekt sind.
Precision	Misst, wie viele als positiv vorhergesagte Klassen tatsächlich positiv sind.
Recall	Zeigt, wie viele der tatsächlichen positiven Klassen korrekt erkannt wurden.
F1-Score	Harmonic Mean zwischen Precision & Recall.
Confusion Matrix	Zeigt die Verteilung der Vorhersagen über die Klassen.
🔥 Erweiterungen & Optimierungen
📌 Multilinguale Unterstützung – Modelle für verschiedene Sprachen trainieren
📌 Optimierte Modelle für Mobile & Edge-Devices – Quantisierte Student-Modelle für IoT
📌 Automatische Hyperparameter-Suche – Grid Search für die besten Modellparameter

📜 Lizenz
Dieses Projekt steht unter der MIT-Lizenz.

🤝 Mitwirken
Beiträge sind willkommen! Falls du dieses Projekt verbessern möchtest:

Forke das Repository 🚀
Erstelle einen neuen Branch (feature/neues-feature)
Committe deine Änderungen
Erstelle einen Pull-Request 💡
📩 Kontakt & Support
👤 Dein Name
📧 dein.email@example.com
🌍 GitHub-Projektseite

📌 Fazit
✅ End-to-End Machine Learning Pipeline – Von Training bis API-Deployment
✅ Knowledge Distillation für optimierte Modelle – Schnelle & kompakte Student-Modelle
✅ Echtzeit-API für Textklassifikation – Sentiment-Analyse in Sekundenschnelle
✅ Skalierbar & erweiterbar für Deep Learning – Perfekt für Edge- & Cloud-Deployment 🚀

Falls du weitere Fragen hast oder Anpassungen benötigst, lass es mich wissen! 🎯🚀

Hier ist der vollständige Inhalt der requirements.txt-Datei, die alle benötigten Python-Abhängigkeiten für dein Knowledge Distillation für Textklassifikation-Projekt enthält.

📄 requirements.txt
txt
Kopieren
Bearbeiten
# PyTorch und verwandte Pakete für Deep Learning
torch>=1.10.0
torchvision>=0.11.0
torchaudio>=0.10.0

# NLP-Bibliotheken für Textverarbeitung
nltk>=3.6.5
spacy>=3.2.0
transformers>=4.15.0
sentencepiece>=0.1.96

# Datenverarbeitung und Speicherung
numpy>=1.21.2
scipy>=1.7.1
pandas>=1.3.3
pyyaml>=5.4.1

# Modellbewertung und Visualisierung
scikit-learn>=0.24.2
matplotlib>=3.4.3
seaborn>=0.11.2

# Web-API für Inferenz
flask>=2.0.2
flask-restful>=0.3.9

# Logging und Debugging
loguru>=0.5.3

# Tests und Qualitätssicherung
pytest>=6.2.5
pytest-cov>=2.12.1

# Optimierung & Scheduling
tqdm>=4.62.3
📌 Erklärung der Abhängigkeiten
Kategorie	Pakete	Beschreibung
Deep Learning	torch, torchvision, torchaudio	PyTorch-Framework für das Training von neuronalen Netzwerken.
NLP & Textverarbeitung	nltk, spacy, transformers, sentencepiece	Tokenisierung, Embeddings, vortrainierte Modelle (z. B. BERT, DistilBERT).
Datenverarbeitung	numpy, scipy, pandas, pyyaml	Verarbeitung von Textdaten und Speicherung von Konfigurationen.
Modellbewertung & Visualisierung	scikit-learn, matplotlib, seaborn	Berechnung von Metriken (Accuracy, F1-Score) und Visualisierung der Ergebnisse.
Web-API & Flask	flask, flask-restful	Bereitstellung eines REST-API-Servers für Modellvorhersagen.
Logging & Debugging	loguru	Erweiterte Logging-Funktionalitäten für einfaches Debugging.
Tests & Qualitätssicherung	pytest, pytest-cov	Automatische Tests für Modelltraining, Inferenz & Datenverarbeitung.
Optimierung & Training	tqdm	Fortschrittsbalken für lange Trainingsprozesse.
📌 Installation der Abhängigkeiten
Nach dem Klonen des Repositorys kannst du die Pakete mit folgendem Befehl installieren:

pip install -r requirements.txt
Falls du eine virtuelle Umgebung verwenden möchtest:

python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows
pip install -r requirements.txt
📌 Warum diese requirements.txt?
✅ Minimal & Optimiert – Enthält nur essenzielle Pakete für Training, Inferenz & API.
✅ Modular & Skalierbar – Unterstützt sowohl Deep Learning als auch klassische NLP-Techniken.
✅ Einfach zu installieren – Keine unnötigen Bibliotheken oder redundante Pakete.

Falls du weitere Pakete für GPU-Support (z. B. CUDA), Model Deployment oder Hyperparameter-Tuning hinzufügen möchtest, lass es mich wissen! 🚀

Warum verwenden wir Soft Labels vom Teacher für das Student-Modell?
Die Übertragung von Soft Labels (weicheren Wahrscheinlichkeitsverteilungen) vom Teacher-Modell auf das Student-Modell ist der Kern von Knowledge Distillation. Diese Methode wurde von Geoffrey Hinton et al. eingeführt und hilft dabei, kleinere Modelle zu trainieren, die sich ähnlich wie das größere Modell verhalten, aber effizienter sind.

1️⃣ Was sind Soft Labels?
Anstatt dass das Teacher-Modell nur eine harte 1/0-Klassifikation ausgibt, gibt es eine Wahrscheinlichkeitsverteilung über alle Klassen zurück.
Beispiel für harte Labels (klassische Cross-Entropy):

{"Positive": 1, "Negative": 0}
Beispiel für Soft Labels (Teacher-Modell mit Temperatur=2):

{"Positive": 0.85, "Negative": 0.15}
→ Das Modell ist sich sicher, aber nicht zu 100 %.
💡 Soft Labels enthalten mehr Informationen darüber, wie das Teacher-Modell entscheidet!

2️⃣ Warum ist das sinnvoll für das Student-Modell?
1️⃣ Verhindert Overfitting an harte Labels

Wenn das Student-Modell nur harte Labels sieht, lernt es weniger über feine Unterschiede zwischen Klassen.
Soft Labels zeigen an, wie stark ein Beispiel zu einer Klasse gehört, nicht nur ob es richtig oder falsch ist.
2️⃣ Führt zu besseren Generalisierungen

Klassische harte Labels geben keine Ähnlichkeitsinformationen weiter.
Beispiel:
"Great movie" → 100 % Positiv
"Nice film" → 85 % Positiv, 15 % Neutral
"Not bad" → 55 % Positiv, 45 % Negativ
Das Student-Modell lernt so, dass manche Sätze ambivalent sind und nicht immer klar einer Kategorie zugeordnet werden können.
3️⃣ Komprimiert Wissen in ein kleineres Modell

Das Teacher-Modell ist groß & komplex (z. B. BERT, LSTMs).
Das Student-Modell ist kompakt & schneller.
Das Student-Modell kann durch Soft Labels viel von der Intelligenz des Teacher-Modells übernehmen, ohne dessen Größe zu haben.
4️⃣ Bessere Performance auf Edge-Geräten & Mobile AI

Viele Unternehmen (Google, OpenAI) setzen Knowledge Distillation ein, um kleinere Modelle zu trainieren, die fast so gut sind wie große Modelle, aber weniger Rechenleistung benötigen.
3️⃣ Beispiel: Klassische vs. Distillation-Training
Ohne Knowledge Distillation (nur harte Labels)
Das Student-Modell sieht nur 0/1-Klassen:

["Great movie!" → 1]
["Terrible acting!" → 0]
🔴 Problem:

Das Modell lernt, dass eine Vorhersage immer 100 % sicher ist.
Keine Information darüber, wie ähnlich zwei Sätze sind.
Mit Knowledge Distillation (Soft Labels vom Teacher)
Das Teacher-Modell sagt für "Okay film":

{"Positive": 0.55, "Negative": 0.45}
Das Student-Modell lernt:

"Okay film" ist nicht ganz positiv, sondern eher neutral.
Es bekommt mehr Kontext & Nuancen, was das Lernen verbessert.
4️⃣ Wie wird das mathematisch umgesetzt?
Wir kombinieren zwei Loss-Funktionen:

1️⃣ KL-Divergenz (Distillation Loss):
Vergleich der Softmax-Wahrscheinlichkeiten vom Teacher & Student.
2️⃣ Klassische Cross-Entropy (CE-Loss):
Vergleich mit den echten Labels.
📌 Formel für den kombinierten Loss:

Loss=α⋅KL(softmax(S/T),softmax(T/T))+(1−α)⋅CE(S,Y)
𝑆 = Student-Modell-Logits
𝑇 = Teacher-Modell-Logits
𝑌= Harte Labels
𝑇(Temperatur) → Glättet die Wahrscheinlichkeiten
𝛼 → Gewichtung zwischen Distillation und klassischem Loss
📌 Fazit
✅ Soft Labels enthalten mehr Wissen als harte Labels.
✅ Das Student-Modell kann besser verallgemeinern & Overfitting vermeiden.
✅ Ermöglicht kleinere & schnellere Modelle mit vergleichbarer Performance.
✅ Wird häufig in Mobile AI & Edge-Deployments genutzt.

📌 Beispiel in der Praxis: Google verwendet Knowledge Distillation für Google Assistant, um ein kleines Modell auf Smartphones laufen zu lassen, das fast so gut ist wie große Server-Modelle. 🚀