"""
app.py

Diese Flask-API dient zur Echtzeitanwendung des distillierten Student-Modells für die Textklassifikation (z. B. Sentiment-Analyse).

Die API erwartet einen POST-Request an den Endpunkt /predict mit einem JSON-Objekt, das ein "text"-Feld enthält.
Das Modell verarbeitet den Text und gibt eine Vorhersage (z. B. "positiv" oder "negativ") zurück.

Beispiel-Request (JSON):
{
    "text": "I love this movie"
}

Beispiel-Antwort (JSON):
{
    "text": "I love this movie",
    "prediction": 1,
    "sentiment": "positiv"
}
"""

import json
import torch
from flask import Flask, request, jsonify
from src.models import StudentNetText
from src.utils import load_model, get_device, setup_logging
from src.data_preprocessing import load_vocab, preprocess_texts

# Restlicher Code...


# Flask-Anwendung initialisieren
app = Flask(__name__)

# Logging-Konfiguration laden (optional, falls in utils eingerichtet)
setup_logging()

# Vokabular laden
vocab = load_vocab("data/vocab.json")
vocab_size = len(vocab)

# Geräteauswahl (GPU falls verfügbar, sonst CPU)
device = get_device()

# Modell initialisieren und Gewichtungen laden
# Die Parameter sollten idealerweise mit den in der Konfiguration definierten Werten übereinstimmen
model = StudentNetText(vocab_size=vocab_size, embed_dim=128, num_classes=2)
model = load_model(model, "models/student_model.pth", device)
model.eval()

# Definierte maximale Sequenzlänge (muss mit der Konfiguration übereinstimmen)
MAX_SEQUENCE_LENGTH = 20

@app.route("/predict", methods=["POST"])
def predict():
    """
    Endpunkt für die Echtzeit-Inferenz.
    
    Erwartet einen POST-Request mit JSON-Body:
    {
        "text": "Beispieltext für die Klassifikation"
    }
    
    Gibt eine JSON-Antwort zurück, die den Originaltext, die numerische Vorhersage und die interpretierte Klassifikation (z. B. positiv/negativ) enthält.
    """
    data = request.get_json(force=True)
    
    # Überprüfe, ob der Schlüssel "text" vorhanden ist
    if "text" not in data:
        return jsonify({"error": "Kein Text im Request gefunden"}), 400

    input_text = data["text"]
    # Vorverarbeitung: Text in Sequenz umwandeln und auf MAX_SEQUENCE_LENGTH paddieren
    sequence = preprocess_texts([input_text], vocab, max_length=MAX_SEQUENCE_LENGTH)
    input_tensor = torch.tensor(sequence, dtype=torch.long).to(device)
    
    # Inferenz: Modell im Evaluierungsmodus
    with torch.no_grad():
        outputs = model(input_tensor)
        prediction = torch.argmax(outputs, dim=1).item()
    
    # Annahme: 1 = positiv, 0 = negativ
    sentiment = "positiv" if prediction == 1 else "negativ"
    
    return jsonify({
        "text": input_text,
        "prediction": prediction,
        "sentiment": sentiment
    })

if __name__ == "__main__":
    # Starte die Flask-API, die standardmäßig auf Port 5000 erreichbar ist
    app.run(host="0.0.0.0", port=5000)
