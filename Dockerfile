# Basis-Image für Python mit minimalem Overhead
FROM python:3.8-slim

# Setze das Arbeitsverzeichnis im Container
WORKDIR /app

# Kopiere die requirements.txt und installiere die Abhängigkeiten
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Kopiere den gesamten Projektinhalt in den Container
COPY . .

# Setze Umgebungsvariablen (optional)
ENV PYTHONUNBUFFERED=1

# Exponiere den Port, auf dem die API läuft (z. B. 5000)
EXPOSE 5000

# Standardkommando: Starte die Flask-API
CMD ["python", "app.py"]
