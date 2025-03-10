import os
import json
import yaml
import logging
import logging.config
import torch

def setup_logging(logging_config_path="config/logging.conf"):
    """
    Richtet das Logging basierend auf einer Konfigurationsdatei ein.
    
    Falls die Konfigurationsdatei nicht existiert, wird eine Standardkonfiguration verwendet.
    
    Args:
        logging_config_path (str): Pfad zur Logging-Konfigurationsdatei.
    """
    if os.path.exists(logging_config_path):
        # Stelle sicher, dass das Log-Verzeichnis existiert:
        log_file = "logs/project.log"
        log_dir = os.path.dirname(log_file)
        os.makedirs(log_dir, exist_ok=True)
        
        logging.config.fileConfig(logging_config_path, disable_existing_loggers=False)
        logging.info("Logging-Konfiguration aus '{}' geladen.".format(logging_config_path))
    else:
        # Standard-Logging-Konfiguration, falls die Datei nicht vorhanden ist
        logging.basicConfig(level=logging.INFO,
                            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
        logging.warning("Logging-Konfigurationsdatei '{}' nicht gefunden. Standardkonfiguration wird verwendet.".format(logging_config_path))


def load_config(config_path="config/default.yaml"):
    """
    Lädt die Konfiguration aus einer YAML-Datei und gibt sie als Dictionary zurück.
    
    Args:
        config_path (str): Relativer Pfad zur YAML-Konfigurationsdatei.
    
    Returns:
        dict: Die geladene Konfiguration.
    """
    # Ermittle das Stammverzeichnis des Projekts relativ zu diesem Modul
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    full_path = os.path.join(base_dir, config_path)
    
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"Konfigurationsdatei nicht gefunden: {full_path}")
    with open(full_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    logging.info("Konfiguration aus '{}' geladen.".format(full_path))
    return config

def save_model(model, path):
    """
    Speichert das übergebene Modell als .pth-Datei.
    
    Args:
        model (torch.nn.Module): Das zu speichernde Modell.
        path (str): Pfad, unter dem das Modell gespeichert werden soll.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    logging.info(f"Modell wurde unter '{path}' gespeichert.")

def load_model(model, path, device="cpu"):
    """
    Lädt Modellgewichte aus einer .pth-Datei in das gegebene Modell.
    
    Args:
        model (torch.nn.Module): Das Modell, in das die Gewichte geladen werden sollen.
        path (str): Pfad zur gespeicherten Modell-Datei.
        device (str): Das Gerät ("cpu" oder "cuda").
    
    Returns:
        torch.nn.Module: Das Modell mit geladenen Gewichten.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Modell-Datei nicht gefunden: {path}")
    model.load_state_dict(torch.load(path, map_location=torch.device(device)))
    model.to(device)
    model.eval()
    logging.info(f"Modellgewichte wurden aus '{path}' geladen und auf {device} gesetzt.")
    return model

def get_device():
    """
    Bestimmt das verfügbare Gerät (GPU falls verfügbar, ansonsten CPU).
    
    Returns:
        str: "cuda" oder "cpu"
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Verwendetes Gerät: {device}")
    return device

if __name__ == "__main__":
    # Beispielhafte Nutzung der Hilfsfunktionen:
    setup_logging()
    config = load_config("config/default.yaml")
    print("Geladene Konfiguration:", config)

    # Beispielmodell (Dummy-Modell) erstellen, speichern und wieder laden
    from src.models import StudentNetText  # Beispiel: Nutze ein Modell aus src/models.py
    model = StudentNetText(vocab_size=5000, embed_dim=128, num_classes=2)
    save_model(model, "models/dummy_model.pth")
    loaded_model = load_model(model, "models/dummy_model.pth", device=get_device())
    print("Modell erfolgreich geladen:", loaded_model)

