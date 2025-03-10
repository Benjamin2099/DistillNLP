"""
distillation_edge_text_extended.py

Dieses Skript steuert den gesamten Workflow für DistillNLP:
- Laden der Konfiguration aus 'config/default.yaml'
- Initialisierung der Daten (Beispieldatensatz & Vokabular)
- Training des Teacher-Modells (klassisches Training)
- Training des Student-Modells mittels Knowledge Distillation
- Evaluierung des trainierten Modells

Über Kommandozeilenargumente kann der Modus gewählt werden:
  --mode teacher    : Training des Teacher-Modells
  --mode student    : Training des Student-Modells via Distillation
  --mode evaluate   : Evaluierung des Student-Modells

Beispielaufruf:
  python distillation_edge_text_extended.py --mode teacher --epochs 10
"""

import argparse
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.data_preprocessing import load_vocab, preprocess_texts
from src.models import TeacherNetText, StudentNetText
from src.training import train_teacher, train_student
from src.evaluation import evaluate_model
from src.utils import load_config, setup_logging, save_model, load_model, get_device

def create_dataloaders(vocab, max_length, batch_size):
    """
    Erzeugt Beispiel-Daten und DataLoader für Training und Validierung.
    Für ein reales Projekt sollten hier echte Datensätze verwendet werden.
    
    Args:
        vocab (dict): Vokabular-Mapping.
        max_length (int): Maximale Sequenzlänge.
        batch_size (int): Batch-Größe.
    
    Returns:
        DataLoader, DataLoader: Trainings- und Validierungs-DataLoader.
    """
    # Beispiel-Daten (Dummy-Daten)
    sample_texts = [
        "I love this movie", 
        "This film was terrible", 
        "Fantastic storyline", 
        "Not worth watching",
        "An amazing performance", 
        "Poor acting", 
        "Great direction", 
        "Worst movie ever"
    ]
    sample_labels = [1, 0, 1, 0, 1, 0, 1, 0]  # 1 = positiv, 0 = negativ

    sequences = preprocess_texts(sample_texts, vocab, max_length)
    sequences = torch.tensor(sequences, dtype=torch.long)
    labels = torch.tensor(sample_labels, dtype=torch.long)

    # Erstelle einen einfachen TensorDataset
    dataset = TensorDataset(sequences, labels)
    # Verwende denselben DataLoader für Training und Validierung (für dieses Beispiel)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader

def main(mode, epochs, config_path="config/default.yaml"):
    """
    Führt basierend auf dem ausgewählten Modus den Trainings-, Evaluierungs- oder Distillations-Workflow aus.
    
    Args:
        mode (str): "teacher", "student" oder "evaluate".
        epochs (int): Anzahl der Epochen.
        config_path (str): Pfad zur Konfigurationsdatei.
    """
    # Logging und Konfiguration laden
    setup_logging()
    config = load_config(config_path)
    device = get_device()
    
    # Vokabular laden
    vocab = load_vocab(config["data"]["vocab_path"])
    vocab_size = len(vocab)
    
    # Erstelle DataLoader
    max_length = config["data"]["max_sequence_length"]
    batch_size = config["data"]["batch_size"]
    train_loader, val_loader = create_dataloaders(vocab, max_length, batch_size)
    
    if mode == "teacher":
        print("🔹 Starte Training des Teacher-Modells...")
        teacher_model = TeacherNetText(
            vocab_size=vocab_size,
            embed_dim=config["model"]["teacher"].get("embed_dim", 256),
            hidden_dim=config["model"]["teacher"].get("hidden_dim", 512),
            num_classes=config["model"]["teacher"].get("num_classes", 2),
            dropout=config["model"]["teacher"].get("dropout", 0.3)
        ).to(device)
        optimizer = torch.optim.Adam(teacher_model.parameters(), lr=config["training"]["learning_rate"])
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)
        
        teacher_model = train_teacher(teacher_model, train_loader, val_loader, optimizer, scheduler, epochs=epochs, device=device)
        save_model(teacher_model, config["distillation"]["teacher_path"])
        print("✅ Teacher-Modell erfolgreich trainiert und gespeichert.")
    
    elif mode == "student":
        print("🔹 Starte Training des Student-Modells mittels Knowledge Distillation...")
        teacher_model = TeacherNetText(
            vocab_size=vocab_size,
            embed_dim=config["model"]["teacher"].get("embed_dim", 256),
            hidden_dim=config["model"]["teacher"].get("hidden_dim", 512),
            num_classes=config["model"]["teacher"].get("num_classes", 2),
            dropout=config["model"]["teacher"].get("dropout", 0.3)
        ).to(device)
        # Lade die vortrainierten Teacher-Gewichte
        teacher_model = load_model(teacher_model, config["distillation"]["teacher_path"], device)
        teacher_model.eval()  # Fixiere das Teacher-Modell

        student_model = StudentNetText(
            vocab_size=vocab_size,
            embed_dim=config["model"]["student"].get("embedding_dim", 128),
            num_classes=config["model"]["student"].get("num_classes", 2)
        ).to(device)
        optimizer = torch.optim.Adam(student_model.parameters(), lr=config["training"]["learning_rate"])
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)
        
        student_model = train_student(
            teacher_model, student_model, train_loader, val_loader, optimizer, scheduler,
            epochs=epochs,
            temperature=config["distillation"].get("temperature", 2.0),
            alpha=config["distillation"].get("alpha", 0.7),
            device=device
        )
        save_model(student_model, config["distillation"]["student_path"])
        print("✅ Student-Modell erfolgreich trainiert und gespeichert.")

    elif mode == "evaluate":
        print("🔹 Starte Evaluierung des Student-Modells...")
        student_model = StudentNetText(
            vocab_size=vocab_size,
            embed_dim=config["model"]["student"].get("embedding_dim", 128),
            num_classes=config["model"]["student"].get("num_classes", 2)
        ).to(device)
        student_model = load_model(student_model, config["distillation"]["student_path"], device)
        metrics = evaluate_model(student_model, val_loader, device)
        print("Evaluierungsergebnisse:")
        print(metrics)
    else:
        print("⚠️ Ungültiger Modus! Bitte wähle 'teacher', 'student' oder 'evaluate'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hauptskript für DistillNLP: Training, Evaluierung & Knowledge Distillation")
    parser.add_argument("--mode", type=str, choices=["teacher", "student", "evaluate"], required=True,
                        help="Wähle 'teacher' für Training des Teacher-Modells, 'student' für Distillation oder 'evaluate' für Evaluierung.")
    parser.add_argument("--epochs", type=int, default=10, help="Anzahl der Trainings-Epochen.")
    args = parser.parse_args()
    main(args.mode, args.epochs)
