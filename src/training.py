import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from tqdm import tqdm

def train_teacher(model, train_loader, val_loader, optimizer, scheduler, epochs=10, device="cpu"):
    """
    Trainiert das Teacher-Modell mit klassischer Cross-Entropy Loss.
    
    Args:
        model (nn.Module): Das Teacher-Modell.
        train_loader (DataLoader): Trainingsdaten.
        val_loader (DataLoader): Validierungsdaten.
        optimizer (torch.optim.Optimizer): Optimierer.
        scheduler (torch.optim.lr_scheduler._LRScheduler): Lernraten-Scheduler.
        epochs (int): Anzahl der Epochen.
        device (str): "cuda" oder "cpu".
    
    Returns:
        nn.Module: Das trainierte Teacher-Modell.
    """
    criterion = nn.CrossEntropyLoss()
    model.to(device)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Teacher Training - Epoch {epoch+1}/{epochs}")
        
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            progress_bar.set_postfix(loss=f"{running_loss/len(train_loader):.4f}")
        
        # Validierung
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        val_loss_avg = val_loss / len(val_loader)
        logging.info(f"Teacher - Epoch {epoch+1}/{epochs}: Train Loss: {running_loss/len(train_loader):.4f}, Val Loss: {val_loss_avg:.4f}")
        scheduler.step(val_loss_avg)
    
    return model

def train_student(teacher_model, student_model, train_loader, val_loader, optimizer, scheduler, epochs=10, temperature=2.0, alpha=0.7, device="cpu"):
    """
    Trainiert das Student-Modell mittels Knowledge Distillation.
    
    Der Teacher wird als festes Modell verwendet, um weiche Zielwerte (Soft Labels)
    für das Student-Modell zu liefern. Der kombinierte Loss aus KL-Divergenz und 
    klassischer Cross-Entropy wird zur Optimierung genutzt.
    
    Args:
        teacher_model (nn.Module): Vortrainiertes Teacher-Modell.
        student_model (nn.Module): Das zu trainierende Student-Modell.
        train_loader (DataLoader): Trainingsdaten.
        val_loader (DataLoader): Validierungsdaten.
        optimizer (torch.optim.Optimizer): Optimierer für das Student-Modell.
        scheduler (torch.optim.lr_scheduler._LRScheduler): Lernraten-Scheduler.
        epochs (int): Anzahl der Epochen.
        temperature (float): Temperaturparameter zur Glättung der Softmax-Ausgaben.
        alpha (float): Gewichtung zwischen KL-Divergenz und Cross-Entropy Loss.
        device (str): "cuda" oder "cpu".
    
    Returns:
        nn.Module: Das trainierte Student-Modell.
    """
    teacher_model.to(device)
    student_model.to(device)
    
    # Teacher-Modell bleibt fixiert
    teacher_model.eval()
    criterion_ce = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        student_model.train()
        running_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Student Training - Epoch {epoch+1}/{epochs}")
        
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Teacher liefert weiche Zielwerte (Soft Labels)
            with torch.no_grad():
                teacher_logits = teacher_model(inputs)
            
            # Student Vorhersagen
            student_logits = student_model(inputs)
            
            # Berechne die KL-Divergenz
            # Verwende Temperatur-Skalierung, um die Softmax-Verteilungen zu glätten
            teacher_probs = F.softmax(teacher_logits / temperature, dim=1)
            student_log_probs = F.log_softmax(student_logits / temperature, dim=1)
            kl_loss = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * (temperature ** 2)
            
            # Klassische Cross-Entropy
            ce_loss = criterion_ce(student_logits, labels)
            
            # Kombiniere beide Verluste
            loss = alpha * kl_loss + (1 - alpha) * ce_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            progress_bar.set_postfix(loss=f"{running_loss/len(train_loader):.4f}")
        
        logging.info(f"Student - Epoch {epoch+1}/{epochs}: Loss: {running_loss/len(train_loader):.4f}")
        scheduler.step(running_loss/len(train_loader))
    
    return student_model

# Optional: Main-Funktionen für den direkten Aufruf über die Kommandozeile
def train_teacher_main():
    import argparse
    from src.data_preprocessing import load_vocab, preprocess_texts
    from torch.utils.data import DataLoader, TensorDataset
    from src.models import TeacherNetText
    from src.utils import load_config, setup_logging, save_model, get_device

    parser = argparse.ArgumentParser(description="Training des Teacher-Modells")
    parser.add_argument("--epochs", type=int, default=10, help="Anzahl der Trainings-Epochen")
    args = parser.parse_args()

    setup_logging()
    config = load_config()
    device = get_device()

    # Beispiel: Dummy-Daten generieren
    sample_texts = ["I love this movie", "This film was terrible", "Fantastic storyline", "Not worth watching"]
    sample_labels = [1, 0, 1, 0]
    
    vocab = load_vocab(config["data"]["vocab_path"])
    sequences = preprocess_texts(sample_texts, vocab, max_length=config["data"]["max_sequence_length"])
    sequences = torch.tensor(sequences, dtype=torch.long)
    labels = torch.tensor(sample_labels, dtype=torch.long)

    dataset = TensorDataset(sequences, labels)
    train_loader = DataLoader(dataset, batch_size=config["data"]["batch_size"], shuffle=True)
    
    teacher_model = TeacherNetText(len(vocab), **config["model"]["teacher"]).to(device)
    optimizer = torch.optim.Adam(teacher_model.parameters(), lr=config["training"]["learning_rate"])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

    teacher_model = train_teacher(teacher_model, train_loader, train_loader, optimizer, scheduler, epochs=args.epochs, device=device)
    save_model(teacher_model, config["distillation"]["teacher_path"])

def train_student_main():
    import argparse
    from src.data_preprocessing import load_vocab, preprocess_texts
    from torch.utils.data import DataLoader, TensorDataset
    from src.models import TeacherNetText, StudentNetText
    from src.utils import load_config, setup_logging, save_model, load_model, get_device

    parser = argparse.ArgumentParser(description="Training des Student-Modells mittels Knowledge Distillation")
    parser.add_argument("--epochs", type=int, default=10, help="Anzahl der Trainings-Epochen")
    args = parser.parse_args()

    setup_logging()
    config = load_config()
    device = get_device()

    vocab = load_vocab(config["data"]["vocab_path"])
    
    # Beispiel: Dummy-Daten generieren
    sample_texts = ["I love this movie", "This film was terrible", "Fantastic storyline", "Not worth watching"]
    sample_labels = [1, 0, 1, 0]
    
    sequences = preprocess_texts(sample_texts, vocab, max_length=config["data"]["max_sequence_length"])
    sequences = torch.tensor(sequences, dtype=torch.long)
    labels = torch.tensor(sample_labels, dtype=torch.long)

    dataset = TensorDataset(sequences, labels)
    train_loader = DataLoader(dataset, batch_size=config["data"]["batch_size"], shuffle=True)

    teacher_model = TeacherNetText(len(vocab), **config["model"]["teacher"]).to(device)
    teacher_model = load_model(teacher_model, config["distillation"]["teacher_path"], device)
    
    student_model = StudentNetText(len(vocab), **config["model"]["student"]).to(device)
    optimizer = torch.optim.Adam(student_model.parameters(), lr=config["training"]["learning_rate"])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

    student_model = train_student(teacher_model, student_model, train_loader, train_loader, optimizer, scheduler,
                                  epochs=args.epochs, temperature=config["distillation"]["temperature"],
                                  alpha=config["distillation"]["alpha"], device=device)
    save_model(student_model, config["distillation"]["student_path"])

if __name__ == "__main__":
    # Zum direkten Aufruf kann man zwischen den Trainingsmodi wählen.
    parser = argparse.ArgumentParser(description="Trainingsskript für DistillNLP")
    parser.add_argument("--mode", type=str, choices=["teacher", "student"], required=True,
                        help="Wähle 'teacher' oder 'student' zum Trainieren des entsprechenden Modells")
    parser.add_argument("--epochs", type=int, default=10, help="Anzahl der Trainings-Epochen")
    args = parser.parse_args()

    if args.mode == "teacher":
        train_teacher_main()
    else:
        train_student_main()



"""
Wie wird das Skript verwendet?
Zum Training des Teacher-Modells:
python training.py --mode teacher --epochs 10
Das Skript trainiert dann das Teacher-Modell und speichert es unter dem in der Konfiguration angegebenen Pfad.

Zum Training des Student-Modells mittels Knowledge Distillation:
python training.py --mode student --epochs 10
Hierbei wird zuerst das vortrainierte Teacher-Modell geladen und dann das Student-Modell mittels Knowledge Distillation trainiert.

"""