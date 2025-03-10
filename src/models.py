import torch
import torch.nn as nn
import torch.nn.functional as F

class TeacherNetText(nn.Module):
    """
    Teacher-Modell für die Textklassifikation.
    
    Dieses Modell nutzt einen Embedding-Layer, ein bidirektionales LSTM und eine 
    Fully Connected Layer, um komplexe kontextuelle Informationen aus den Eingabetexten zu extrahieren.
    
    Parameter:
        vocab_size (int): Größe des Vokabulars.
        embed_dim (int): Dimension der Wort-Embeddings.
        hidden_dim (int): Dimension der LSTM-Hidden States.
        num_classes (int): Anzahl der Ausgabeklassen (z. B. 0 = negativ, 1 = positiv).
        dropout (float): Dropout-Rate zur Regularisierung.
    """
    def __init__(self, vocab_size, embed_dim=256, hidden_dim=512, num_classes=2, dropout=0.3):
        super(TeacherNetText, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        # Bidirektionales LSTM, das die Sequenzinformationen in beide Richtungen verarbeitet
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        # Fully Connected Layer: Da das LSTM bidirektional ist, wird hidden_dim * 2 verwendet.
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Führt den Forward-Pass des Teacher-Modells aus.
        
        Args:
            x (torch.Tensor): Eingabetensor der Form (batch_size, seq_length)
            
        Returns:
            torch.Tensor: Logits der Form (batch_size, num_classes)
        """
        # Embedding: (batch_size, seq_length) -> (batch_size, seq_length, embed_dim)
        x = self.embedding(x)
        # LSTM: (batch_size, seq_length, embed_dim) -> (batch_size, seq_length, hidden_dim*2)
        # Wir nutzen hier nur den letzten Zeitschritt, der die kumulierten Informationen enthält.
        lstm_out, _ = self.lstm(x)
        # Extrahiere den letzten Zeitschritt (Alternativ kann man auch ein Pooling über die Sequenz machen)
        # Da LSTM bidirektional ist, nutzen wir die Ausgabe des letzten Zeitschritts in jeder Richtung
        last_output = lstm_out[:, -1, :]
        last_output = self.dropout(last_output)
        logits = self.fc(last_output)
        return logits

class StudentNetText(nn.Module):
    """
    Student-Modell für die Textklassifikation.
    
    Dieses Modell ist kompakter und nutzt einen Embedding-Layer, gefolgt von einer Mean-Pooling-Schicht, 
    um die Wort-Embeddings über die gesamte Sequenz zu mitteln. Anschließend erfolgt die Klassifikation über eine 
    Fully Connected Layer.
    
    Parameter:
        vocab_size (int): Größe des Vokabulars.
        embed_dim (int): Dimension der Wort-Embeddings.
        num_classes (int): Anzahl der Ausgabeklassen.
    """
    def __init__(self, vocab_size, embed_dim=128, num_classes=2):
        super(StudentNetText, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        """
        Führt den Forward-Pass des Student-Modells aus.
        
        Args:
            x (torch.Tensor): Eingabetensor der Form (batch_size, seq_length)
            
        Returns:
            torch.Tensor: Logits der Form (batch_size, num_classes)
        """
        # Embedding: (batch_size, seq_length) -> (batch_size, seq_length, embed_dim)
        x = self.embedding(x)
        # Mean-Pooling über die Sequenzdimension: (batch_size, seq_length, embed_dim) -> (batch_size, embed_dim)
        x = x.mean(dim=1)
        # Klassifikation: (batch_size, embed_dim) -> (batch_size, num_classes)
        logits = self.fc(x)
        return logits

def count_parameters(model):
    """
    Zählt die Anzahl der trainierbaren Parameter im Modell.

    Args:
        model (nn.Module): Das zu untersuchende Modell.
    
    Returns:
        int: Anzahl der trainierbaren Parameter.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    # Beispiel-Test: Initialisierung und Forward-Pass
    vocab_size = 5000
    num_classes = 2
    batch_size = 4
    seq_length = 20

    # Initialisiere beide Modelle
    teacher_model = TeacherNetText(vocab_size, embed_dim=256, hidden_dim=512, num_classes=num_classes)
    student_model = StudentNetText(vocab_size, embed_dim=128, num_classes=num_classes)

    # Zeige die Anzahl der Parameter
    print(f"Teacher-Modell Parameter: {count_parameters(teacher_model):,}")
    print(f"Student-Modell Parameter: {count_parameters(student_model):,}")

    # Beispiel-Eingabe (zufällige Token-Indizes)
    sample_input = torch.randint(0, vocab_size, (batch_size, seq_length))
    teacher_output = teacher_model(sample_input)
    student_output = student_model(sample_input)

    print("Teacher Output Shape:", teacher_output.shape)  # Erwartet: (batch_size, num_classes)
    print("Student Output Shape:", student_output.shape)  # Erwartet: (batch_size, num_classes)
