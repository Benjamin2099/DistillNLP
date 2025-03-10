import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_model(model, data_loader, device="cpu"):
    """
    Evaluiert das gegebene Modell anhand der übergebenen Daten und berechnet Metriken wie Accuracy,
    Precision, Recall, F1-Score und die Konfusionsmatrix.
    
    Args:
        model (torch.nn.Module): Das zu evaluierende Modell.
        data_loader (DataLoader): DataLoader mit den Evaluierungsdaten.
        device (str): "cpu" oder "cuda", je nachdem, wo das Modell evaluiert werden soll.
    
    Returns:
        dict: Enthält die berechneten Metriken:
            - accuracy: Genauigkeit
            - precision: Präzision (gewichteter Durchschnitt)
            - recall: Recall (gewichteter Durchschnitt)
            - f1_score: F1-Score (gewichteter Durchschnitt)
            - confusion_matrix: Die Konfusionsmatrix als NumPy-Array
    """
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1_score, _ = precision_recall_fscore_support(all_labels, all_preds, average="weighted")
    conf_matrix = confusion_matrix(all_labels, all_preds)
    
    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "confusion_matrix": conf_matrix
    }
    
    return metrics

def plot_confusion_matrix(conf_matrix, class_names=None, title="Confusion Matrix", cmap="Blues"):
    """
    Erstellt eine Heatmap der Konfusionsmatrix.

    Args:
        conf_matrix (numpy.ndarray): Die Konfusionsmatrix.
        class_names (list, optional): Liste der Klassennamen. Standardmäßig werden numerische Indizes angezeigt.
        title (str): Titel der Grafik.
        cmap (str): Farbkarte für die Heatmap.
    """
    plt.figure(figsize=(6, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap=cmap,
                xticklabels=class_names if class_names else range(conf_matrix.shape[1]),
                yticklabels=class_names if class_names else range(conf_matrix.shape[0]))
    plt.xlabel("Vorhergesagte Klasse")
    plt.ylabel("Tatsächliche Klasse")
    plt.title(title)
    plt.show()

if __name__ == "__main__":
    # Beispiel: Dummy-Evaluation, falls du kein reales Dataset zur Hand hast.
    # Erstelle Dummy-Daten:
    from torch.utils.data import DataLoader, TensorDataset
    dummy_inputs = torch.randint(0, 5000, (32, 20))
    dummy_labels = torch.randint(0, 2, (32,))
    dummy_dataset = TensorDataset(dummy_inputs, dummy_labels)
    dummy_loader = DataLoader(dummy_dataset, batch_size=8, shuffle=False)
    
    # Dummy-Modell: Einfache lineare Schicht als Platzhalter
    class DummyModel(torch.nn.Module):
        def __init__(self):
            super(DummyModel, self).__init__()
            # Angepasst: Linear-Layer, der 1 Input-Feature erwartet
            self.fc = torch.nn.Linear(1, 2)
        def forward(self, x):
            # Mean-Pooling mit keepdim=True, um die Dimension beizubehalten:
            x = x.float().mean(dim=1, keepdim=True)
            return self.fc(x)
    
    model = DummyModel()
    metrics = evaluate_model(model, dummy_loader, device="cpu")
    print("Evaluierungsergebnisse:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1-Score: {metrics['f1_score']:.4f}")
    print("Confusion Matrix:")
    print(metrics["confusion_matrix"])
    
    # Visualisierung der Konfusionsmatrix
    plot_confusion_matrix(metrics["confusion_matrix"], class_names=["Negativ", "Positiv"])
