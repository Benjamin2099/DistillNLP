import unittest
import torch
from torch.utils.data import DataLoader, TensorDataset
from src.models import TeacherNetText, StudentNetText
from src.training import train_teacher, train_student
from src.evaluation import evaluate_model
from src.utils import get_device

class DummyDataset(torch.utils.data.Dataset):
    """
    Ein Dummy-Dataset zur Simulation von Trainings- und Validierungsdaten.
    Es erzeugt zufällige Eingabetensoren und binäre Labels.
    """
    def __init__(self, num_samples=20, seq_length=20, vocab_size=5000):
        self.num_samples = num_samples
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        self.data = torch.randint(0, vocab_size, (num_samples, seq_length))
        self.labels = torch.randint(0, 2, (num_samples,))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

class TestTrainingFunctions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.device = get_device()
        cls.vocab_size = 5000
        cls.num_classes = 2
        cls.seq_length = 20
        cls.batch_size = 4
        cls.epochs = 2  # Für Testzwecke nur wenige Epochen
        # Erstelle ein Dummy-Dataset und entsprechende DataLoader
        dataset = DummyDataset(num_samples=20, seq_length=cls.seq_length, vocab_size=cls.vocab_size)
        cls.train_loader = DataLoader(dataset, batch_size=cls.batch_size, shuffle=True)
        cls.val_loader = DataLoader(dataset, batch_size=cls.batch_size, shuffle=False)

    def test_train_teacher(self):
        """Testet, ob das Teacher-Modell über den Trainingsprozess korrekt trainiert wird."""
        teacher_model = TeacherNetText(self.vocab_size, embed_dim=256, hidden_dim=512, num_classes=self.num_classes, dropout=0.3).to(self.device)
        optimizer = torch.optim.Adam(teacher_model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
        
        # Training des Teacher-Modells
        trained_teacher = train_teacher(teacher_model, self.train_loader, self.val_loader, optimizer, scheduler, epochs=self.epochs, device=self.device)
        self.assertIsInstance(trained_teacher, torch.nn.Module)
        
        # Überprüfe, ob das Modell im Evaluierungsmodus ist (z. B. Dropout deaktiviert)
        trained_teacher.eval()
        inputs, _ = next(iter(self.val_loader))
        outputs1 = trained_teacher(inputs.to(self.device))
        outputs2 = trained_teacher(inputs.to(self.device))
        self.assertTrue(torch.allclose(outputs1, outputs2, atol=1e-5), "Die Ausgaben im Evaluierungsmodus sollten deterministisch sein.")

    def test_train_student(self):
        """Testet das Training des Student-Modells mittels Knowledge Distillation."""
        # Initialisiere Teacher- und Student-Modelle
        teacher_model = TeacherNetText(self.vocab_size, embed_dim=256, hidden_dim=512, num_classes=self.num_classes, dropout=0.3).to(self.device)
        student_model = StudentNetText(self.vocab_size, embed_dim=128, num_classes=self.num_classes).to(self.device)
        
        # Dummy-Training: Trainiere den Teacher kurz, damit er als Fixpunkt dient
        teacher_optimizer = torch.optim.Adam(teacher_model.parameters(), lr=0.001)
        teacher_scheduler = torch.optim.lr_scheduler.StepLR(teacher_optimizer, step_size=1, gamma=0.5)
        teacher_model = train_teacher(teacher_model, self.train_loader, self.val_loader, teacher_optimizer, teacher_scheduler, epochs=self.epochs, device=self.device)
        teacher_model.eval()

        student_optimizer = torch.optim.Adam(student_model.parameters(), lr=0.001)
        student_scheduler = torch.optim.lr_scheduler.StepLR(student_optimizer, step_size=1, gamma=0.5)
        
        trained_student = train_student(teacher_model, student_model, self.train_loader, self.val_loader,
                                        student_optimizer, student_scheduler, epochs=self.epochs,
                                        temperature=2.0, alpha=0.7, device=self.device)
        self.assertIsInstance(trained_student, torch.nn.Module)

    def test_evaluate_model(self):
        """Testet die Evaluierung des Modells und die Rückgabe der Metriken."""
        # Erstelle ein Dummy-Modell (Student-Modell als Beispiel)
        model = StudentNetText(self.vocab_size, embed_dim=128, num_classes=self.num_classes).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
        # Führe ein kurzes Training durch, um das Modell leicht anzupassen
        model = train_teacher(model, self.train_loader, self.val_loader, optimizer, scheduler, epochs=self.epochs, device=self.device)
        
        eval_metrics = evaluate_model(model, self.val_loader, device=self.device)
        # Überprüfe, ob alle erwarteten Metriken vorhanden sind
        self.assertIn("accuracy", eval_metrics)
        self.assertIn("precision", eval_metrics)
        self.assertIn("recall", eval_metrics)
        self.assertIn("f1_score", eval_metrics)
        self.assertIn("confusion_matrix", eval_metrics)
        print("Evaluierungsmesswerte:", eval_metrics)

if __name__ == "__main__":
    unittest.main()
