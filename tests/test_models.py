import unittest
import torch
from src.models import TeacherNetText, StudentNetText, count_parameters

class TestModels(unittest.TestCase):
    def setUp(self):
        """Initialisiert Parameter und Modellinstanzen für die Tests."""
        self.vocab_size = 5000
        self.num_classes = 2
        self.batch_size = 4
        self.sequence_length = 20

        # Initialisierung des Teacher-Modells
        self.teacher_model = TeacherNetText(
            vocab_size=self.vocab_size,
            embed_dim=256,
            hidden_dim=512,
            num_classes=self.num_classes,
            dropout=0.3
        )
        # Initialisierung des Student-Modells
        self.student_model = StudentNetText(
            vocab_size=self.vocab_size,
            embed_dim=128,
            num_classes=self.num_classes
        )

    def test_teacher_forward_shape(self):
        """Testet den Forward-Pass des Teacher-Modells und dessen Ausgabeform."""
        inputs = torch.randint(0, self.vocab_size, (self.batch_size, self.sequence_length))
        outputs = self.teacher_model(inputs)
        self.assertEqual(outputs.shape, (self.batch_size, self.num_classes),
                         "Teacher-Modell: Die Ausgabedimensionen stimmen nicht.")

    def test_student_forward_shape(self):
        """Testet den Forward-Pass des Student-Modells und dessen Ausgabeform."""
        inputs = torch.randint(0, self.vocab_size, (self.batch_size, self.sequence_length))
        outputs = self.student_model(inputs)
        self.assertEqual(outputs.shape, (self.batch_size, self.num_classes),
                         "Student-Modell: Die Ausgabedimensionen stimmen nicht.")

    def test_parameter_counts(self):
        """Überprüft, dass das Teacher-Modell mehr Parameter hat als das Student-Modell."""
        teacher_params = count_parameters(self.teacher_model)
        student_params = count_parameters(self.student_model)
        # Das Teacher-Modell sollte in der Regel komplexer sein
        self.assertGreater(teacher_params, student_params,
                           "Das Teacher-Modell sollte mehr Parameter haben als das Student-Modell.")

    def test_deterministic_forward_in_eval_mode(self):
        """
        Testet, dass im Evaluierungsmodus (eval) bei gleicher Eingabe konsistente Ausgaben erzeugt werden.
        (Dropout sollte deaktiviert sein.)
        """
        self.teacher_model.eval()  # Dropout deaktivieren
        inputs = torch.randint(0, self.vocab_size, (self.batch_size, self.sequence_length))
        output1 = self.teacher_model(inputs)
        output2 = self.teacher_model(inputs)
        self.assertTrue(torch.allclose(output1, output2, atol=1e-5),
                        "Im Evaluierungsmodus sollten die Ausgaben deterministisch sein.")

if __name__ == "__main__":
    unittest.main()
