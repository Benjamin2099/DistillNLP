import unittest
import os
import json
from src.data_preprocessing import (
    clean_text,
    tokenize,
    build_vocab,
    text_to_sequence,
    pad_sequence,
    preprocess_texts,
    save_vocab,
    load_vocab
)

class TestDataPreprocessing(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        """Wird einmal vor allen Tests ausgeführt: Erzeugt Beispieltexte und speichert ein Test-Vokabular."""
        cls.sample_texts = [
            "I love this movie!",
            "This film was terrible...",
            "Fantastic acting and great story!",
            "Not worth watching."
        ]
        # Erstelle ein Vokabular aus den Beispielen
        cls.vocab = build_vocab(cls.sample_texts, min_freq=1)
        cls.test_vocab_path = "tests/test_vocab.json"
        save_vocab(cls.vocab, cls.test_vocab_path)

    def test_clean_text(self):
        """Testet die Textbereinigung: Kleinbuchstabenkonvertierung und Entfernen von Sonderzeichen."""
        self.assertEqual(clean_text("Hello, World!"), "hello world")
        self.assertEqual(clean_text("I love Python!!!"), "i love python")
        self.assertEqual(clean_text("12345"), "12345")  # Zahlen sollten erhalten bleiben

    def test_tokenize(self):
        """Testet die Tokenisierung, ob ein Text korrekt in Tokens zerlegt wird."""
        self.assertEqual(tokenize("Hello World!"), ["hello", "world"])
        self.assertEqual(tokenize("Testing, testing, 1 2 3!"), ["testing", "testing", "1", "2", "3"])

    def test_build_vocab(self):
        """Testet, ob das Vokabular korrekt aufgebaut wird und Sondertokens enthalten sind."""
        vocab = build_vocab(["hello world", "hello there"], min_freq=1)
        # Sondertokens sollten vorhanden sein
        self.assertIn("<PAD>", vocab)
        self.assertIn("<UNK>", vocab)
        # Wörter aus den Texten sollten im Vokabular enthalten sein
        self.assertIn("hello", vocab)
        self.assertIn("world", vocab)
        self.assertIn("there", vocab)
        # Überprüfe, dass alle Werte eindeutig sind
        unique_indices = set(vocab.values())
        self.assertEqual(len(unique_indices), len(vocab))
    
    def test_text_to_sequence(self):
        """Testet die Umwandlung eines Textes in eine numerische Sequenz unter Verwendung des Vokabulars."""
        text = "I love this movie!"
        seq = text_to_sequence(text, self.vocab)
        self.assertIsInstance(seq, list)
        # Sicherstellen, dass bekannte Wörter nicht als <UNK> kodiert werden
        for token in tokenize(text):
            self.assertNotEqual(seq[token != "<UNK>"], self.vocab["<UNK>"])

    def test_pad_sequence(self):
        """Testet das Padding von Sequenzen auf eine feste Länge."""
        seq = [1, 2, 3]
        padded_seq = pad_sequence(seq, max_length=5, pad_value=self.vocab["<PAD>"])
        self.assertEqual(len(padded_seq), 5)
        self.assertEqual(padded_seq[-1], self.vocab["<PAD>"])
        # Testen, dass eine zu lange Sequenz gekürzt wird
        trimmed_seq = pad_sequence(seq, max_length=2)
        self.assertEqual(len(trimmed_seq), 2)
        self.assertEqual(trimmed_seq, [1, 2])
    
    def test_preprocess_texts(self):
        """Testet die komplette Vorverarbeitung: Tokenisierung, Umwandlung in Sequenzen und Padding."""
        max_length = 10
        processed = preprocess_texts(self.sample_texts, self.vocab, max_length)
        self.assertEqual(len(processed), len(self.sample_texts))
        for seq in processed:
            self.assertEqual(len(seq), max_length)

    def test_save_and_load_vocab(self):
        """Testet, ob das Vokabular korrekt gespeichert und wieder geladen werden kann."""
        loaded_vocab = load_vocab(self.test_vocab_path)
        self.assertEqual(loaded_vocab, self.vocab)

    @classmethod
    def tearDownClass(cls):
        """Aufräumen nach den Tests: Löscht die Test-Vokabular-Datei."""
        if os.path.exists(cls.test_vocab_path):
            os.remove(cls.test_vocab_path)

if __name__ == "__main__":
    unittest.main()
