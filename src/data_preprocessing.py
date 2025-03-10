import json
import re
import nltk
from nltk.tokenize import word_tokenize
from collections import Counter

# Stelle sicher, dass die benötigten NLTK-Module heruntergeladen wurden
nltk.download("punkt")

# Sondertokens für das Vokabular
SPECIAL_TOKENS = {"<PAD>": 0, "<UNK>": 1, "<CLS>": 2, "<SEP>": 3, "<MASK>": 4}

def clean_text(text):
    """
    Bereinigt den Eingabetext (Entfernung von Sonderzeichen, Umwandlung in Kleinbuchstaben).
    
    Args:
        text (str): Eingabetext.
    
    Returns:
        str: Bereinigter Text.
    """
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)  # Entferne Sonderzeichen
    return text.strip()

def tokenize(text):
    """
    Tokenisiert den bereinigten Text.
    
    Args:
        text (str): Eingabetext.
    
    Returns:
        list: Liste von Tokens.
    """
    text = clean_text(text)
    return word_tokenize(text)

def build_vocab(texts, min_freq=1):
    """
    Erstellt ein Vokabular basierend auf den Eingabetexten.
    
    Args:
        texts (list of str): Liste von Sätzen.
        min_freq (int): Minimale Frequenz, um ein Wort ins Vokabular aufzunehmen.
    
    Returns:
        dict: Wörterbuch mit Wort-Index-Zuordnung.
    """
    token_counter = Counter()
    for text in texts:
        tokens = tokenize(text)
        token_counter.update(tokens)

    vocab = SPECIAL_TOKENS.copy()
    for word, freq in token_counter.items():
        if freq >= min_freq:
            vocab[word] = len(vocab)

    return vocab

def save_vocab(vocab, filepath="data/vocab.json"):
    import os, json
    # Erstelle das Verzeichnis, falls es nicht existiert
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=4)


def load_vocab(filepath="data/vocab.json"):
    """
    Lädt das Vokabular aus einer JSON-Datei.
    
    Args:
        filepath (str): Pfad zur JSON-Datei.
    
    Returns:
        dict: Wörterbuch mit Wort-Index-Zuordnung.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    return vocab

def text_to_sequence(text, vocab):
    """
    Wandelt einen Text in eine Sequenz numerischer Token-Indizes um.
    
    Args:
        text (str): Eingabetext.
        vocab (dict): Wörterbuch mit Wort-Index-Zuordnung.
    
    Returns:
        list: Liste numerischer Token-Indizes.
    """
    tokens = tokenize(text)
    return [vocab.get(token, vocab["<UNK>"]) for token in tokens]

def pad_sequence(seq, max_length, pad_value=0):
    """
    Füllt eine Sequenz auf die gewünschte Länge auf oder kürzt sie.
    
    Args:
        seq (list): Liste von Token-Indizes.
        max_length (int): Ziel-Länge der Sequenz.
        pad_value (int): Padding-Wert (standardmäßig 0 für "<PAD>").
    
    Returns:
        list: Gepaddete oder gekürzte Sequenz.
    """
    if len(seq) < max_length:
        return seq + [pad_value] * (max_length - len(seq))
    return seq[:max_length]

def preprocess_texts(texts, vocab, max_length=128):
    """
    Führt eine vollständige Vorverarbeitung durch: Tokenisierung, Vokabular-Mapping und Padding.
    
    Args:
        texts (list of str): Liste von Sätzen.
        vocab (dict): Wörterbuch mit Wort-Index-Zuordnung.
        max_length (int): Maximale Sequenzlänge.
    
    Returns:
        list: Liste von gepaddeten Sequenzen.
    """
    sequences = [text_to_sequence(text, vocab) for text in texts]
    return [pad_sequence(seq, max_length) for seq in sequences]

# Beispielverwendung
if __name__ == "__main__":
    sample_texts = [
        "I love this movie!",
        "The acting was terrible...",
        "A fantastic experience!",
        "Not worth watching."
    ]
    
    # Vokabular generieren
    vocab = build_vocab(sample_texts, min_freq=1)
    save_vocab(vocab)  # Speichert es als JSON
    
    # Laden des gespeicherten Vokabulars
    vocab = load_vocab()
    
    # Verarbeitung der Texte
    processed_sequences = preprocess_texts(sample_texts, vocab, max_length=10)
    for text, seq in zip(sample_texts, processed_sequences):
        print(f"Original: {text}\nTokenized & Padded: {seq}\n")
