import json
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, List, Dict, Optional


class CharVocab:
    """Character Vocabulary manager for encoding and decoding character tokens."""

    def __init__(self, chars: Optional[List[str]] = None):
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        
        if chars is None:
            chars = []
        
        # Build deterministic unique sorted char list
        unique_chars = sorted(list(set(chars)))
        
        # Reserved tokens if needed
        self.idx2char: List[str] = unique_chars
        self.char2idx: Dict[str, int] = {char: idx for idx, char in enumerate(self.idx2char)}

    def __len__(self) -> int:
        return len(self.idx2char)

    def encode(self, text: str) -> List[int]:
        """Convert string of text to list of character integer indices."""
        return [self.char2idx.get(char, 0) for char in text]

    def decode(self, indices: List[int]) -> str:
        """Convert list of character integer indices back to string."""
        return "".join([self.idx2char[idx] if 0 <= idx < len(self.idx2char) else "?" for idx in indices])

    def save(self, filepath: str) -> None:
        """Save vocabulary to JSON file."""
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump({"idx2char": self.idx2char}, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, filepath: str) -> "CharVocab":
        """Load vocabulary from JSON file."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        vocab = cls(chars=data.get("idx2char", []))
        return vocab


class CharDataset(Dataset):
    """PyTorch Dataset for Character-Level RNN Language Modeling."""

    def __init__(self, text: str, seq_len: int = 100, vocab: Optional[CharVocab] = None):
        self.text = text
        self.seq_len = seq_len
        
        if vocab is None:
            self.vocab = CharVocab(list(text))
        else:
            self.vocab = vocab
            
        self.encoded_text = torch.tensor(self.vocab.encode(text), dtype=torch.long)

    def __len__(self) -> int:
        if len(self.encoded_text) <= self.seq_len:
            return 1
        return len(self.encoded_text) - self.seq_len

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if len(self.encoded_text) <= self.seq_len:
            # Handle edge case where text is shorter than sequence length
            chunk = self.encoded_text
            inputs = torch.zeros(self.seq_len, dtype=torch.long)
            targets = torch.zeros(self.seq_len, dtype=torch.long)
            inputs[: len(chunk) - 1] = chunk[:-1]
            targets[: len(chunk) - 1] = chunk[1:]
            return inputs, targets
            
        inputs = self.encoded_text[idx : idx + self.seq_len]
        targets = self.encoded_text[idx + 1 : idx + self.seq_len + 1]
        return inputs, targets


def load_corpus(filepath: str) -> str:
    """Load text data from a file."""
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def create_dataloader(
    text: str,
    seq_len: int = 100,
    batch_size: int = 64,
    shuffle: bool = True,
    vocab: Optional[CharVocab] = None,
) -> Tuple[DataLoader, CharVocab]:
    """Factory function to build DataLoader and CharVocab from raw text string."""
    dataset = CharDataset(text, seq_len=seq_len, vocab=vocab)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=(len(dataset) > batch_size))
    return dataloader, dataset.vocab
