import os
import math
import time
import torch
import torch.nn as nn
from torch.optim import Adam, AdamW
from typing import Dict, List, Callable, Optional, Tuple

from src.rnn_model import CharRNN
from src.dataset import CharVocab, create_dataloader
from src.text_generator import CharTextGenerator


class RNNTrainer:
    """Trainer class for Character-Level RNN Text Generation Model."""

    def __init__(
        self,
        model: CharRNN,
        vocab: CharVocab,
        lr: float = 0.002,
        optimizer_type: str = "Adam",
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.vocab = vocab
        
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
        else:
            self.device = device
            
        self.model.to(self.device)
        
        if optimizer_type == "AdamW":
            self.optimizer = AdamW(self.model.parameters(), lr=lr)
        else:
            self.optimizer = Adam(self.model.parameters(), lr=lr)
            
        self.criterion = nn.CrossEntropyLoss()
        
        self.history: Dict[str, List[float]] = {
            "loss": [],
            "perplexity": [],
            "sample_texts": [],
        }

    def train_epoch(
        self, dataloader: torch.utils.data.DataLoader, clip_grad: float = 5.0
    ) -> float:
        """Run a single training epoch across the dataset."""
        self.model.train()
        total_loss = 0.0
        total_batches = 0
        
        for inputs, targets in dataloader:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass: logits shape (batch_size, seq_len, vocab_size)
            logits, _ = self.model(inputs)
            
            # Reshape for CrossEntropyLoss: (batch_size * seq_len, vocab_size) vs (batch_size * seq_len)
            loss = self.criterion(logits.view(-1, self.model.vocab_size), targets.view(-1))
            
            loss.backward()
            
            # Clip gradients to avoid exploding gradient problem in RNNs
            if clip_grad > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), clip_grad)
                
            self.optimizer.step()
            
            total_loss += loss.item()
            total_batches += 1
            
        avg_loss = total_loss / max(1, total_batches)
        return avg_loss

    def train(
        self,
        text_corpus: str,
        epochs: int = 20,
        seq_len: int = 100,
        batch_size: int = 64,
        clip_grad: float = 5.0,
        save_dir: str = "models",
        sample_prompt: str = "The ",
        progress_callback: Optional[Callable[[int, int, float, float, str], None]] = None,
    ) -> Dict[str, List[float]]:
        """
        Train the character RNN model over multiple epochs.
        
        Args:
            text_corpus: Full text data for dataset creation.
            epochs: Total training iterations.
            seq_len: Input sequence chunk length.
            batch_size: DataLoader mini-batch size.
            clip_grad: Gradient clipping norm threshold.
            save_dir: Directory to save model checkpoint and vocab.
            sample_prompt: Seed text prompt for sampling progress monitoring.
            progress_callback: Optional UI callback function (epoch, total_epochs, loss, perplexity, sample_text).
            
        Returns:
            Dictionary containing loss and perplexity history.
        """
        os.makedirs(save_dir, exist_ok=True)
        dataloader, _ = create_dataloader(text_corpus, seq_len=seq_len, batch_size=batch_size, vocab=self.vocab)
        generator = CharTextGenerator(self.model, self.vocab, device=self.device)
        
        start_time = time.time()
        
        for epoch in range(1, epochs + 1):
            loss = self.train_epoch(dataloader, clip_grad=clip_grad)
            perplexity = math.exp(min(loss, 20.0))  # Cap loss for numeric stability
            
            # Generate sample text preview
            sample_text = generator.generate(prompt=sample_prompt, num_chars=120, temperature=0.7)
            
            self.history["loss"].append(loss)
            self.history["perplexity"].append(perplexity)
            self.history["sample_texts"].append(sample_text)
            
            if progress_callback:
                progress_callback(epoch, epochs, loss, perplexity, sample_text)
                
        # Save model weights and vocabulary
        model_save_path = os.path.join(save_dir, "char_rnn.pt")
        vocab_save_path = os.path.join(save_dir, "vocab.json")
        
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "vocab_size": self.model.vocab_size,
                "embed_dim": self.model.embed_dim,
                "hidden_dim": self.model.hidden_dim,
                "num_layers": self.model.num_layers,
                "rnn_type": self.model.rnn_type,
                "dropout": self.model.dropout_p,
            },
            model_save_path,
        )
        self.vocab.save(vocab_save_path)
        
        return self.history

    @classmethod
    def load_checkpoint(cls, checkpoint_path: str, vocab_path: str, device: Optional[torch.device] = None) -> Tuple[CharRNN, CharVocab]:
        """Load trained CharRNN model and vocabulary from saved checkpoints."""
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
            
        vocab = CharVocab.load(vocab_path)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        model = CharRNN(
            vocab_size=checkpoint["vocab_size"],
            embed_dim=checkpoint["embed_dim"],
            hidden_dim=checkpoint["hidden_dim"],
            num_layers=checkpoint["num_layers"],
            rnn_type=checkpoint.get("rnn_type", "LSTM"),
            dropout=checkpoint.get("dropout", 0.2),
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()
        
        return model, vocab
