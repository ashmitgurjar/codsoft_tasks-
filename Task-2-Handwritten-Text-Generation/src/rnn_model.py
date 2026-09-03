import torch
# pyrefly: ignore [missing-import]
import torch.nn as nn
from typing import Tuple, Optional, Union


class CharRNN(nn.Module):
    """Character-Level Recurrent Neural Network (LSTM / GRU / Vanilla RNN)."""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 2,
        rnn_type: str = "LSTM",
        dropout: float = 0.2,
    ):
        super(CharRNN, self).__init__()
        
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.rnn_type = rnn_type.upper()
        self.dropout_p = dropout

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        # Recurrent layer backbone selection
        rnn_dropout = dropout if num_layers > 1 else 0.0
        if self.rnn_type == "LSTM":
            self.rnn = nn.LSTM(
                embed_dim,
                hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=rnn_dropout,
            )
        elif self.rnn_type == "GRU":
            self.rnn = nn.GRU(
                embed_dim,
                hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=rnn_dropout,
            )
        elif self.rnn_type == "RNN":
            self.rnn = nn.RNN(
                embed_dim,
                hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=rnn_dropout,
            )
        else:
            raise ValueError(f"Unsupported rnn_type: {rnn_type}. Choose from 'LSTM', 'GRU', 'RNN'.")

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]] = None,
    ) -> Tuple[torch.Tensor, Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass.
        x shape: (batch_size, seq_len)
        Returns logits (batch_size, seq_len, vocab_size) and updated hidden state.
        """
        # Embed token indices: (batch_size, seq_len, embed_dim)
        embedded = self.embedding(x)
        
        # Pass through Recurrent Backbone
        out, hidden = self.rnn(embedded, hidden)
        
        # Apply Dropout and linear output projection
        out = self.dropout(out)
        logits = self.fc(out)
        
        return logits, hidden

    def init_hidden(
        self, batch_size: int = 1, device: Optional[torch.device] = None
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Initialize zero state for hidden layers."""
        if device is None:
            device = next(self.parameters()).device
            
        if self.rnn_type == "LSTM":
            h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
            c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
            return (h0, c0)
        else:
            h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
            return h0

    def get_num_params(self) -> int:
        """Return total trainable parameter count."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
