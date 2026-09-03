import os
import torch
import pytest
from PIL import Image

from src.dataset import CharVocab, CharDataset, create_dataloader
from src.rnn_model import CharRNN
from src.text_generator import CharTextGenerator
from src.handwriting_renderer import HandwritingRenderer
from src.trainer import RNNTrainer


def test_char_vocab():
    text = "Hello World! 123"
    vocab = CharVocab(list(text))
    assert len(vocab) > 0
    encoded = vocab.encode("Hello")
    assert len(encoded) == 5
    decoded = vocab.decode(encoded)
    assert decoded == "Hello"


def test_char_dataset():
    text = "The quick brown fox jumps over the lazy dog."
    dataset = CharDataset(text, seq_len=10)
    assert len(dataset) > 0
    x, y = dataset[0]
    assert x.shape[0] == 10
    assert y.shape[0] == 10


def test_char_rnn_forward():
    vocab_size = 30
    model = CharRNN(vocab_size=vocab_size, embed_dim=16, hidden_dim=32, num_layers=2, rnn_type="LSTM")
    batch_size = 4
    seq_len = 10
    x = torch.randint(0, vocab_size, (batch_size, seq_len))
    logits, hidden = model(x)
    assert logits.shape == (batch_size, seq_len, vocab_size)
    assert model.get_num_params() > 0


def test_char_rnn_gru():
    vocab_size = 25
    model = CharRNN(vocab_size=vocab_size, embed_dim=16, hidden_dim=32, num_layers=1, rnn_type="GRU")
    x = torch.randint(0, vocab_size, (2, 8))
    logits, hidden = model(x)
    assert logits.shape == (2, 8, vocab_size)


def test_text_generator():
    text = "Artificial intelligence and neural networks learn handwriting patterns."
    vocab = CharVocab(list(text))
    model = CharRNN(vocab_size=len(vocab), embed_dim=16, hidden_dim=32, num_layers=1)
    generator = CharTextGenerator(model, vocab)
    generated = generator.generate(prompt="Art", num_chars=20, temperature=0.8)
    assert len(generated) == 23
    assert generated.startswith("Art")


def test_handwriting_renderer():
    renderer = HandwritingRenderer()
    img = renderer.render_page(
        text="Handwritten text generation sample test.",
        paper_style="ruled",
        ink_color="blue",
        font_size=20,
        line_spacing=30,
    )
    assert isinstance(img, Image.Image)
    assert img.size == (900, 1100)


def test_trainer_single_epoch():
    text = "Short text example for training test. Recurrent neural network text generation."
    vocab = CharVocab(list(text))
    model = CharRNN(vocab_size=len(vocab), embed_dim=16, hidden_dim=32, num_layers=1)
    trainer = RNNTrainer(model=model, vocab=vocab, lr=0.01)
    dataloader, _ = create_dataloader(text, seq_len=15, batch_size=2, vocab=vocab)
    loss = trainer.train_epoch(dataloader)
    assert loss > 0.0
