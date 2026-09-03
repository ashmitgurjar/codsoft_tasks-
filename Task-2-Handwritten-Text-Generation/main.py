#!/usr/bin/env python3
"""
Command Line Interface for Character-Level RNN Handwritten Text Generation.
"""

import os
import sys
import argparse
import torch

from src.dataset import load_corpus, CharVocab
from src.rnn_model import CharRNN
from src.trainer import RNNTrainer
from src.text_generator import CharTextGenerator
from src.handwriting_renderer import HandwritingRenderer


def train_command(args):
    """Handle model training CLI command."""
    print("=" * 60)
    print("CHAR-RNN HANDWRITTEN TEXT GENERATION - MODEL TRAINING")
    print("=" * 60)
    
    corpus_path = args.corpus
    if not os.path.exists(corpus_path):
        print(f"Error: Corpus file not found at '{corpus_path}'.")
        sys.exit(1)
        
    print(f"Loading corpus from: {corpus_path}")
    text_corpus = load_corpus(corpus_path)
    print(f"Total corpus character length: {len(text_corpus):,} characters.")
    
    vocab = CharVocab(list(text_corpus))
    print(f"Vocabulary size: {len(vocab)} unique characters.")
    
    model = CharRNN(
        vocab_size=len(vocab),
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        rnn_type=args.rnn_type,
        dropout=args.dropout,
    )
    print(f"Initialized {args.rnn_type} model with {model.get_num_params():,} parameters.")
    
    trainer = RNNTrainer(model=model, vocab=vocab, lr=args.lr, optimizer_type=args.optimizer)
    
    print(f"Starting training for {args.epochs} epochs...")
    def print_progress(epoch, total_epochs, loss, perplexity, sample_text):
        print(f"Epoch [{epoch:02d}/{total_epochs:02d}] - Loss: {loss:.4f} | Perplexity: {perplexity:.2f}")
        print(f"  Sample: \"{sample_text[:80]}...\"")

    trainer.train(
        text_corpus=text_corpus,
        epochs=args.epochs,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        save_dir=args.save_dir,
        sample_prompt=args.prompt,
        progress_callback=print_progress,
    )
    
    print("=" * 60)
    print(f"Training Complete! Checkpoint saved to directory: '{args.save_dir}'")
    print("=" * 60)


def generate_command(args):
    """Handle text generation CLI command."""
    checkpoint_path = os.path.join(args.model_dir, "char_rnn.pt")
    vocab_path = os.path.join(args.model_dir, "vocab.json")
    
    if not os.path.exists(checkpoint_path) or not os.path.exists(vocab_path):
        print(f"Model checkpoint not found in '{args.model_dir}'. Training a lightweight initial model...")
        # Auto train fallback if no checkpoint exists
        corpus_path = "data/handwritten_corpus.txt"
        text_corpus = load_corpus(corpus_path)
        vocab = CharVocab(list(text_corpus))
        model = CharRNN(vocab_size=len(vocab), hidden_dim=128, num_layers=2)
        trainer = RNNTrainer(model=model, vocab=vocab, lr=0.003)
        trainer.train(text_corpus, epochs=15, batch_size=32, save_dir=args.model_dir)
        
    model, vocab = RNNTrainer.load_checkpoint(checkpoint_path, vocab_path)
    generator = CharTextGenerator(model=model, vocab=vocab)
    
    print(f"\n--- Generating Handwritten Text from Prompt: '{args.prompt}' ---")
    generated_text = generator.generate(
        prompt=args.prompt,
        num_chars=args.num_chars,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
    )
    print("\n" + generated_text + "\n")
    
    if args.render:
        renderer = HandwritingRenderer()
        img = renderer.render_page(generated_text, paper_style=args.paper_style, ink_color=args.ink_color)
        out_path = args.output_img
        img.save(out_path)
        print(f"Handwritten visual document rendered and saved to: '{out_path}'")


def render_command(args):
    """Handle rendering existing text file to handwritten image CLI command."""
    text = args.text
    if os.path.exists(text):
        with open(text, "r", encoding="utf-8") as f:
            text = f.read()
            
    renderer = HandwritingRenderer()
    img = renderer.render_page(text, paper_style=args.paper_style, ink_color=args.ink_color)
    img.save(args.output)
    print(f"Handwritten document successfully rendered and saved to: '{args.output}'")


def main():
    parser = argparse.ArgumentParser(description="Character-Level RNN Handwritten Text Generation System")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train Char-RNN model on handwritten text dataset")
    train_parser.add_argument("--corpus", type=str, default="data/handwritten_corpus.txt", help="Path to training corpus file")
    train_parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    train_parser.add_argument("--batch-size", type=int, default=32, help="Mini-batch size")
    train_parser.add_argument("--seq-len", type=int, default=80, help="Sequence chunk length")
    train_parser.add_argument("--embed-dim", type=int, default=128, help="Embedding dimension")
    train_parser.add_argument("--hidden-dim", type=int, default=256, help="RNN hidden dimension")
    train_parser.add_argument("--num-layers", type=int, default=2, help="Number of RNN layers")
    train_parser.add_argument("--rnn-type", type=str, default="LSTM", choices=["LSTM", "GRU", "RNN"], help="RNN architecture")
    train_parser.add_argument("--lr", type=float, default=0.002, help="Learning rate")
    train_parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate")
    train_parser.add_argument("--optimizer", type=str, default="Adam", choices=["Adam", "AdamW"], help="Optimizer type")
    train_parser.add_argument("--save-dir", type=str, default="models", help="Directory to save model checkpoints")
    train_parser.add_argument("--prompt", type=str, default="The ", help="Sample prompt during training")

    # Generate command
    gen_parser = subparsers.add_parser("generate", help="Generate handwritten text from seed prompt")
    gen_parser.add_argument("--prompt", type=str, default="The quick brown", help="Seed prompt for text generation")
    gen_parser.add_argument("--num-chars", type=int, default=250, help="Number of characters to generate")
    gen_parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    gen_parser.add_argument("--top-k", type=int, default=0, help="Top-K sampling")
    gen_parser.add_argument("--top-p", type=float, default=0.0, help="Top-P nucleus sampling")
    gen_parser.add_argument("--model-dir", type=str, default="models", help="Model checkpoint directory")
    gen_parser.add_argument("--render", action="store_true", help="Render output text as handwritten visual image")
    gen_parser.add_argument("--paper-style", type=str, default="ruled", choices=["ruled", "parchment", "clean", "grid"], help="Paper background style")
    gen_parser.add_argument("--ink-color", type=str, default="blue", choices=["blue", "black", "red", "sepia", "green"], help="Ink color")
    gen_parser.add_argument("--output-img", type=str, default="handwritten_output.png", help="Output handwritten image filepath")

    # Render command
    ren_parser = subparsers.add_parser("render", help="Render arbitrary text as visual handwritten document")
    ren_parser.add_argument("--text", type=str, required=True, help="Raw text string or text file path")
    ren_parser.add_argument("--paper-style", type=str, default="ruled", choices=["ruled", "parchment", "clean", "grid"], help="Paper background style")
    ren_parser.add_argument("--ink-color", type=str, default="blue", choices=["blue", "black", "red", "sepia", "green"], help="Ink color")
    ren_parser.add_argument("--output", type=str, default="rendered_handwriting.png", help="Output visual file path")

    args = parser.parse_args()
    
    if args.command == "train":
        train_command(args)
    elif args.command == "generate":
        generate_command(args)
    elif args.command == "render":
        render_command(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
