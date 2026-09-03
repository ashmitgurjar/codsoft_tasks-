import torch
import torch.nn.functional as F
from typing import Optional
from src.rnn_model import CharRNN
from src.dataset import CharVocab


class CharTextGenerator:
    """Text generation engine using trained CharRNN model and sampling techniques."""

    def __init__(self, model: CharRNN, vocab: CharVocab, device: Optional[torch.device] = None):
        self.model = model
        self.vocab = vocab
        
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
        else:
            self.device = device
            
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def generate(
        self,
        prompt: str = "The ",
        num_chars: int = 200,
        temperature: float = 0.8,
        top_k: int = 0,
        top_p: float = 0.0,
    ) -> str:
        """
        Generate continuous text starting from a seed prompt.
        
        Args:
            prompt: Seed string to initialize RNN state.
            num_chars: Number of new characters to synthesize.
            temperature: Sampling temperature (higher = more creative/random, lower = more conservative).
            top_k: If > 0, restrict sampling to top k highest probability characters.
            top_p: If > 0.0, restrict sampling to nucleus of cumulative probability top_p.
            
        Returns:
            Generated text string including the prompt.
        """
        if not prompt:
            prompt = "A"
            
        # Encode seed prompt
        prompt_indices = self.vocab.encode(prompt)
        hidden = self.model.init_hidden(batch_size=1, device=self.device)
        
        # Warm-up RNN hidden state with seed prompt
        for idx in prompt_indices[:-1]:
            input_tensor = torch.tensor([[idx]], dtype=torch.long, device=self.device)
            _, hidden = self.model(input_tensor, hidden)
            
        current_idx = prompt_indices[-1]
        generated_chars = list(prompt)
        
        for _ in range(num_chars):
            input_tensor = torch.tensor([[current_idx]], dtype=torch.long, device=self.device)
            logits, hidden = self.model(input_tensor, hidden)
            
            # Take last timestep logits: shape (vocab_size,)
            logits = logits[0, -1, :]
            
            # Temperature scaling
            if temperature <= 0.001:
                # Deterministic argmax
                next_idx = int(torch.argmax(logits).item())
            else:
                logits = logits / temperature
                
                # Apply Top-K filtering
                if top_k > 0:
                    top_k_val = min(top_k, logits.size(-1))
                    indices_to_remove = logits < torch.topk(logits, top_k_val)[0][..., -1, None]
                    logits[indices_to_remove] = float("-inf")
                    
                # Apply Top-P (Nucleus) filtering
                if top_p > 0.0 and top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above top_p threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    # Shift indices to keep first token above threshold
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    logits[indices_to_remove] = float("-inf")
                    
                probs = F.softmax(logits, dim=-1)
                
                # Fallback to categorical multinomial sampling
                if torch.isnan(probs).any() or probs.sum() <= 0:
                    next_idx = int(torch.argmax(logits).item())
                else:
                    next_idx = int(torch.multinomial(probs, num_samples=1).item())
                    
            next_char = self.vocab.idx2char[next_idx] if 0 <= next_idx < len(self.vocab) else "?"
            generated_chars.append(next_char)
            current_idx = next_idx

        return "".join(generated_chars)
