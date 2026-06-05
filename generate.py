import os
import shutil
import textwrap
import torch
import torch.nn.functional as F
from tokenizers import ByteLevelBPETokenizer

from src.custom_tokenizer import CustomTokenizer
from src.config import (
    max_new_tokens, temperature, top_k,
    vocab_size, embedding_dim, context_length,
    num_layers, num_heads, d_model, hidden_dim_ffn,
    TOKENIZER_VOCAB_PATH, TOKENIZER_MERGES_PATH,
    LANGUAGE
)
from src.model import GPT

def wrap_text_for_terminal(text, width):
    lines = text.split('\n')
    wrapped_lines = []
    for line in lines:
        if not line:
            wrapped_lines.append("")
            continue
        chunks = textwrap.wrap(line, width=width, drop_whitespace=False, break_long_words=False)
        if not chunks:
            wrapped_lines.append("")
        else:
            wrapped_lines.extend(chunks)
    return wrapped_lines

@torch.no_grad()
def generate(model, tokenizer, prompt, max_new_tokens, temperature, top_k, top_p, repetition_penalty, device):
    model.eval()
    
    # Encode prompt
    if hasattr(tokenizer, 'encode'):
        encoded = tokenizer.encode(prompt)
        # CustomTokenizer.encode returns an object with .ids
        if hasattr(encoded, 'ids'):
            ids = encoded.ids
        else:
            ids = encoded
    else:
        # Fallback for ByteLevelBPETokenizer if needed
        encoded = tokenizer.encode(prompt)
        ids = encoded.ids
        
    if not ids:
        print(f"\n--- Prompt: {prompt} ---")
        print("Error: The prompt could not be tokenized. It might contain unsupported characters (e.g., English in a Devanagari-only model).")
        print("--- End of Generation ---\n")
        return
    
    x = torch.tensor([ids], dtype=torch.long, device=device) 
    
    print(f"\n--- Prompt: {prompt} ---")
    
    try:
        cols = shutil.get_terminal_size().columns
    except Exception:
        cols = 80
        
    initial_text = tokenizer.decode(x[0].tolist())
    wrapped_lines = wrap_text_for_terminal(initial_text, cols - 2)
    last_line_count = len(wrapped_lines)
    print("\n".join(wrapped_lines), end="", flush=True)
    
    for _ in range(max_new_tokens):

        x_cond = x[:, -context_length:]
        
        logits = model(x_cond) 
        logits = logits[:, -1, :] 
        
        # Apply repetition penalty
        if repetition_penalty != 1.0:
            for token_id in set(x[0].tolist()):
                if logits[0, token_id] < 0:
                    logits[0, token_id] *= repetition_penalty
                else:
                    logits[0, token_id] /= repetition_penalty
        
        if temperature > 0.0:
            logits = logits / temperature
            
            # 1. Top-k filtering
            if top_k is not None and top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
                
            # 2. Top-p (Nucleus) filtering
            if top_p is not None and top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Identify which tokens exceed top_p cumulative threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                # Shift indices right to ensure we keep the first token exceeding the threshold
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = False
                
                # Map mask back to original logits shape
                indices_to_remove = torch.zeros_like(logits, dtype=torch.bool)
                indices_to_remove.scatter_(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = -float('Inf')
                
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
        else:
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
            
            
        x = torch.cat((x, next_token), dim=1)
        
        full_text = tokenizer.decode(x[0].tolist())
        wrapped_lines = wrap_text_for_terminal(full_text, cols - 2)
        if len(wrapped_lines) > last_line_count:
            print('\n' * (len(wrapped_lines) - last_line_count), end="")
            last_line_count = len(wrapped_lines)
        print('\r' + wrapped_lines[-1], end="", flush=True)
        
    print("\n--- End of Generation ---\n")

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # 1. Load Tokenizer
    if not os.path.exists(TOKENIZER_VOCAB_PATH):
        raise FileNotFoundError(f"Tokenizer files not found in {TOKENIZER_VOCAB_PATH}!")
    
    if LANGUAGE in ["hindi", "hinglish"]:
        tokenizer = CustomTokenizer()
        tokenizer.load()
        print(f"Loaded CustomTokenizer for {LANGUAGE}")
    else:
        tokenizer = ByteLevelBPETokenizer(TOKENIZER_VOCAB_PATH, TOKENIZER_MERGES_PATH)
        print("Loaded ByteLevelBPETokenizer")
    
    # 2. Instantiate Model
    model = GPT(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        context_length=context_length,
        num_layers=num_layers,
        num_heads=num_heads,
        d_model=d_model,
        hidden_dim_ffn=hidden_dim_ffn
    )
    
    # 3. Load Checkpoint
    checkpoint_path = "checkpoints/ckpt_step_95000.pt"
    if not os.path.exists(checkpoint_path):
        print(f"Could not find exact checkpoint: {checkpoint_path}")
        available = [f for f in os.listdir("checkpoints") if f.endswith(".pt")]
        if not available:
            print("No checkpoints found in checkpoints/ folder!")
            return
        checkpoint_path = os.path.join("checkpoints", sorted(available)[-1])
        print(f"Defaulting to latest available checkpoint: {checkpoint_path}")
        
    print(f"Loading weights from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Strip "_orig_mod." prefix from state_dict keys if they were saved from a compiled model
    state_dict = checkpoint["model"]
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("_orig_mod."):
            new_state_dict[k[len("_orig_mod.") :]] = v
        else:
            new_state_dict[k] = v
            
    model.load_state_dict(new_state_dict)
    model.to(device)
    print("Model loaded successfully!")
    
    # Default search parameters (nucleus sampling & repetition penalty)
    top_p = 0.9
    repetition_penalty = 1.15
    
    # 4. Interactive Generation Loop
    while True:
        try:
            prompt = input("Enter prompt (or press Ctrl+C to exit): ")
            if not prompt.strip():
                continue
            generate(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                device=device
            )
        except KeyboardInterrupt:
            print("\nExiting generation interface.")
            break

if __name__ == "__main__":
    main()
