import os
import torch
import torch.nn.functional as F
from tokenizers import ByteLevelBPETokenizer

from src.config import (
    max_new_tokens, temperature, top_k,
    vocab_size, embedding_dim, context_length,
    num_layers, num_heads, d_model, hidden_dim_ffn,
    TOKENIZER_VOCAB_PATH, TOKENIZER_MERGES_PATH
)
from src.model import GPT

@torch.no_grad()
def generate(model, tokenizer, prompt, max_new_tokens, temperature, top_k, device):
    model.eval()
    
    # Encode prompt
    encoded = tokenizer.encode(prompt)
    x = torch.tensor([encoded.ids], dtype=torch.long, device=device) 
    
    print(f"\n--- Prompt: {prompt} ---")
    print(prompt, end="", flush=True)
    
    for _ in range(max_new_tokens):

        x_cond = x[:, -context_length:]
        
        logits = model(x_cond) 
        logits = logits[:, -1, :] 
        
        if temperature > 0.0:
            logits = logits / temperature
            
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
                
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
        else:
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
            
        x = torch.cat((x, next_token), dim=1)
        
        new_token_str = tokenizer.decode([next_token.item()])
        print(new_token_str, end="", flush=True)
        
    print("\n--- End of Generation ---\n")

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # 1. Load Tokenizer
    if not os.path.exists(TOKENIZER_VOCAB_PATH):
        raise FileNotFoundError("Tokenizer files not found in data/ directory!")
    tokenizer = ByteLevelBPETokenizer(TOKENIZER_VOCAB_PATH, TOKENIZER_MERGES_PATH)
    
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
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    print("Model loaded successfully!")
    
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
                device=device
            )
        except KeyboardInterrupt:
            print("\nExiting generation interface.")
            break

if __name__ == "__main__":
    main()
