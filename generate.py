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
    LANGUAGE, use_gqa, num_kv_heads
)
from src.model import GPT

def arabic_to_devanagari(text):
    return text.translate(str.maketrans('0123456789', '०१२३४५६७८९'))

def devanagari_to_arabic(text):
    return text.translate(str.maketrans('०१२३४५६७८९', '0123456789'))

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
    
    prompt_len = x.shape[1]
    print("\nModel Output: ", end="", flush=True)
    printed_len = 0
    
    # --- KV Cache Prefill ---
    # Process the entire prompt in one pass and cache the KV states
    logits, kv_cache = model(x, position_offset=0)
    
    # Get the first token prediction from the last position of the prefill
    next_logits = logits[:, -1, :]
    position = prompt_len  # Next token's position in the sequence
    
    for _ in range(max_new_tokens):
        logits_for_sampling = next_logits
        
        # Apply repetition penalty
        if repetition_penalty != 1.0:
            for token_id in set(x[0].tolist()):
                if logits_for_sampling[0, token_id] < 0:
                    logits_for_sampling[0, token_id] *= repetition_penalty
                else:
                    logits_for_sampling[0, token_id] /= repetition_penalty
        
        if temperature > 0.0:
            logits_for_sampling = logits_for_sampling / temperature
            
            # 1. Top-k filtering
            if top_k is not None and top_k > 0:
                v, _ = torch.topk(logits_for_sampling, min(top_k, logits_for_sampling.size(-1)))
                logits_for_sampling[logits_for_sampling < v[:, [-1]]] = -float('Inf')
                
            # 2. Top-p (Nucleus) filtering
            if top_p is not None and top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits_for_sampling, descending=True, dim=-1)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Identify which tokens exceed top_p cumulative threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                # Shift indices right to ensure we keep the first token exceeding the threshold
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = False
                
                # Map mask back to original logits shape
                indices_to_remove = torch.zeros_like(logits_for_sampling, dtype=torch.bool)
                indices_to_remove.scatter_(1, sorted_indices, sorted_indices_to_remove)
                logits_for_sampling[indices_to_remove] = -float('Inf')
                
            probs = F.softmax(logits_for_sampling, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
        else:
            next_token = torch.argmax(logits_for_sampling, dim=-1, keepdim=True)
            
        eos_id = tokenizer.tokenizer.token_to_id.get("</s>", 2) if hasattr(tokenizer, "tokenizer") else 2
        if next_token.item() == eos_id:
            break
            
        x = torch.cat((x, next_token), dim=1)
        
        # --- KV Cache Decode Step ---
        # Only process the new token, reusing cached KV states from all previous tokens
        next_logits_out, kv_cache = model(next_token, kv_cache=kv_cache, position_offset=position)
        next_logits = next_logits_out[:, -1, :]
        position += 1
        
        generated_text = tokenizer.decode(x[0, prompt_len:].tolist())
        generated_text = devanagari_to_arabic(generated_text)
        
        # Stream only the newly generated characters
        new_text = generated_text[printed_len:]
        print(new_text, end="", flush=True)
        printed_len = len(generated_text)
        
    print("\n")

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
        num_kv_heads=num_kv_heads,
        d_model=d_model,
        hidden_dim_ffn=hidden_dim_ffn,
        use_gqa=use_gqa
    )
    
    # 3. Load Checkpoint
    checkpoint_path = "sft_checkpoints_instruct/ckpt_instruct_epoch_2.pt"
    if not os.path.exists(checkpoint_path):
        print(f"Could not find exact checkpoint: {checkpoint_path}")
        available = [f for f in os.listdir("sft_checkpoints_instruct") if f.endswith(".pt")]
        if not available:
            print("No checkpoints found in sft_checkpoints_instruct/ folder!")
            return
        checkpoint_path = os.path.join("sft_checkpoints_instruct", sorted(available)[-1])
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
            
    model.load_state_dict(new_state_dict, strict=False)
    model.to(device)
    print("Model loaded successfully!")
    
    # Default search parameters (nucleus sampling & repetition penalty)
    top_p = 0.9
    repetition_penalty = 1.15
    
    # 4. Interactive Generation Loop
    while True:
        try:
            print("Enter prompt (Enter an EMPTY line to submit, or press Ctrl+C to exit): ")
            lines = []
            while True:
                line = input()
                if line.strip() == "":
                    break
                lines.append(line)
            prompt = "\n".join(lines).strip()
            
            if not prompt:
                continue
                
            prompt = arabic_to_devanagari(prompt)
            formatted_prompt = f"प्रश्न: {prompt}\nउत्तर: "
            
            generate(
                model=model,
                tokenizer=tokenizer,
                prompt=formatted_prompt,
                max_new_tokens=max_new_tokens,
                temperature=0.1,  # Low temperature is required for Extractive QA
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
