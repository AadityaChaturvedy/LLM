import os
import argparse
import torch
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm

from src.custom_tokenizer import CustomTokenizer
from src.config import (
    vocab_size, embedding_dim, context_length,
    num_layers, num_heads, d_model, hidden_dim_ffn, LANGUAGE
)
from src.model import GPT

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_PATH = "sft_checkpoints_instruct/ckpt_instruct_epoch_2.pt"

def load_model_and_tokenizer():
    print(f"Using device: {DEVICE}")
    if LANGUAGE in ["hindi", "hinglish"]:
        tokenizer = CustomTokenizer()
        tokenizer.load()
    else:
        raise ValueError("This benchmark suite is designed for Hindi/Hinglish custom tokenizer.")

    model = GPT(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        context_length=context_length,
        num_layers=num_layers,
        num_heads=num_heads,
        d_model=d_model,
        hidden_dim_ffn=hidden_dim_ffn
    )

    if not os.path.exists(CHECKPOINT_PATH):
        print(f"Checkpoint {CHECKPOINT_PATH} not found. Defaulting to latest.")
        available = sorted([f for f in os.listdir("checkpoints") if f.endswith(".pt")])
        if not available:
            raise FileNotFoundError("No checkpoints found.")
        cp_path = os.path.join("checkpoints", available[-1])
    else:
        cp_path = CHECKPOINT_PATH

    print(f"Loading weights from {cp_path}...")
    checkpoint = torch.load(cp_path, map_location=DEVICE, weights_only=True)
    
    state_dict = checkpoint["model"]
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("_orig_mod."):
            new_state_dict[k[len("_orig_mod.") :]] = v
        else:
            new_state_dict[k] = v
            
    model.load_state_dict(new_state_dict)
    model.to(DEVICE)
    model.eval()
    return model, tokenizer

def arabic_to_devanagari(text):
    if not isinstance(text, str):
        text = str(text)
    mapping = str.maketrans('0123456789', '०१२३४५६७८९')
    return text.translate(mapping)

def get_row_data(row):
    vals = list(row.values())
    question = str(vals[0])
    opts = [str(vals[1]), str(vals[2]), str(vals[3]), str(vals[4])]
    ans_letter = str(vals[5]).strip().upper()
    return question, opts, ans_letter

@torch.no_grad()
def evaluate_mmlu(model, tokenizer, limit=None):
    print("\n--- Downloading/Loading FreedomIntelligence/MMLU_Hindi ---")
    try:
        # Load the streaming dataset
        ds = load_dataset("FreedomIntelligence/MMLU_Hindi", "default", split="test", streaming=True)
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return

    # We know the schema is broken and the first row is used as keys.
    iterator = iter(ds)
    try:
        first_row = next(iterator)
    except StopIteration:
        print("Dataset is empty.")
        return
        
    all_rows = []
    
    print("Caching rows...")
    for row in iterator:
        if limit is not None and len(all_rows) >= limit:
            break
        q, opts, a = get_row_data(row)
        all_rows.append((q, opts, a))
        
    print(f"Loaded {len(all_rows)} questions.")
    
    ans_map = {'A': 'क', 'B': 'ख', 'C': 'ग', 'D': 'घ'}
    options_letters = ["क", "ख", "ग", "घ"]
    
    correct = 0
    total = 0
    
    # Progress bar
    pbar = tqdm(total=len(all_rows), desc="Evaluating MMLU")
    
    for q_text, opts, ans_letter in all_rows:
        if ans_letter not in ans_map:
            pbar.update(1)
            continue
            
        true_ans_hindi = ans_map[ans_letter]
        
        # Build the prompt
        prompt = f"प्रश्न: {q_text}\n"
        for i, opt_text in enumerate(opts):
            prompt += f"{options_letters[i]}) {opt_text}\n"
        prompt += "उत्तर: "
        
        prompt = arabic_to_devanagari(prompt)
        enc_prompt = tokenizer.encode(prompt)
        prompt_ids = enc_prompt.ids if hasattr(enc_prompt, 'ids') else enc_prompt
        
        if len(prompt_ids) == 0:
            pbar.update(1)
            continue
            
        best_loss = float('inf')
        best_option = None
        
        for opt_letter in options_letters:
            enc_opt = tokenizer.encode(opt_letter)
            opt_ids = enc_opt.ids if hasattr(enc_opt, 'ids') else enc_opt
            
            if len(opt_ids) == 0:
                continue
            
            full_ids = prompt_ids + opt_ids
            # Truncate if it exceeds context length
            if len(full_ids) > context_length:
                full_ids = full_ids[-context_length:]
                prompt_len_in_full = len(full_ids) - len(opt_ids)
            else:
                prompt_len_in_full = len(prompt_ids)
                
            if prompt_len_in_full <= 0:
                continue
                
            x = torch.tensor([full_ids[:-1]], dtype=torch.long, device=DEVICE)
            y = torch.tensor([full_ids[1:]], dtype=torch.long, device=DEVICE)
            
            logits, _ = model(x)
            
            start_idx = prompt_len_in_full - 1
            end_idx = start_idx + len(opt_ids)
            
            loss = F.cross_entropy(
                logits[0, start_idx:end_idx],
                y[0, start_idx:end_idx],
                reduction='mean'
            ).item()
            
            if loss < best_loss:
                best_loss = loss
                best_option = opt_letter
                
        if best_option is None:
            pbar.update(1)
            continue
                
        if best_option == true_ans_hindi:
            correct += 1
            
        total += 1
        pbar.update(1)
        
        # Update progress bar description with running accuracy
        if total % 10 == 0:
            pbar.set_description(f"MMLU Acc: {(correct/total)*100:.1f}%")
            
    pbar.close()
    
    if total > 0:
        accuracy = (correct / total) * 100
        print(f"\n========================================================")
        print(f"                 MMLU HINDI RESULTS                     ")
        print(f"========================================================")
        print(f"Questions Evaluated : {total}")
        print(f"Correct Answers     : {correct}")
        print(f"Final Accuracy      : {accuracy:.2f}%")
        print(f"========================================================\n")
    else:
        print("No valid questions evaluated.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=500, help="Number of questions to evaluate (default: 500)")
    args = parser.parse_args()
    
    print("Initializing MMLU Evaluator...")
    model, tokenizer = load_model_and_tokenizer()
    evaluate_mmlu(model, tokenizer, limit=args.limit)
