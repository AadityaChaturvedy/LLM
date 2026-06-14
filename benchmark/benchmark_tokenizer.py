"""
Evaluates custom and baseline model tokenizers locally on the same datasets,
under identical normalization, word counting, and coverage protocols.
"""

import os
import sys
import json
import time
import math
import unicodedata
from collections import Counter
from datasets import load_dataset
from tqdm import tqdm

# Setup python path to import src modules
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(BASE_DIR)

from src.custom_bpe import CustomIndicBPE
from src.config import LANGUAGE

# Define paths
VOCAB_PATH  = os.path.join(BASE_DIR, "data/hindi/model-vocab.json")
MERGES_PATH = os.path.join(BASE_DIR, "data/hindi/model-merges.txt")

# Word splitter (whitespace-based split after Unicode normalization)
def word_split(text: str) -> list:
    text = unicodedata.normalize("NFC", text)
    return [w for w in text.split() if w]

# Character coverage metric to quantify information loss
def compute_char_coverage(original_line: str, decoded_line: str) -> float:
    orig_chars = [c for c in original_line if not c.isspace()]
    dec_chars = [c for c in decoded_line if not c.isspace()]
    if not orig_chars:
        return 1.0
        
    dec_counts = Counter(dec_chars)
    matched = 0
    for c in orig_chars:
        if dec_counts[c] > 0:
            matched += 1
            dec_counts[c] -= 1
    return matched / len(orig_chars)

# Unified tokenizer wrapper
class TokenizerWrapper:
    def __init__(self, name, tokenizer, is_custom=False):
        self.name = name
        self.tokenizer = tokenizer
        self.is_custom = is_custom
        
        if is_custom:
            self.vocab_size = len(tokenizer.vocab)
            self.unk_id = tokenizer.token_to_id.get("<unk>")
        else:
            self.vocab_size = tokenizer.vocab_size
            self.unk_id = tokenizer.unk_token_id if tokenizer.unk_token_id is not None else -999
            
    def encode(self, text: str) -> list:
        if self.is_custom:
            return self.tokenizer.encode(text).ids
        else:
            return self.tokenizer.encode(text, add_special_tokens=False)
            
    def decode(self, ids: list) -> str:
        if self.is_custom:
            return self.tokenizer.decode(ids)
        else:
            return self.tokenizer.decode(ids)

def evaluate_tokenizer(wrapper: TokenizerWrapper, lines: list, words_per_line: list, total_words: int, total_raw_bytes: int, total_raw_chars: int) -> dict:
    total_tokens = 0
    total_continued = 0
    unk_count = 0
    empty_lines = 0
    total_coverage = 0.0
    
    # Run tokenization metrics collection
    for line, words in zip(lines, words_per_line):
        ids = wrapper.encode(line)
        decoded = wrapper.decode(ids)
        
        if len(ids) == 0:
            empty_lines += 1
            
        total_tokens += len(ids)
        unk_count += ids.count(wrapper.unk_id)
        total_coverage += compute_char_coverage(line, decoded)
        
        for w in words:
            w_ids = wrapper.encode(w)
            if len(w_ids) > 1:
                total_continued += 1
                
    # Run speed measurement separately (to exclude word-split BPE overhead)
    start_speed = time.time()
    for line in lines:
        _ = wrapper.encode(line)
    elapsed_speed = time.time() - start_speed
    
    lines_sec = len(lines) / elapsed_speed if elapsed_speed > 0 else 0
    mb_sec = (total_raw_bytes / (1024 * 1024)) / elapsed_speed if elapsed_speed > 0 else 0
    
    return {
        "name": wrapper.name,
        "vocab_size": wrapper.vocab_size,
        "fertility": total_tokens / total_words if total_words > 0 else 0,
        "bpt": total_raw_bytes / total_tokens if total_tokens > 0 else 0,
        "chars_token": total_raw_chars / total_tokens if total_tokens > 0 else 0,
        "cont_pct": (total_continued / total_words) * 100 if total_words > 0 else 0,
        "unk_rate": (unk_count / total_tokens) * 100 if total_tokens > 0 else 0,
        "empty_rate": (empty_lines / len(lines)) * 100 if lines else 0,
        "coverage": (total_coverage / len(lines)) * 100 if lines else 0,
        "speed_ls": lines_sec,
        "speed_mbs": mb_sec
    }

def print_table(dataset_name: str, results: list):
    print(f"\nDataset: {dataset_name}")
    print("Normalization: NFC")
    print("Special tokens counted: no")
    print("Word split: whitespace after Unicode normalization")
    print(f"{'─'*126}")
    header_fmt = "{:<16} {:>10} {:>10} {:>8} {:>12} {:>8} {:>8} {:>8} {:>10} {:>12} {:>12}"
    print(header_fmt.format(
        "Tokenizer", "Vocab Size", "Fertility", "BPT", "Chars/Token", "UNK%", "Cont%*", "Empty%", "Char Cov%", "Speed (L/s)", "Speed(MB/s)"
    ))
    print(f"{'─'*126}")
    for r in results:
        print(header_fmt.format(
            r["name"],
            f"{r['vocab_size']:,}",
            f"{r['fertility']:.4f}",
            f"{r['bpt']:.2f}",
            f"{r['chars_token']:.2f}",
            f"{r['unk_rate']:.4f}%",
            f"{r['cont_pct']:.2f}%",
            f"{r['empty_rate']:.2f}%",
            f"{r['coverage']:.2f}%",
            f"{r['speed_ls']:.1f}",
            f"{r['speed_mbs']:.2f}"
        ))
    print(f"{'─'*126}")
    print("* Note: Some baseline tokenizers (e.g. Krutrim-2) may be omitted if they fail to load due to gating or network restrictions.")
    print("* Note on Continuation %: Computed via word-isolated BPE encoding. Isolated word tokenization can differ from in-context tokenization.")
    print(f"{'─'*126}\n")

def print_reference_scores():
    print(f"{'═'*70}")
    print(" REFERENCE ONLY: Published paper scores (MUTANT paper)")
    print(" (Not directly comparable due to differences in corpus/eval protocols)")
    print(f"{'═'*70}")
    print("  MUTANT Custom Indic Corpus Hindi Fertility (Table 3):")
    print("    MUTANT-Indic : 1.23")
    print("    Gemma-3      : 1.47")
    print("    Sarvam-2B    : 1.53")
    print("    Sutra        : 1.62")
    print("    GPT-OSS      : 1.72")
    print("    LLaMA-4      : 1.83")
    print("  Eval corpus: MUTANT's own custom Indic dataset (Table 3 in paper)")
    print("  NOT evaluated on FLORES-200 or XL-Sum — not directly comparable")
    print(f"{'═'*70}\n")

def main():
    # ── 1. Initializing Tokenizers ───────────────────────────────────────────
    print("Initializing tokenizers ...")
    tokenizers = []
    
    # Custom BPE Tokenizer
    try:
        custom_tok = CustomIndicBPE(devanagari_only=(LANGUAGE == "hindi"))
        custom_tok.load(VOCAB_PATH, MERGES_PATH)
        tokenizers.append(TokenizerWrapper("Your Tokenizer", custom_tok, is_custom=True))
        print("  Loaded Custom BPE.")
    except Exception as e:
        print(f"  Failed to load Custom BPE: {e}")
        
    # Baseline Tokenizers
    from transformers import AutoTokenizer
    baselines = {
        "Gemma-2": "unsloth/gemma-2-9b",
        "LLaMA-3": "unsloth/llama-3-8b",
        "Qwen-2.5": "Qwen/Qwen2.5-7B",
        "Mistral-v0.3": "unsloth/mistral-7b-v0.3",
        "Sarvam-1": "sarvamai/sarvam-1",
        "Krutrim-2": "krutrim-ai-labs/Krutrim-2-Instruct-0131",
        "MuRIL": "google/muril-base-cased"
    }
    
    for name, path in baselines.items():
        try:
            print(f"  Loading {name} ({path}) ...")
            t = AutoTokenizer.from_pretrained(path)
            tokenizers.append(TokenizerWrapper(name, t))
        except Exception as e:
            print(f"  Failed to load {name} tokenizer: {e}")
            
    print(f"Initialized {len(tokenizers)} tokenizers for evaluation.\n")
    
    # ── 2. Loading Datasets ──────────────────────────────────────────────────
    datasets = {}
    
    # Dataset 1: FLORES-200
    print("Loading FLORES-200 Hindi validation split ...")
    flores_lines = []
    try:
        print("  Trying openlanguagedata/flores_plus (dev) ...")
        ds_flores = load_dataset("openlanguagedata/flores_plus", "hin_Deva", split="dev")
        flores_lines = [row["sentence"] for row in ds_flores]
    except Exception as e:
        print(f"  Could not load official flores_plus (gating or auth required): {e}")
        try:
            print("  Trying facebook/flores (dev) ...")
            ds_flores = load_dataset("facebook/flores", "hin_Deva", split="dev")
            flores_lines = [row["sentence"] for row in ds_flores]
        except Exception as e2:
            print(f"  Could not load facebook/flores: {e2}")
            try:
                print("  Falling back to un-gated mirror: tomasmajercik/flores-parquet ...")
                ds_flores = load_dataset("tomasmajercik/flores-parquet", "hin_Deva", split="validation")
                flores_lines = [row["sentence"] for row in ds_flores]
            except Exception as e3:
                print(f"  Failed to load FLORES-200 mirror: {e3}")
    if flores_lines:
        datasets["FLORES-200 Hindi Validation"] = flores_lines
        print(f"  Loaded {len(flores_lines)} lines for FLORES-200.")
        
    # Dataset 2: XL-Sum Test Parquet
    print("Loading XL-Sum Hindi test split (Parquet) ...")
    try:
        url = "https://huggingface.co/datasets/csebuetnlp/xlsum/resolve/refs%2Fconvert%2Fparquet/hindi/test/0000.parquet"
        ds_xl = load_dataset("parquet", data_files=url, split="train")
        xl_lines = [row["text"] for row in ds_xl][:1000] # Use first 1000 lines
        datasets["XL-Sum Hindi Test (1k lines)"] = xl_lines
        print(f"  Loaded {len(xl_lines)} lines.")
    except Exception as e:
        print(f"  Failed to load XL-Sum Parquet: {e}")
        
    # Dataset 3: SQuAD Hindi (Target domain)
    print("Loading SQuAD Hindi validation contexts ...")
    try:
        squad_path = os.path.join(BASE_DIR, "data/hindi_squad_large.jsonl")
        if os.path.exists(squad_path):
            seen_ctx = set()
            squad_lines = []
            with open(squad_path, "r", encoding="utf-8") as f:
                for line in f:
                    row = json.loads(line)
                    ctx = row.get("context", "").strip()
                    if ctx and ctx not in seen_ctx:
                        seen_ctx.add(ctx)
                        squad_lines.append(ctx)
                        if len(squad_lines) >= 1000: # Use first 1000 unique contexts
                            break
            datasets["SQuAD Hindi Contexts (1k unique)"] = squad_lines
            print(f"  Loaded {len(squad_lines)} unique contexts.")
        else:
            print(f"  SQuAD file not found at {squad_path}")
    except Exception as e:
        print(f"  Failed to load SQuAD Hindi contexts: {e}")
        
    print(f"\nLoaded {len(datasets)} datasets for benchmarking.\n")
    
    # ── 3. Evaluating ────────────────────────────────────────────────────────
    for ds_name, raw_lines in datasets.items():
        print(f"Benchmarking on {ds_name} ...")
        # Unicode Normalization (NFC) applied globally
        lines = [unicodedata.normalize("NFC", line).strip() for line in raw_lines]
        lines = [line for line in lines if line]
        
        # Word counts pre-computed
        words_per_line = [word_split(line) for line in lines]
        total_words = sum(len(w) for w in words_per_line)
        total_raw_bytes = sum(len(line.encode("utf-8")) for line in lines)
        total_raw_chars = sum(len(line) for line in lines)
        
        results = []
        for wrapper in tokenizers:
            res = evaluate_tokenizer(wrapper, lines, words_per_line, total_words, total_raw_bytes, total_raw_chars)
            results.append(res)
            
        print_table(ds_name, results)
        
    print_reference_scores()

if __name__ == "__main__":
    main()
