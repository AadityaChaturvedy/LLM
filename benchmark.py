"""
MUTANT-Exact Fertility Benchmark
=================================
Replicates the evaluation methodology from:
  "MUTANT: A Recipe for Multilingual Tokenizer Design"
  Rana et al., ACL 2026 | arXiv:2511.03237

Dataset  : csebuetnlp/xlsum  (Hindi split — same dataset MUTANT paper uses for Hindi)
Metrics  : Fertility, Continuation %, Bytes-per-Token, Rényi Efficiency, Unk Rate

Install  :
    pip install datasets tqdm

Usage    :
    python mutant_benchmark.py
"""

import math
import unicodedata
from collections import Counter
from datasets import load_dataset
from tqdm import tqdm

# ── 0. Import YOUR tokenizer ──────────────────────────────────────────────────
from src.custom_bpe import CustomIndicBPE

VOCAB_PATH  = "data/hindi/model-vocab.json"
MERGES_PATH = "data/hindi/model-merges.txt"

# ── 1. Configuration ──────────────────────────────────────────────────────────
MAX_LINES   = 2000       # MUTANT uses ~1000–2000 lines per language
RENYI_ALPHA = 2.5        # α used in MUTANT for Rényi entropy (§3 of paper)

# ── 2. Word tokenizer (MUTANT uses whitespace split) ──────────────────────────
def word_split(text: str) -> list:
    text = unicodedata.normalize("NFC", text)
    return [w for w in text.split() if w]

# ── 3. Rényi Efficiency ───────────────────────────────────────────────────────
def renyi_efficiency(token_ids: list, alpha: float = 2.5) -> float:
    """
    H_α(P) = (1/(1-α)) * log(Σ p_i^α)
    Efficiency = H_α / log(|unique_tokens|)
    Higher is better (more uniform vocab usage).
    """
    if not token_ids or alpha == 1.0:
        return 0.0
    counts = Counter(token_ids)
    total  = sum(counts.values())
    probs  = [c / total for c in counts.values()]
    sum_pa = sum(p ** alpha for p in probs)
    if sum_pa <= 0:
        return 0.0
    h_renyi = (1.0 / (1.0 - alpha)) * math.log(sum_pa)
    h_max   = math.log(len(counts)) if len(counts) > 1 else 1.0
    return h_renyi / h_max

# ── 4. Load tokenizer ─────────────────────────────────────────────────────────
print("Loading tokenizer …")
tok = CustomIndicBPE()
tok.load(VOCAB_PATH, MERGES_PATH)
unk_id = tok.token_to_id.get("<unk>", 0)
print(f"  vocab={len(tok.vocab):,}  merges={len(tok.merges):,}\n")

# ── 5. Load XL-Sum Hindi from local directory ────────────────────────────────
import os

local_data_path = "/home/aaditya/LLM/hindi_XLSum_v2.0"

# If the folder contains a .jsonl or .json file
print(f"Loading local dataset from: {local_data_path} …")

# Example: If your directory has a 'test.jsonl'
file_path = os.path.join(local_data_path, "hindi_test.jsonl") 
ds = load_dataset("json", data_files=file_path, split="train")

lines = [row["text"] for row in ds][:MAX_LINES]
print(f"  Lines loaded: {len(lines):,}\n")

# ── 6. Evaluate ───────────────────────────────────────────────────────────────
total_words     = 0
total_subtoks   = 0
total_continued = 0
total_bytes     = 0
all_ids         = []
unk_count       = 0

for line in tqdm(lines, desc="Tokenizing"):
    line = line.strip()
    if not line:
        continue

    words = word_split(line)
    if not words:
        continue

    enc    = tok.encode(line)
    ids    = enc.ids
    tokens = enc.tokens

    total_words   += len(words)
    total_subtoks += len(ids)
    total_bytes   += len(line.encode("utf-8"))
    unk_count     += ids.count(unk_id)
    all_ids.extend(ids)

    # Continuation: tokenize each word solo, flag if ≥2 subtokens
    for w in words:
        if len(tok.encode(w).ids) > 1:
            total_continued += 1

# ── 7. Aggregate & print ──────────────────────────────────────────────────────
f_score  = total_subtoks / total_words     if total_words   else 0
cont_pct = total_continued / total_words * 100 if total_words else 0
bpt      = total_bytes / total_subtoks     if total_subtoks else 0
unk_rate = unk_count / total_subtoks * 100 if total_subtoks else 0
re_eff   = renyi_efficiency(all_ids, alpha=RENYI_ALPHA)

print(f"""
  -------------------------------------------------------
      MUTANT-Exact Metrics — Hindi (XL-Sum)      
  -------------------------------------------------------
     Lines evaluated    : {len(lines):>8,}           
     Total words        : {total_words:>8,}            
     Total tokens       : {total_subtoks:>8,}            
  -------------------------------------------------------
     Fertility       ↓  : {f_score:>8.4f}           
     Continuation %  ↓  : {cont_pct:>8.2f}%      
     Bytes/Token     ↑  : {bpt:>8.2f}            
     Rényi Effic.    ↑  : {re_eff:>8.4f}             
     Unk Rate        ↓  : {unk_rate:>8.4f}%            
  -------------------------------------------------------
""")

# ── 8. Comparison table ───────────────────────────────────────────────────────
# Published Hindi fertility scores (MUTANT Table 2, BharatBench Table 1)
PUBLISHED = {
    "LLaMA-3.1"    : 4.20,
    "mBERT"        : 3.80,
    "Gemma-2"      : 3.50,
    "XLM-R"        : 3.20,
    "GPT-4o"       : 2.80,
    "MuRIL"        : 2.10,
    "Sarvam-1"     : 1.60,
    "Krutrim"      : 1.55,
    "MUTANT-Indic" : 1.28,
}

print(f"{'═'*52}")
print(f"  Hindi Fertility — Published vs Yours (↓ better)")
print(f"{'═'*52}")
ranking = sorted(list(PUBLISHED.items()) + [("YOUR TOKENIZER", f_score)],
                 key=lambda x: x[1])
for rank, (name, score) in enumerate(ranking, 1):
    you    = " ◀ YOU" if name == "YOUR TOKENIZER" else ""
    bar    = "█" * int(score * 7)
    print(f"  {rank:2}. {name:<18} {score:.3f}  {bar}{you}")

print(f"{'═'*52}")
delta = f_score - PUBLISHED["MUTANT-Indic"]
print(f"\n  MUTANT-Indic SOTA : {PUBLISHED['MUTANT-Indic']:.3f}")
print(f"  Your tokenizer    : {f_score:.4f}")
if delta <= 0:
    print(f"  Result            : {abs(delta):.4f} BETTER than SOTA ✓")
else:
    print(f"  Gap vs SOTA       : +{delta:.4f} above MUTANT-Indic")
print()
