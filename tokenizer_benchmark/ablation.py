import argparse
import json
import os
import re
import string
import sys
import unicodedata


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from src import custom_bpe
from src.config import TOKENIZER_MERGES_PATH, TOKENIZER_VOCAB_PATH
from src.custom_bpe import CustomIndicBPE
from tokenizer_benchmark.benchmark_datasets import load_named_dataset
from tokenizer_benchmark.metrics import evaluate_tokenizer_on_lines


DEVA_BLOCK = r"\u0900-\u097F"
DEVA_DIGITS = r"\u0966-\u096F"
DEVA_CLUSTER = (
    r"(?:"
    r"[{d}]"
    r"[\u094D{d}]*"
    r"[\u0900-\u0903\u093A-\u094C\u094E-\u094F\u0951-\u0957\u0962\u0963]*"
    r")+".format(d=DEVA_BLOCK)
)
DEFAULT_PRETOK_RE = re.compile(rf"{DEVA_CLUSTER}|[{DEVA_DIGITS}]+|[A-Za-z0-9Ċ]+|[^\w\s]", re.UNICODE)
NO_CLUSTER_PRETOK_RE = re.compile(rf"[{DEVA_BLOCK}]|[{DEVA_DIGITS}]+|[A-Za-z0-9Ċ]+|[^\w\s]", re.UNICODE)


def pre_tokenize_variant(text, devanagari_only=True, transliterate=True, cluster=True):
    text = unicodedata.normalize("NFC", text)
    if devanagari_only:
        if transliterate:
            text = custom_bpe.transliterate_acronyms(text)
        text = re.sub(r"[^\u0900-\u097F0-9\s!\"#$%&'()*+,\-./:;<=>?@\[\\\]\^_`{|}~]", " ", text)
    text = text.replace("\n", " Ċ ")

    regex = DEFAULT_PRETOK_RE if cluster else NO_CLUSTER_PRETOK_RE
    tokens = []
    for chunk in re.split(r"(\s+)", text):
        if not chunk or chunk.isspace():
            continue
        first = True
        for match in regex.finditer(chunk):
            sub = match.group(0)
            tokens.append(("Ġ" + sub) if first else sub)
            first = False
    return tokens


class AblationTokenizer:
    def __init__(self, name, base_tokenizer, transliterate=True, cluster=True, cache_enabled=True):
        self.name = name
        self.base = base_tokenizer
        self.vocab_size = len(base_tokenizer.vocab)
        self.unk_id = base_tokenizer.token_to_id.get("<unk>")
        self.transliterate = transliterate
        self.cluster = cluster
        self.cache_enabled = cache_enabled
        self._cache = {}

    def _encode_word(self, word_str):
        if self.cache_enabled and word_str in self._cache:
            return list(self._cache[word_str])
        word = list(word_str)
        if len(word) >= 2:
            while True:
                best_i = -1
                best_rank = float("inf")
                for i in range(len(word) - 1):
                    pair = (word[i], word[i + 1])
                    rank = self.base._merge_ranks.get(pair)
                    if rank is not None and rank < best_rank:
                        best_rank = rank
                        best_i = i
                if best_i == -1:
                    break
                word[best_i] = word[best_i] + word[best_i + 1]
                del word[best_i + 1]
        if self.cache_enabled:
            if len(self._cache) >= 100000:
                self._cache.clear()
            self._cache[word_str] = list(word)
        return list(word)

    def encode(self, text):
        tokens = []
        for pretoken in pre_tokenize_variant(
            text,
            devanagari_only=self.base.devanagari_only,
            transliterate=self.transliterate,
            cluster=self.cluster,
        ):
            if pretoken in self.base.SPECIAL_TOKENS:
                tokens.append(pretoken)
            else:
                tokens.extend(self._encode_word(pretoken))
        return [self.base.token_to_id.get(token, self.unk_id) for token in tokens]

    def decode(self, ids):
        return self.base.decode(ids)


def load_base_tokenizer():
    tokenizer = CustomIndicBPE(devanagari_only=True)
    tokenizer.load(TOKENIZER_VOCAB_PATH, TOKENIZER_MERGES_PATH)
    return tokenizer


def main():
    parser = argparse.ArgumentParser(description="Run tokenizer ablations without editing the main LLM.")
    parser.add_argument("--dataset", default="sample")
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--local-path", default=None)
    parser.add_argument("--output", default="tokenizer_benchmark/results/ablation_results.json")
    args = parser.parse_args()

    base = load_base_tokenizer()
    variants = [
        AblationTokenizer("custom_normal", base, transliterate=True, cluster=True, cache_enabled=True),
        AblationTokenizer("no_acronym_transliteration", base, transliterate=False, cluster=True, cache_enabled=True),
        AblationTokenizer("no_devanagari_cluster", base, transliterate=True, cluster=False, cache_enabled=True),
        AblationTokenizer("no_encode_cache", base, transliterate=True, cluster=True, cache_enabled=False),
    ]
    lines = load_named_dataset(args.dataset, limit=args.limit, seed=args.seed, local_path=args.local_path)
    rows = []
    for variant in variants:
        print(f"[INFO] Ablation {variant.name} on {args.dataset}")
        row = evaluate_tokenizer_on_lines(variant, lines, repeat_speed=3)
        row["dataset"] = args.dataset
        rows.append(row)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump({"dataset": args.dataset, "results": rows}, handle, ensure_ascii=False, indent=2)
    print(f"[DONE] Wrote {args.output}")


if __name__ == "__main__":
    main()
