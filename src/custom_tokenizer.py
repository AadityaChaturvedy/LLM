from datasets import load_dataset
from src.custom_bpe import CustomIndicBPE
from src.config import (
    hindi_vocab_size, english_vocab_size,
    TOKENIZER_VOCAB_PATH,
    TOKENIZER_MERGES_PATH,
)

class CustomTokenizer:
    def __init__(self):
        self.tokenizer = CustomIndicBPE()

    def train(self, language, max_docs_hindi=hindi_vocab_size, max_docs_english=english_vocab_size):

        if language == "hinglish":
            # Train Hindi alone
            print("--- Training Hindi ---")
            hindi_dataset = load_dataset("ai4bharat/sangraha", 
                                        data_dir="verified/hin",
                                        split="train", streaming=True)
            t_hindi = CustomIndicBPE()
            t_hindi.train(hindi_dataset, vocab_size=hindi_vocab_size, max_docs=max_docs_hindi)

            # Train English alone  
            print("--- Training English ---")
            english_dataset = load_dataset("HuggingFaceFW/fineweb",
                                            name="sample-10BT",
                                            split="train", streaming=True)
            t_english = CustomIndicBPE()
            t_english.train(english_dataset, vocab_size=english_vocab_size, max_docs=max_docs_english)

            # Merge vocabs
            special_tokens = ["<s>", "<pad>", "</s>", "<unk>", "<mask>"]
            merged_vocab = list(special_tokens)
            seen = set(special_tokens)
            for tok in t_hindi.vocab:
                if tok not in seen:
                    merged_vocab.append(tok)
                    seen.add(tok)
            for tok in t_english.vocab:
                if tok not in seen:
                    merged_vocab.append(tok)
                    seen.add(tok)

            # Interleave merges by rank position — not concatenate
            # This preserves relative importance within each language
            hindi_merges = t_hindi.merges
            english_merges = t_english.merges
            
            merged_merges = []
            seen_merges = set()
            hi, en = 0, 0
            hindi_slots = 4   # 4 Hindi merges per 1 English merge
            
            while hi < len(hindi_merges) or en < len(english_merges):
                # Add hindi_slots Hindi merges
                for _ in range(hindi_slots):
                    if hi < len(hindi_merges):
                        pair, tok = hindi_merges[hi]
                        hi += 1
                        if pair not in seen_merges:
                            merged_merges.append((pair, tok))
                            seen_merges.add(pair)
                # Add 1 English merge
                if en < len(english_merges):
                    pair, tok = english_merges[en]
                    en += 1
                    if pair not in seen_merges:
                        merged_merges.append((pair, tok))
                        seen_merges.add(pair)

            self.tokenizer.vocab = merged_vocab
            self.tokenizer.token_to_id = {t: i for i, t in enumerate(merged_vocab)}
            self.tokenizer.id_to_token = {i: t for i, t in enumerate(merged_vocab)}
            self.tokenizer.merges = merged_merges
            self.tokenizer._cache = {}
            self.tokenizer._build_merge_ranks()

            print(f"Merged vocab: {len(merged_vocab)}")
            self.tokenizer.save(TOKENIZER_VOCAB_PATH, TOKENIZER_MERGES_PATH)

        elif language == "hindi":
            print("--- Training Hindi ---")
            hindi_dataset = load_dataset("ai4bharat/sangraha", 
                                        data_dir="verified/hin",
                                        split="train", streaming=True)
            self.tokenizer.train(hindi_dataset, vocab_size=hindi_vocab_size, max_docs=max_docs_hindi)
            self.tokenizer.save(TOKENIZER_VOCAB_PATH, TOKENIZER_MERGES_PATH)

    def load(self):
        self.tokenizer.load(TOKENIZER_VOCAB_PATH, TOKENIZER_MERGES_PATH)

    def encode(self, text):
        return self.tokenizer.encode(text)

    def decode(self, ids):
        return self.tokenizer.decode(ids)
