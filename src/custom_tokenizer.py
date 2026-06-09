from datasets import load_dataset
from src.custom_bpe import CustomIndicBPE
from src.config import (
    TOKENIZER_ROWS,
    hindi_vocab_size, english_vocab_size,
    TOKENIZER_VOCAB_PATH,
    TOKENIZER_MERGES_PATH,
    LANGUAGE,
)

class CustomTokenizer:
    def __init__(self):
        self.tokenizer = CustomIndicBPE(devanagari_only=(LANGUAGE == "hindi"))

    def train(self, language, max_docs_hindi=TOKENIZER_ROWS, max_docs_english=TOKENIZER_ROWS):

        if language == "hinglish":
            # Train Hindi alone
            print("--- Training Hindi ---")
            hindi_dataset = load_dataset("ai4bharat/sangraha", 
                                        data_dir="verified/hin",
                                        split="train", streaming=True)
            t_hindi = CustomIndicBPE(devanagari_only=True)
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
            
            hindi_merge_tokens = {tok for pair, tok in t_hindi.merges}
            english_merge_tokens = {tok for pair, tok in t_english.merges}

            # Add Hindi base chars
            for tok in t_hindi.vocab:
                if tok not in hindi_merge_tokens and tok not in seen:
                    merged_vocab.append(tok)
                    seen.add(tok)
            # Add English base chars
            for tok in t_english.vocab:
                if tok not in english_merge_tokens and tok not in seen:
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
            
            base_size = len(merged_vocab)
            
            while hi < len(hindi_merges) or en < len(english_merges):
                # Add hindi_slots Hindi merges
                for _ in range(hindi_slots):
                    if hi < len(hindi_merges):
                        pair, tok = hindi_merges[hi]
                        hi += 1
                        if pair not in seen_merges:
                            merged_merges.append((pair, tok))
                            seen_merges.add(pair)
                            if tok not in seen:
                                merged_vocab.append(tok)
                                seen.add(tok)
                # Add 1 English merge
                if en < len(english_merges):
                    pair, tok = english_merges[en]
                    en += 1
                    if pair not in seen_merges:
                        merged_merges.append((pair, tok))
                        seen_merges.add(pair)
                        if tok not in seen:
                            merged_vocab.append(tok)
                            seen.add(tok)

            # Enforce combined vocab size limit (from config)
            from src.config import vocab_size as target_vocab_size
            if len(merged_vocab) > target_vocab_size:
                assert target_vocab_size >= base_size, f"target_vocab_size ({target_vocab_size}) is too small to fit base chars and specials ({base_size})"
                merged_vocab = merged_vocab[:target_vocab_size]
                # Filter merges that create tokens outside the trimmed vocab
                allowed_tokens = set(merged_vocab)
                merged_merges = [m for m in merged_merges if m[1] in allowed_tokens]

            self.tokenizer.vocab = merged_vocab
            self.tokenizer.token_to_id = {t: i for i, t in enumerate(merged_vocab)}
            self.tokenizer.id_to_token = {i: t for i, t in enumerate(merged_vocab)}
            self.tokenizer.merges = merged_merges
            self.tokenizer._cache = {}
            self.tokenizer._build_merge_ranks()
            
            # Explicitly set devanagari_only to False so English is allowed in Hinglish
            self.tokenizer.devanagari_only = False

            print(f"Merged vocab: {len(merged_vocab)}")
            self.tokenizer.save(TOKENIZER_VOCAB_PATH, TOKENIZER_MERGES_PATH)

        elif language == "hindi":
            print("--- Training Hindi ---")
            hindi_dataset = load_dataset("ai4bharat/sangraha", 
                                         data_dir="verified/hin",
                                         split="train", streaming=True)
            self.tokenizer.devanagari_only = True
            self.tokenizer.train(hindi_dataset, vocab_size=hindi_vocab_size, max_docs=max_docs_hindi)
            self.tokenizer.save(TOKENIZER_VOCAB_PATH, TOKENIZER_MERGES_PATH)
            
        else:
            raise ValueError(f"Unsupported language mode for training: {language}")

    def load(self):
        self.tokenizer.load(TOKENIZER_VOCAB_PATH, TOKENIZER_MERGES_PATH)

    def encode(self, text):
        return self.tokenizer.encode(text)

    def decode(self, ids):
        return self.tokenizer.decode(ids)
