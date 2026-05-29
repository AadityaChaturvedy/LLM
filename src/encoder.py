import numpy as np
import torch
import os
import multiprocessing as mp
from functools import partial
from src.custom_bpe import CustomIndicBPE, pre_tokenize
from src.config import (
    TOTAL_ROWS,
    TOKENIZED_DATA_PATH,
    TOKENIZER_MERGES_PATH,
    TOKENIZER_VOCAB_PATH,
    batch_size_encoder,
    context_length,
)
from tqdm import tqdm


def load_tokenized_data(path=TOKENIZED_DATA_PATH):
    return np.load(path)


def iter_dataset_rows(dataset, total_rows, desc):
    for i, row in enumerate(tqdm(dataset, total=total_rows, desc=desc)):
        yield row
        if i >= total_rows - 1:
            break


def get_batch(data, batch_size, context_length):
    ix = torch.randint(0, len(data) - context_length - 1, (batch_size,))
    x = torch.stack([data[i: i + context_length] for i in ix])
    y = torch.stack([data[i + 1: i + context_length + 1] for i in ix])
    return x, y


# ---------------------------------------------------------------------------
# Module-level state for worker processes
# Loaded once per worker via initializer — avoids re-loading 60k merges
# for every single document.
# ---------------------------------------------------------------------------
_worker_tokenizer: CustomIndicBPE = None


def _worker_init(vocab_path: str, merges_path: str):
    """Called once per worker process. Loads tokenizer into global."""
    global _worker_tokenizer
    _worker_tokenizer = CustomIndicBPE()
    _worker_tokenizer.load(vocab_path, merges_path)


def _encode_doc(text: str) -> list:
    """Encode a single document text → list of uint16-safe ints."""
    if not text or not text.strip():
        return []
    return _worker_tokenizer.encode(text).ids


# ---------------------------------------------------------------------------
# Encoder class
# ---------------------------------------------------------------------------
class Encoder:
    def __init__(self):
        self.tokenizer = CustomIndicBPE()
        self.tokenizer.load(TOKENIZER_VOCAB_PATH, TOKENIZER_MERGES_PATH)

    def TestTokenizer(self):
        text = "नमस्ते, आप कैसे हैं?"
        encoded = self.tokenizer.encode(text)
        print("Tokens:", encoded.tokens)
        print("IDs:   ", encoded.ids)
        print("Decode:", self.tokenizer.decode(encoded.ids))

    def EncodeTokens(self, dataset, num_workers: int = None):
        """
        Encode dataset to a flat uint16 numpy array and save to disk.

        Parameters
        ----------
        dataset   : iterable of dicts with a 'text' key
        num_workers : number of parallel processes.
                      Defaults to min(8, cpu_count - 1).
                      Set to 0 to disable multiprocessing (useful for debugging).
        """
        if num_workers is None:
            num_workers = min(8, max(1, os.cpu_count() - 1))

        # Collect raw texts first (streaming datasets need this before forking)
        print(f"[encoder] Collecting {TOTAL_ROWS:,} documents …")
        texts = []
        for row in iter_dataset_rows(dataset, TOTAL_ROWS, desc="Collecting"):
            if row and "text" in row:
                texts.append(row["text"])

        print(f"[encoder] Encoding {len(texts):,} documents with {num_workers} workers …")

        if num_workers == 0:
            # Single-process path (debug / CPU-only)
            all_ids = []
            for text in tqdm(texts, desc="Encoding"):
                all_ids.extend(self.tokenizer.encode(text).ids)
        else:
            # Multi-process path
            # Chunk texts into batches to reduce IPC overhead
            chunk_size = max(1, len(texts) // (num_workers * 20))

            with mp.Pool(
                processes=num_workers,
                initializer=_worker_init,
                initargs=(TOKENIZER_VOCAB_PATH, TOKENIZER_MERGES_PATH),
            ) as pool:
                results = list(
                    tqdm(
                        pool.imap(_encode_doc, texts, chunksize=chunk_size),
                        total=len(texts),
                        desc="Encoding",
                    )
                )

            all_ids = [id_ for doc_ids in results for id_ in doc_ids]

        # Save as uint16 — vocab < 65536 so this is safe and halves disk/RAM
        all_tokens = np.array(all_ids, dtype=np.uint16)
        np.save(TOKENIZED_DATA_PATH, all_tokens)

        print(f"[encoder] Saved {all_tokens.shape[0]:,} tokens → {TOKENIZED_DATA_PATH}")
        loaded = load_tokenized_data()
        print(f"[encoder] Verified load: {len(loaded):,} tokens")