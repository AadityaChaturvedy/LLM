import numpy as np
import torch
import os
import multiprocessing as mp
from functools import partial
from src.custom_bpe import CustomIndicBPE, pre_tokenize
from src.config import (
    LLM_ROWS,
    TOKENIZED_DATA_PATH,
    TOKENIZER_MERGES_PATH,
    TOKENIZER_VOCAB_PATH,
    batch_size_encoder,
    context_length,
    LANGUAGE,
)
from tqdm import tqdm


def load_tokenized_data(path=TOKENIZED_DATA_PATH):
    return np.load(path, mmap_mode="r")


def iter_dataset_rows(dataset, total_rows, desc):
    for i, row in enumerate(tqdm(dataset, total=total_rows, desc=desc)):
        yield row
        if i >= total_rows - 1:
            break


def get_batch(data, batch_size, context_length):
    ix = np.random.randint(0, len(data) - context_length - 1, size=(batch_size,))
    indices = ix[:, None] + np.arange(context_length)[None, :]
    x = data[indices].astype(np.int64)
    y = data[indices + 1].astype(np.int64)
    return torch.from_numpy(x), torch.from_numpy(y)


# ---------------------------------------------------------------------------
# Module-level state for worker processes
# Loaded once per worker via initializer — avoids re-loading 60k merges
# for every single document.
# ---------------------------------------------------------------------------
_worker_tokenizer: CustomIndicBPE = None


def _worker_init(vocab_path: str, merges_path: str):
    """Called once per worker process. Loads tokenizer into global."""
    global _worker_tokenizer
    _worker_tokenizer = CustomIndicBPE(devanagari_only=(LANGUAGE == "hindi"))
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
        self.tokenizer = CustomIndicBPE(devanagari_only=(LANGUAGE == "hindi"))
        self.tokenizer.load(TOKENIZER_VOCAB_PATH, TOKENIZER_MERGES_PATH)

    def TestTokenizer(self):
        text = "नमस्ते, आप कैसे हैं?"
        encoded = self.tokenizer.encode(text)
        print("Tokens:", encoded.tokens)
        print("IDs:   ", encoded.ids)
        print("Decode:", self.tokenizer.decode(encoded.ids))

    def _process_and_append_batch(self, texts, f_bin, pool, num_workers):
        if num_workers == 0 or pool is None:
            batch_ids = []
            for text in texts:
                batch_ids.extend(self.tokenizer.encode(text).ids)
        else:
            chunk_size = max(1, len(texts) // (num_workers * 20))
            results = list(
                pool.imap(_encode_doc, texts, chunksize=chunk_size)
            )
            batch_ids = [id_ for doc_ids in results for id_ in doc_ids]

        # Convert to uint16 and write to binary file
        batch_arr = np.array(batch_ids, dtype=np.uint16)
        batch_arr.tofile(f_bin)

    def EncodeTokens(self, dataset, num_workers: int = None):
        """
        Encode dataset to a flat uint16 numpy array in batches to avoid OOM,
        and save to disk.

        Parameters
        ----------
        dataset   : iterable of dicts with a 'text' key
        num_workers : number of parallel processes.
                      Defaults to min(8, cpu_count - 1).
                      Set to 0 to disable multiprocessing (useful for debugging).
        """
        if num_workers is None:
            num_workers = min(8, max(1, os.cpu_count() - 1))

        batch_size = 100_000
        temp_bin_path = TOKENIZED_DATA_PATH + ".tmp.bin"
        if os.path.exists(temp_bin_path):
            os.remove(temp_bin_path)

        print(f"[encoder] Streaming and encoding {LLM_ROWS:,} documents in batches of {batch_size:,} …")
        
        # Initialize a single persistent pool if multiprocessing is enabled
        pool = None
        if num_workers > 0:
            pool = mp.Pool(
                processes=num_workers,
                initializer=_worker_init,
                initargs=(TOKENIZER_VOCAB_PATH, TOKENIZER_MERGES_PATH),
            )

        try:
            # Open binary file for appending
            with open(temp_bin_path, "ab") as f_bin:
                batch_texts = []
                total_processed = 0

                pbar = tqdm(total=LLM_ROWS, desc="Processing")
                for row in dataset:
                    if total_processed >= LLM_ROWS:
                        break
                    if row and "text" in row:
                        batch_texts.append(row["text"])
                        
                    # Once batch is full, process it
                    if len(batch_texts) >= batch_size:
                        self._process_and_append_batch(batch_texts, f_bin, pool, num_workers)
                        total_processed += len(batch_texts)
                        pbar.update(len(batch_texts))
                        batch_texts = [] # Clear memory
                        
                # Process remaining documents
                if batch_texts and total_processed < LLM_ROWS:
                    # Truncate remaining batch if it exceeds LLM_ROWS
                    remaining_needed = LLM_ROWS - total_processed
                    batch_texts = batch_texts[:remaining_needed]
                    if batch_texts:
                        self._process_and_append_batch(batch_texts, f_bin, pool, num_workers)
                        total_processed += len(batch_texts)
                        pbar.update(len(batch_texts))
                        batch_texts = []
                    
                pbar.close()
        finally:
            # Close the pool cleanly
            if pool is not None:
                pool.close()
                pool.join()

        print(f"[encoder] Finished encoding. Loading binary file and saving to {TOKENIZED_DATA_PATH} …")
        
        # Load the flat binary file and save as standard npy
        all_tokens = np.fromfile(temp_bin_path, dtype=np.uint16)
        np.save(TOKENIZED_DATA_PATH, all_tokens)
        
        if os.path.exists(temp_bin_path):
            os.remove(temp_bin_path)

        print(f"[encoder] Saved {all_tokens.shape[0]:,} tokens → {TOKENIZED_DATA_PATH}")
        loaded = load_tokenized_data()
        print(f"[encoder] Verified load: {len(loaded):,} tokens")
