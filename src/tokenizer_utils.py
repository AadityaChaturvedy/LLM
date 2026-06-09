import os

from tokenizers import ByteLevelBPETokenizer
from tqdm import tqdm

from src.config import (
    TOKENIZER_ROWS,
    vocab_size, 
    TOKENIZER_DIR,
    TOKENIZER_JSON_PATH,
    batch_size_tokenizer,
)


def iter_dataset_rows(dataset, total_rows, desc):
    for i, row in enumerate(tqdm(dataset, total=total_rows, desc=desc)):
        yield row
        if i >= total_rows - 1:
            break

class Tokenizer:
    def __init__(self):
        self.tokenizer = ByteLevelBPETokenizer()
    
    def batch_iterator(self, dataset):
        batch = []
        for row in iter_dataset_rows(dataset, TOKENIZER_ROWS, desc="Tokenizing"):
            batch.append(row["text"])
            if len(batch) == batch_size_tokenizer:
                yield batch
                batch = []
        if batch:
            yield batch

    def train(self, dataset):
        os.makedirs(TOKENIZER_DIR, exist_ok=True)
        print("Training tokenizer")
        self.tokenizer.train_from_iterator(
            self.batch_iterator(dataset),
            vocab_size=52_000,
            min_frequency=2,
            special_tokens=["<s>","<pad>","</s>","<unk>","<mask>"]
        )

        print("Tokenizer Training Completed")

        self.tokenizer.save(TOKENIZER_JSON_PATH)
        self.tokenizer.save_model(TOKENIZER_DIR, "fineweb")