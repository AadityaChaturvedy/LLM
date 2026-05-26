import numpy as np
import torch
from tokenizers import ByteLevelBPETokenizer

from src.config import (
    TOTAL_ROWS,
    TOKENIZED_DATA_PATH,
    TOKENIZER_MERGES_PATH,
    TOKENIZER_VOCAB_PATH,
    batch_size_encoder,
    context_length,
)
from src.tokenizer_utils import iter_dataset_rows


def load_tokenized_data(path=TOKENIZED_DATA_PATH):
    return np.load(path)


def get_batch(data, batch_size, context_length):
    ix = torch.randint(0, len(data) - context_length - 1, (batch_size,), device=data.device)

    grid = ix.unsqueeze(1) + torch.arange(context_length, device=data.device)

    x = data[grid]
    y = data[grid + 1]
    
    return x, y


class Encoder:
    def __init__(self):
        self.tokenizer = ByteLevelBPETokenizer(TOKENIZER_VOCAB_PATH, TOKENIZER_MERGES_PATH)
    
    def TestTokenizer(self):
        text = "This can be used to test the encoding."

        encoded = self.tokenizer.encode(text)

        print(encoded.tokens)
        print(encoded.ids)

        decoded = self.tokenizer.decode(encoded.ids)
        print(decoded)
    
    def EncodeTokens(self, dataset):
        all_token_ids = []

        for row in iter_dataset_rows(dataset, TOTAL_ROWS, desc="Encoding"):
            encoded = self.tokenizer.encode(row["text"])
            all_token_ids.append(encoded.ids)

        all_tokens = np.fromiter((token for row in all_token_ids for token in row), dtype=np.uint16)
        np.save(TOKENIZED_DATA_PATH, all_tokens)

        print(f"Total tokens processed: {all_tokens.shape[0]}")

        loaded_tokens = load_tokenized_data()
        print(f"Total tokens in dataset: {len(loaded_tokens):,}")