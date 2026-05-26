import os

train_loop = 20

eval_every = 50
eval_steps = 5
save_every = 200
max_grad_norm = 1.0

TOTAL_ROWS = 50_000
batch_size_tokenizer = 1000

batch_size_encoder = 64
context_length = 128

vocab_size = 52000
embedding_dim = 256

batch_size_attention = 64
num_heads = 8
d_model = 256

hidden_dim_ffn = 704

TOKENIZER_DIR = "data"
TOKENIZER_VOCAB_PATH = os.path.join(TOKENIZER_DIR, "fineweb-vocab.json")
TOKENIZER_MERGES_PATH = os.path.join(TOKENIZER_DIR, "fineweb-merges.txt")
TOKENIZER_JSON_PATH = os.path.join(TOKENIZER_DIR, "fineweb-tokenizer.json")
TOKENIZED_DATA_PATH = os.path.join(TOKENIZER_DIR, "fineweb_tokens.npy")