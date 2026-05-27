import os

# Generation
max_new_tokens = 128
temperature = 0.7
top_k = 50

train_loop = 100000

eval_every = 2000
eval_steps = 5
save_every = 5000
max_grad_norm = 1.0

TOTAL_ROWS = 100_000
batch_size_tokenizer = 1000

batch_size_encoder = 128
context_length = 128

vocab_size = 52000
embedding_dim = 256

batch_size_attention = 64
num_heads = 8
d_model = 256

hidden_dim_ffn = 704

num_layers = 6

TOKENIZER_DIR = "data"
TOKENIZER_VOCAB_PATH = os.path.join(TOKENIZER_DIR, "fineweb-vocab.json")
TOKENIZER_MERGES_PATH = os.path.join(TOKENIZER_DIR, "fineweb-merges.txt")
TOKENIZER_JSON_PATH = os.path.join(TOKENIZER_DIR, "fineweb-tokenizer.json")
TOKENIZED_DATA_PATH = os.path.join(TOKENIZER_DIR, "fineweb_tokens.npy")
