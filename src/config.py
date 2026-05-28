import os

#Generate
max_new_tokens = 128
temperature = 0.7
top_k = 50

# Training Loop
train_loop = 150000

# Evaluation & Checkpointing
eval_every = 2000
eval_steps = 5
save_every = 10000
max_grad_norm = 1.0

# Tokenizer & Dataset Rows
TOTAL_ROWS = 1_000_000
batch_size_tokenizer = 1000

# Model Batch & Sequence Sizing
batch_size_encoder = 16        # Mini-batch size to prevent OOM (reduced from 32 to 16)
accumulation_steps = 8        # Accumulate gradients to keep effective batch size at 128 (16 * 8)
context_length = 256          # Doubled context window (from 128 to 256)

# Model Dimensions (153M parameters)
vocab_size = 52000
embedding_dim = 768
num_layers = 12
num_heads = 12
d_model = 768
hidden_dim_ffn = 3072        

# File Paths
TOKENIZER_DIR = "data"
TOKENIZER_VOCAB_PATH = os.path.join(TOKENIZER_DIR, "fineweb-vocab.json")
TOKENIZER_MERGES_PATH = os.path.join(TOKENIZER_DIR, "fineweb-merges.txt")
TOKENIZER_JSON_PATH = os.path.join(TOKENIZER_DIR, "fineweb-tokenizer.json")
TOKENIZED_DATA_PATH = os.path.join(TOKENIZER_DIR, "fineweb_tokens.npy")
