import os

#Generate
max_new_tokens = 128
temperature = 0.7
top_k = 50

# Execution Toggles
TRAIN_TOKENIZER = True
TRAIN_LLM = True

# Training Loop
train_loop = 29000

# Evaluation & Checkpointing
eval_every = 500
eval_steps = 50
save_every = 10000
max_grad_norm = 1.0
patience = 5
min_delta = 0.001

# Tokenizer & Dataset Rows
TOKENIZER_ROWS = 10_000
LLM_ROWS = 10_000_000
batch_size_tokenizer = 1000

# Language Mode: 'english' or 'hindi' or 'hinglish'
LANGUAGE = "hindi"

hindi_vocab_size = 0
english_vocab_size = 0

# Model Dimensions (based on LANGUAGE mode)
use_gqa = True

if LANGUAGE == "hindi":
    hindi_vocab_size = 64_000
    vocab_size = hindi_vocab_size
    embedding_dim = 1280
    d_model = 1280
    num_layers = 24
    num_heads = 16
    num_kv_heads = 4
    hidden_dim_ffn = 3584
    batch_size_encoder = 26
    accumulation_steps = 8
    context_length = 512
elif LANGUAGE == "hinglish":
    hindi_vocab_size = 100_000
    english_vocab_size = 28_000
    vocab_size = hindi_vocab_size + english_vocab_size 
    embedding_dim = 1024
    num_layers = 24
    num_heads = 16
    num_kv_heads = 4
    d_model = 1024
    hidden_dim_ffn = 4096
    batch_size_encoder = 4
    accumulation_steps = 8
    context_length = 1024
else:
    vocab_size = 52_000
    embedding_dim = 768
    num_layers = 12
    num_heads = 12
    num_kv_heads = 4
    d_model = 768
    hidden_dim_ffn = 3072 
    batch_size_encoder = 8
    accumulation_steps = 8
    context_length = 512

# File Paths
TOKENIZER_DIR = os.path.join("data", LANGUAGE)
TOKENIZER_VOCAB_PATH = os.path.join(TOKENIZER_DIR, "model-vocab.json")
TOKENIZER_MERGES_PATH = os.path.join(TOKENIZER_DIR, "model-merges.txt")
TOKENIZER_JSON_PATH = os.path.join(TOKENIZER_DIR, "tokenizer.json")
TOKENIZED_DATA_PATH = os.path.join(TOKENIZER_DIR, "tokens.npy")
CHECKPOINT_PATH = "sft_checkpoints_instruct/ckpt_instruct_epoch_2.pt"

