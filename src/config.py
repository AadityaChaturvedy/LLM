import os

#Generate
max_new_tokens = 128
temperature = 0.7
top_k = 50

# Execution Toggles
TRAIN_TOKENIZER = True
TRAIN_LLM = True

#Activation Checkpointing
ACTIVATION_CHECKPOINTING = True
WINDOW_SIZE = 512
GLOBAL_ATTENTION_INTERVAL = 4

# Training Loop
train_loop = 30365

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
use_gqa = False

# Mixture-of-Experts FFN
# Set MoE = False to use the standard dense FeedForwardNetwork.
MoE = True
moe_num_experts = 8
moe_top_k = 1
moe_capacity_factor = 1.25
moe_eval_capacity_factor = 2.0
moe_min_capacity = 4
moe_aux_loss_weight = 0.01
moe_aux_loss_warmup_steps = 1000
moe_router_z_loss_weight = 0.001
moe_router_noise_std = 0.1
moe_router_temperature = 1.0
moe_num_shared_experts = 0
moe_shared_expert_weight = 1.0
moe_renormalize_after_drop = True
moe_log_every = 100

if LANGUAGE == "hindi":
    hindi_vocab_size = 64_000
    vocab_size = hindi_vocab_size
    embedding_dim = 1024
    d_model = 1024
    num_layers = 16
    num_heads = 16
    num_kv_heads = 16
    hidden_dim_ffn = 2432
    batch_size_encoder = 8
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
CHECKPOINT_PATH = "checkpoints/ckpt_step_80000.pt"
