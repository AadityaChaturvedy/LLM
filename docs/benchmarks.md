# Model Checkpoints & Scaling History

All figures below are self-reported from local training runs (checkpoints/, log/, and data/ are gitignored, so these aren't reproducible from the repo alone without retraining).

| Model | Params | Tokens Trained | Vocab | d_model | Layers | KV Heads | Context | Loss | PPL | Status |
|---|---|---|---|---|---|---|---|---|---|---|
| 18M | 18,164,992 | 61,129,344 | 64k | — | — | — | — | 4.1606 | 64.12 | early validation run |
| 153M | 153,398,016 | 607,516,629 | 64k | — | — | — | — | 3.7403 | 42.11 | early validation run |
| 468M | 468,763,648 | 414,055,961 | 128k (Hinglish) | — | — | — | — | — | — | superseded |
| 252M | 252,216,320 | 7,960,014,699 | 64k | 1024 | 16 | — | 512 | — | — | **primary evaluated checkpoint** (base + 3 fine-tunes, see below) |
| 533M | 533,530,880 | 5,333,690,581 | 64k | 1280 | 26 | 4 | 512 | — | — | trained ~169 hrs (~7 days); Chinchilla ratio 23x vs. 20x recommended |
| 510M (current) | — | — | 64k | 1280 | 24 | 4 | 1024 | — | — | matches the config currently committed in `src/config.py` |
| 1.05B | 1,050,629,888 | 8,252,111,852 | 64k | 1792 | 24 | — | 512 | — | — | large-scale run |

One abandoned attempt worth recording: a ~1.26B-parameter model was started and tested at 13.3% into its first epoch, but its generations were still incoherent and projected to take too long to reach usability — so the project pivoted to training the smaller 252M model much further (7.96B tokens, a ~23x token-to-parameter Chinchilla ratio) instead of pushing a larger, undertrained model. That 252M run became the primary checkpoint used for all evaluation.
