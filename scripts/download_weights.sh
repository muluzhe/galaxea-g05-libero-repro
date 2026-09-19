#!/bin/bash
# Download g05-libero eval bundle (gated repo: accept license at
# https://huggingface.co/OpenGalaxea/G05 first). CN users: use hf-mirror.
set -euo pipefail
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
hf download OpenGalaxea/G05 \
    --include "g05-libero/*" "action_tokenizer.pt" "qwen3_5_2b_base_processor/*" \
    --local-dir checkpoints
cd checkpoints
ln -sfn ../action_tokenizer.pt g05-libero/action_tokenizer.pt
ln -sfn ../qwen3_5_2b_base_processor g05-libero/hf_processor
