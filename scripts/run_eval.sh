#!/bin/bash
# One-command LIBERO evaluation (4 suites, 10 trials/task, 8 parallel envs)
set -euo pipefail
cd GalaxeaVLA
source .venv/bin/activate
export LIBERO_CONFIG_PATH=$(pwd)/experiments/libero
export CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl LP_NUM_THREADS=2
export NUMBA_DISABLE_JIT=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
bash scripts/run/eval_libero.sh checkpoints/g05-libero/model.pt \
    --suites "libero_goal libero_spatial libero_object libero_10" \
    --num_trials 10 --num_parallel 8 --save_videos \
    --output_dir outputs/libero_eval_g05-libero_full
