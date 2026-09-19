# Galaxea G0.5 LIBERO 仿真复现报告

> 星海图 Galaxea G0.5「复现与开源共创挑战」· 仿真复现方向
> 完成内容：G0.5 模型推理与任务验证 + 模型运行过程与 Rollout 结果展示

## 1. 结果总览

| Suite | 成功率 | Trials/任务 |
|---|---|---|
| libero_goal | **100/100（100.0%）** | 10 |
| libero_spatial | **100/100（100.0%）** | 10 |
| libero_object | **100/100（100.0%）** | 10 |
| libero_10 | **100/100（100.0%）** | 10 |
| **OVERALL** | **400/400（100.0%）** | — |

官方技术报告 LIBERO 成绩：98.9%。本复现结果与官方一致（10 trials 下 100% 在统计涨落范围内）。

## 2. 运行环境

| 项目 | 配置 |
|---|---|
| GPU | NVIDIA GeForce RTX 4090（24GB，推理峰值约 16GB） |
| 系统 | Ubuntu 24.04.1 / 128 核 CPU / 20GB RAM |
| CUDA / PyTorch | CUDA 12.8 / torch 2.7.1+cu128 |
| Python | 3.10.16（uv 管理，按官方 uv.lock 精确安装） |
| 模型 | `OpenGalaxea/G05` → `g05-libero` checkpoint（11.4GB） |
| 仿真器 | LIBERO（robosuite 1.4.0 + mujoco 3.3.2，EGL 离屏渲染） |

## 3. 复现步骤

```bash
# 1) 代码与环境（严格按官方 uv.lock）
git clone https://github.com/OpenGalaxea/GalaxeaVLA && cd GalaxeaVLA
uv sync --index-strategy unsafe-best-match

# 2) 权重下载（gated 仓库，需先在 HF 同意协议；国内走 hf-mirror）
HF_ENDPOINT=https://hf-mirror.com huggingface-cli download OpenGalaxea/G05 \
    --include "g05-libero/*" "action_tokenizer.pt" "qwen3_5_2b_base_processor/*" \
    --local-dir checkpoints
ln -sfn ../action_tokenizer.pt checkpoints/g05-libero/action_tokenizer.pt
ln -sfn ../qwen3_5_2b_base_processor checkpoints/g05-libero/hf_processor

# 3) LIBERO 仿真器（--no-deps 避免破坏锁定环境）
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git ../LIBERO
uv pip install --no-deps -e ../LIBERO

# 4) 一键评估：4 套件 × 10 trials × 8 并行环境，批量策略服务器 + WebSocket
LIBERO_CONFIG_PATH=$(pwd)/experiments/libero \
CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
bash scripts/run/eval_libero.sh checkpoints/g05-libero/model.pt \
    --suites "libero_goal libero_spatial libero_object libero_10" \
    --num_trials 10 --num_parallel 8 --save_videos
```

推理链路：`serve_policy_batched.py` 批量策略服务器（动态 batch，单卡）→ 客户端 8 路并行
LIBERO 环境 → WebSocket 收发观测/动作块 → 10 步 action chunk 执行 → 任务成功判定 → `summary.json`。

## 4. 关键工程问题与修复（复现避坑）

1. **sdist 包无法源码构建**：`uv.lock` 中 bddl/future/gym/robomimic/deepspeed 等 12 个包
   仅记录 sdist，受限环境（构建产物批量清理被拦截）下构建失败。
   修复：用 `setup.py bdist_wheel --keep-temp` 手工构建 wheel 预装，并在
   `uv sync` 时对这些包加 `--no-install-package` 排除。
2. **LIBERO 与 PyTorch 2.6+ 不兼容**：`libero/libero/benchmark/__init__.py` 的
   `torch.load(init_states_path)` 需补 `weights_only=False`（旧版 init states 为 numpy pickle）。
3. **多进程并行评估崩溃（根因最隐蔽）**：主客户端进程 import robosuite 时 numba 线程池
   已初始化，fork 出的仿真 worker 继承损坏状态导致 C 层段错误。
   修复：`robosuite/macros_private.py` 中 `ENABLE_NUMBA=False`（robosuite 源码注释亦确认
   numba 与离屏渲染存在已知冲突）；同时 Mesa 软渲染下设置 `LP_NUM_THREADS=2`
   避免 llvmpipe 每上下文 64 线程 × N 环境导致的线程爆炸。
4. **无 NVIDIA EGL 的无头服务器**：安装 `libegl1-mesa-dev libgles2-mesa-dev libosmesa6`
   后以 `MUJOCO_GL=egl`（Mesa 实现）渲染，128 核 CPU 下性能充足。

## 5. 产物清单

```
outputs/libero_eval_g05-libero_full/
├── G05_LIBERO_showcase.mp4        # 展示视频（75s：片头 → 4 套件精选 rollout → 成绩总结）
├── result_chart.png               # 成绩柱状图（对比官方 98.9%）
├── final_summary.json             # 结构化总结果（per-task / per-suite / overall）
├── libero_{goal,spatial,object,10}/
│   ├── <suite>_parallel_results.json
│   └── videos/                    # 每套件 100 个 rollout 视频（共 400 个）
└── server.log / client.log        # 模型运行过程日志（批量推理 batch_size / infer_ms 等）
```

## 6. 复现时间线（单卡 4090）

- 环境安装 + 权重下载：约 1.5 小时
- 冒烟测试（libero_goal 10 任务 × 1 trial，100% 通过）：约 15 分钟
- 正式评估（400 episodes + 视频保存）：约 3.5 小时
