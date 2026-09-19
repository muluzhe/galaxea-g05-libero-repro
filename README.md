# Galaxea G0.5 LIBERO 仿真复现（星海图「复现与开源共创挑战」· 仿真复现方向）

在单卡 RTX 4090 上完整复现 G0.5 官方 LIBERO 仿真评测：**4 套件 × 10 任务 × 10 trials = 400 episodes，全部成功（100.0%）**，官方技术报告成绩 98.9%，结果一致。全部 400 个 rollout 视频留档。

- 官方仓库：https://github.com/OpenGalaxea/GalaxeaVLA
- 技术报告：https://opengalaxea.github.io/G05/
- 模型：https://huggingface.co/OpenGalaxea/G05（gated，`g05-libero` checkpoint）

## 结果

| Suite | Success | Rate |
|---|---|---|
| libero_goal | 100/100 | 100.0% |
| libero_spatial | 100/100 | 100.0% |
| libero_object | 100/100 | 100.0% |
| libero_10 | 100/100 | 100.0% |
| **OVERALL** | **400/400** | **100.0%** |

结构化结果见 `results/final_summary.json`（per-task / per-suite / overall）。

## 环境

| 项 | 配置 |
|---|---|
| GPU | NVIDIA RTX 4090 24GB（评估峰值约 16GB） |
| 系统 / CPU / RAM | Ubuntu 24.04.1 / 128 核 / 20GB |
| Python / Torch | 3.10.16（uv + 官方 uv.lock）/ 2.7.1+cu128 |
| 注意力内核 | flash-attn-4 4.0.0b15、flash-linear-attention 0.5.0 |
| 仿真 | LIBERO：robosuite 1.4.0 + MuJoCo 3.3.2，Mesa EGL 离屏渲染（256×256） |
| 模型权重 | `g05-libero/model.pt`（11.4GB）+ `action_tokenizer.pt` + Qwen3.5 processor |

## 复现步骤

```bash
git clone https://github.com/OpenGalaxea/GalaxeaVLA && cd GalaxeaVLA
uv sync --index-strategy unsafe-best-match          # 官方锁定环境
bash ../scripts/download_weights.sh                  # 权重 + bundle 符号链接
uv pip install --no-deps -e ../LIBERO                # LIBERO 仿真器
bash ../scripts/run_eval.sh                          # 一键评估（见 scripts/）
```

## 推理链路与关键配置

```
客户端(8路并行LIBERO env) --WebSocket+msgpack--> serve_policy_batched.py(动态批量推理, 单卡)
每 10 步 action chunk 推理一次：观测编码 -> Qwen3.5 VLM 前向 -> 连续动作头 -> ActionCodecV2 -> 动作块
```

- `action_steps=10`，批量推理（实测 batch=8 时 prefill≈89ms、动作生成≈390ms）
- 关键环境变量：`MUJOCO_GL=egl`、`LP_NUM_THREADS=2`、`NUMBA_DISABLE_JIT=1`、`LIBERO_CONFIG_PATH`
- 配置来源：checkpoint 自带 `.hydra/config.yaml`（`continuous_action=true`）

## 关键适配与修复（见 patches/）

1. **`patches/01`** — LIBERO 与 PyTorch 2.6+ 兼容：`torch.load(init_states)` 需
   `weights_only=False`（旧版 init states 为 numpy pickle，否则 UnpicklingError）。
2. **`patches/02`** — 多进程并行评估段错误根因修复：主进程 import robosuite 时 numba
   线程池已初始化，fork 出的仿真 worker 段错误（C 层崩溃、无 traceback）。
   禁用 robosuite `ENABLE_NUMBA` 后一次跑通。
3. **`patches/03`** — 受限环境 sdist 构建失败 workaround：12 个 sdist-only 包
   （bddl/future/gym/robomimic/deepspeed 等）用 `setup.py bdist_wheel --keep-temp`
   手工构建 wheel 预装 + `uv sync --no-install-package` 排除。
4. **无 NVIDIA EGL 的无头服务器**：装 Mesa EGL（`libegl1-mesa-dev` 等） +
   `MUJOCO_GL=egl`；llvmpipe 每上下文默认 64 渲染线程，32 环境并发会线程爆炸，
   `LP_NUM_THREADS=2` 限流。

## 产物

- `results/final_summary.json` — 结构化总结果
- `screenshots/` — 运行证明（成绩汇总 / 服务器日志 / 环境配置 / 40 任务明细）
- 完整 400 个 rollout 视频 + 展示视频随作品材料提交（约 110MB）

## License

遵循 G0.5 Community License（非商业研究用途）；本仓库仅包含复现脚本/补丁/结果，
不重分发模型权重。
