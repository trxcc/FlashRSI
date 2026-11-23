# `configs/suite/` 配置说明与自研算法接入指南

本目录存放 五条赛道的全部 **25 份原生 YAML 配置**，统一通过 `flashrsi run --config configs/suite/<name>.yaml` 启动。

---

## 1. 配置文件分类（共 25 份）

### （1）多轮工具交互赛道（6 组 × 双主力壳 = 12 份）
默认配置使用 **`claude_code`**（走 DeepSeek `/anthropic` 兼容接口，适合演化 `CLAUDE.md` / Skills / Hooks / Subagents / Tools）；同名 `*_codex.yaml` 使用纯 Python 内核的 **`codex`**（走 DeepSeek `/responses` 接口，支持连 Agent 底层循环与工具 Python 源码一起演化）。工具截断与上下文压缩均保持两套壳原厂内置默认。

| Claude Code 默认配置 | Codex 对应配置 | 任务划分 | 官方原生超时与轮数设定 |
|---|---|---|---|
| `tb21_clean.yaml` | `tb21_clean_codex.yaml` | TB 2.1 洁净 56 训 / 32 验 | `max_turns: null`（不限轮），完全透传每题 `task.toml` 时钟（`multiplier=1.0`），正式跑 5 次 |
| `swegym_lite.yaml` | `swegym_lite_codex.yaml` | SWE-Gym Lite 230 训 | `max_turns: null`，Solver `3,600s`（仅连模型侧车），独立 `--network=none` 容器官方测试 `1,800s` |
| `engdesign_open.yaml` | `engdesign_open_codex.yaml` | EngDesign 去重 42 训 / 12 验 | 对齐 RRSI：`250` 轮，Solver `5,400s`，独立判分容器 `420s`，Val 配对跑 `4` 次（晋升门槛 `+0.020`） |
| `clawgym.yaml` | `clawgym_codex.yaml` | ClawGym 纯硬测 96 训 / 24 验 | 对齐官方 CLI：Solver `660s`（可选长档 `3,000s`），`max_turns: null`，Checker `120s` |
| `harvey_workspace.yaml` | `harvey_workspace_codex.yaml` | Harvey LAB 95 训 / 24 验 | 对齐官方仓库 `200` 轮 + RRSI `3,600s`；Train 批量判、Val 逐条单判（`per_criterion`），主报 `micro_avg` |
| `workspace_dualgate.yaml` | `workspace_dualgate_codex.yaml` | 合并 191 训 / 48 验双门控 | 分别沿用 Claw 与 Harvey 原生时钟/轮数；Val 配对 2 次，要求硬分不退且 `micro_avg >= +0.004` |

### （2）单题多步程序演化赛道（`program_search`，3 份）
| 配置文件 | 任务划分 | 官方原生预算与评估设定 |
|---|---|---|
| `frontier_eng_dev.yaml` | Frontier-Eng 开发/开放路由 | 官方 `v1.yaml` 逐题 `300 / 600 / 1,200s` 评估超时，跨重启持久 `100` 槽账本，上游 `max_tokens: 32768` |
| `tdc_admet.yaml` | TDC 15 Dev + 3 Val | 官方 Scaffold 切分 × `seeds: [1..5]` × 22 端点各自指标，MolRSI 每端点 `20` 槽，断网拟合 `900s`，默认 `64K` 输出 |
| `kernelbench_train.yaml` | KernelBench 30 训 + 11 验 | 原论文 `oracle_budget: 10`（10 轮精炼），编译/测速各 `180s`，`5 corr / 100 perf`，`3 warmup + 256MiB L2`，`1e-4 / 1e-2` 容差 |

### （3）封存测试集配置（10 份，默认锁死）
`tb41_sealed.yaml` (66)、`tbs01_sealed.yaml` (70)、`swebench_verified_sealed.yaml` (500)、`frontier_eng_sealed.yaml` (47)、`tdc_toxicity_sealed.yaml` (4)、`clawgym_bench_sealed.yaml` (200)、`harvey_lab_sealed.yaml` (40)、`jobbench_sealed.yaml` (65)、`gdpval_sealed.yaml` (220)、`kernelbench_sealed.yaml` (24)。
所有封存配置均设置 `allow_sealed_test: false`，在创建容器和请求模型前直接抛 `PermissionError`（退出码 `2`），防止日常开发调试误跑测试集。

---

## 2. 常用运行参数与环境变量

- **双模式切换**：
  - 正式运行：`flashrsi run --config configs/suite/<name>.yaml`（输出至 `runs/suite/<name>`）
  - 快速冒烟：`flashrsi run --config configs/suite/<name>.yaml --smoke`（自动切单题、单种子轻量配置，输出至 `runs/suite_smoke/<name>`）
  - 命令行临时换壳：追加 `--agent.name=codex`、`--agent.name=claude_code` 或 `--agent.name=pi_agent`
- **关键环境变量**：
  - `FLASHRSI_MAX_CONCURRENCY`：批内并发容器数（建议重型终端/办公设 `2–4`，工程设 `8`，TDC 内部多 seed 自并发设 `1–2`）
  - `FLASHRSI_INFRA_RETRIES`：遇到 HTTP `429/503`、断流或 Docker 偶发异常时的指数退避重试次数（推荐设为 `2`；耗尽后记 `InfraError` 停下，不会污染 `0.0` 分）
  - `GEMINI_API_KEY`：跑 `harvey_workspace` / `workspace_dualgate` 正式 Val 独立 Gemini 逐条裁判时在运行环境中配置
  - `FLASHRSI_GPU_QUEUE` / `FLASHRSI_GPU_UUID` / `FLASHRSI_GPU_AUTH_TOKEN`：跑 `kernelbench_train.yaml` 或 Frontier GPU 题时指向远程 `scripts/gpu_worker_daemon.py`

---

## 3. 如何接入我们自己的 RSI 算法？

在 `FlashRSI` 中研究新演化算法无需改动任何 Benchmark 或容器沙盒代码：
1. 在 `flashrsi/algorithms/` 下新建子类继承 `flashrsi.core.algorithm.Algorithm`（可参考 `flashrsi/algorithms/promotion.py` 中的 `TwoGateWorkspacePromotion` 或 `flashrsi/algorithms/gepa/__init__.py`）。
2. 在 `evolve()` 中通过 `self._evaluate(snapshot, tasks)` 获取逐题得分与脱敏轨迹，调用 `self.worker.ask()` 或 `self.worker.edit()` 生成新候选，通过验证集门控后调用 `self._promote(candidate, incumbent)`。
3. 在 `flashrsi/algorithms/__init__.py`（及 `flashrsi/runner.py` 注册表）登记算法短名后，将任意 `configs/suite/*.yaml` 中的 `algorithm.name` 改为新名字即可在全部五条赛道上直接开跑对比。
