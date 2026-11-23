# FlashRSI 原生基准套件快速上手

原生套件提供 Terminal-Bench / Harbor、SWE-Gym、ClawGym / Harvey Workspace、EngDesign / Frontier-Eng、TDC ADMET 和 KernelBench 的本地评测配置。交互式任务默认使用 Claude Code 或 Codex。程序搜索任务使用 `program_search`。

## 启动一次训练集冒烟

在仓库根目录运行以下命令。准备好对应的本地资产和 Docker 镜像，并在宿主运行环境中配置 `DEEPSEEK_API_KEY`。

```bash
flashrsi run --config configs/suite/clawgym.yaml --smoke
flashrsi run --config configs/suite/clawgym_codex.yaml --smoke
flashrsi run --config configs/suite/clawgym.yaml --agent.name=codex --smoke
```

`--smoke` 默认只评测一个已审核的训练题或开放开发题。使用 `--limit` 调整任务数，或用 `--benchmark.task_ids` 选择明确的开放题目。冒烟输出写入 `runs/suite_smoke/<配置名>/`，正式运行输出写入 `runs/suite/<配置名>/`。

## 配置资产与模型

- 如果仓库与准备好的 `envs/` 同属一个上级目录，套件会自动定位资产。仓库内的默认回退目录是 `artifacts/envs/`。
- 使用 `FLASHRSI_SUITE_ENVS_ROOT` 为整个套件指定资产目录。单个赛道也接受 `benchmark.root` 和 `FLASHRSI_SUITE_HARBOR_TRACK_ROOT`、`FLASHRSI_SUITE_WORKSPACE_TRACK_ROOT`、`FLASHRSI_SUITE_ENG_AND_MOL_ROOT` 等覆盖项。
- Harvey Workspace 正式验证需要独立的 `GEMINI_API_KEY`。KernelBench 和 GPU 题使用 `FLASHRSI_GPU_QUEUE`、`FLASHRSI_GPU_UUID` 和 `FLASHRSI_GPU_AUTH_TOKEN`。
- `*_sealed.yaml` 配置会在加载资产或发起请求之前报 `PermissionError`。调试和迭代使用公开配置。

完整的配置清单、官方参数和算法接入方式见 [套件配置说明](../configs/suite/README.md)。

## 验证代码

```bash
pytest -q tests/test_agent_container_relays.py tests/test_runtime_limits.py \
  tests/test_sandbox_hardening.py tests/test_suite_unification.py \
  tests/test_suite_production_configs.py
```

设置 `FLASHRSI_SUITE_DOCKER_SMOKE=1` 并通过 `FLASHRSI_SUITE_DOCKER_IMAGE` 指定本地镜像，可运行容器内的脚本化模型响应测试。
