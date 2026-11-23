# 工程代码与 Git 协作规范（参考 `typoverflow` 代表性仓库提炼）

本规范对照 [`typoverflow`](https://github.com/typoverflow) 主推代表仓库（[`RLE-Bench`](https://github.com/RLE-Bench/RLE-Bench)、[`flow-rl`](https://github.com/typoverflow/flow-rl)、[`OfflineRL-Lib`](https://github.com/LAMDA-RL/OfflineRL-Lib)、[`WiseRL`](https://github.com/typoverflow/WiseRL)、[`codemimic`](https://github.com/typoverflow/codemimic)）的目录架构、代码风格与协作流程整理，并针对 `Codex` 等编码 Agent 常见坏习惯做了硬性约束与自动化门禁检查。

---

## 1. 仓库目录与依赖隔离规范（对应技能：`repo-scaffold`）

### 1.1 “一条规则决定文件位置”
每个文件的落点只取决于**谁消费它**以及**它是源码、配置、数据还是生成产物**：

```text
<repo-root>/
├── <pkg>/                   # 唯一顶层 Python 包，承载全部自研业务代码
│   ├── __init__.py          # 唯一版本号真源 __version__
│   ├── cli.py               # 统一命令行入口（薄分发层，不堆业务逻辑）
│   ├── types.py             # 公共类型别名与轻量协议
│   ├── core/                # 无状态纯函数与核心评测/数学逻辑（仅依赖标准库 + numpy/jax/torch）
│   ├── config/              # 纯 @dataclass 配置定义（不含执行逻辑）
│   ├── <domain>/            # 一个算法 / 基准 / 服务组件对应一个独立模块文件
│   └── <upstream>_ext/      # 对第三方框架的扩展（通过显式 register() 挂载，不改上游源码）
├── configs/ 或 examples/    # 具体 YAML 超参文件与薄启动脚本
├── scripts/                 # 幂等的环境搭建或数据预处理入口脚本
├── tests/                   # 宿主机 pytest 单测（快速 CPU 单测与 @pytest.mark.heavy 重测试分离）
├── third_party/             # 保持与上游字节级一致的 git submodule，或已忽略的外部源码
├── artifacts/               # [Git 忽略] 生成的数据集、索引与打包载荷
├── outputs/ 或 runs/        # [Git 忽略] 运行日志、轨迹与模型权重
├── AGENTS.md                # 仓库权威说明、硬性不变量与保绿命令清单
├── CLAUDE.md                # 仅一行：@AGENTS.md（让 Claude Code / Codex / Antigravity 共用同一真源）
├── Makefile                 # 薄命令门面（install / lint / test / check）
├── pyproject.toml           # 最小核心依赖 + 按环境拆分的 optional-dependencies
└── .pre-commit-config.yaml  # 提交前自动格式、语法与密钥检查
```

### 1.2 目录与环境硬约束
1. **严禁按机器名或临时任务号命名文件**：禁止出现 `tmp_*` 等把服务器名或对话轮次写进包结构的名字；一律按领域职责命名（如 `admet.py`、`harbor.py`、`test_harbor_clocks.py`）。
2. **严禁大杂烩单文件**：不同基准、不同算法不得塞进同一个 500+ 行的文件；遵循 `flow-rl` / `OfflineRL-Lib` 风格，一个算法或基准适配器独立成文。
3. **冲突依赖分环境隔离（`uv` + Optional Extras）**：当仿真器、评测框架与训练栈存在版本冲突时（如不同 `mujoco` / `numpy` 锁死版本），一律通过 `uv venv` 建立独立虚拟环境，跨环境通过零依赖 HTTP/UDS `client.py` 通信，禁止强行合并到一个脆弱的大环境。
4. **生成产物严禁手改**：由脚本生成的清单、配置或任务目录，要么加入 `.gitignore` 通过 `make` 按需生成，要么在 CI / `make check` 中用 `git diff --exit-code` 校验零手工漂移。
5. **文档收敛，禁止过程报告堆满仓库**：仓库根目录只保留长青的 `AGENTS.md`（附 `@AGENTS.md` 的 `CLAUDE.md`）；多阶段研发项目用 `PLAN.md` 记录里程碑勾选（`M0`–`Mn`），用 `BUILD.md` 记录当前架构与启动方式，不在代码库里散落几十份临时汇报。

---

## 2. Python 代码与测试规范（对应技能：`clean-python-dev`）

### 2.1 专门根治 `Codex` 常见坏习惯的六条准则

| 维度 | ❌ `Codex` 常见反模式 | ✅ 规范写法（`typoverflow` 风格） |
| :--- | :--- | :--- |
| **路径与配置** | 在源码里硬编码 用户目录的绝对路径；同一超时或阈值在多处重复写死 | 仓库相对路径 + 环境变量/配置覆盖；所有版本号、阈值与超时只在**唯一真源**定义 |
| **类与函数层级** | 在函数内部嵌套定义 `class HarborTrack(Benchmark):`（闭包捕获局部变量，无法单测与序列化） | 所有类与核心辅助函数一律放在模块顶层，通过 `__init__` 显式传参 |
| **可读性 vs 炫技** | 自定义单字母/短名断言包装 `_require(c, m)`、三层嵌套推导式、`values, = fetchall()` | 标准直观的 `if not cond: raise ValueError(...)`，局部变量语义清晰，单函数不超过 80 行 |
| **配置类定义** | 在 `@dataclass` 的 `__post_init__` 里手写 50 行 `if type(x) is not int` 循环校验墙 | 保持 `@dataclass(frozen=True, slots=True)` 干净声明；边界解析共用统一的轻量转换函数 |
| **注释与文档串** | 缺少模块说明，或在代码里留 `# Fixed in Stage 2 (#156)` 等对话痕迹与废话注释 | 模块顶部 5–12 行 docstring 说明职责、依赖边界与环境变量；行内注释**只解释非直觉的底层原因**（如容器 root 挂载权限交接、TP>1 词表对齐） |
| **第三方库改动** | 直接复制几千行第三方源码进主包并到处打补丁 | `third_party/` 子模块保持原样，在 `<pkg>/<upstream>_ext/` 外置编写扩展并通过 `register()` 接入 |

### 2.2 分层测试与评测服务“四探针”契约
1. **纯函数优先**：打分公式、数据变换、解析器写成无状态纯函数，配秒级跑完的宿主机 CPU 单元测试（不依赖 Docker、GPU 或外网 API Key）。
2. **执行器 / 判分器四探针验收**（参考 `RLE-Bench` 与 `codemimic`）：任何沙盒执行器、判分器或奖励服务上线前必须通过四类探针：
   - **Golden / Oracle 探针**：标准参考解拿到预期满分（`≈ 1.0`）；
   - **Zero / NOP 探针**：空提交得 `0.0`（确认跨轮次重置无状态泄漏白送分）；
   - **Exception / 脏输入探针**：语法错误或抛异常得 `0.0` 且服务进程不崩溃；
   - **Timeout / 死循环探针**：超时进程被干净回收、无残留僵尸容器，且将 API 欠费/限流/5xx 等基础设施故障严格标记为 `InfraError`（绝不混同为业务 `0` 分）。

---

## 3. Git 提交、分支与多 Agent 协作规范（对应技能：`git-collab-workflow`）

1. **原子提交与 Commit Message 格式**：
   - 一次提交只做一件事（结构重构 `refactor`、新功能 `feat`、修 Bug `fix`、文档 `docs` 分开提交）。
   - 统一格式：`<type>(<optional-scope>): <lowercase imperative summary>`（不超过 72 字符，末尾不加句号）：
     - 示例：`feat(summarize): record per-trial status and timing in summary.json`
     - 示例：`fix(actor): preserve min/max logstd dtype under fp16`
     - 示例：`refactor(agent): separate online and offline policy modules`
   - **禁止出现**：主机名、临时任务号及 `WIP` 等含糊描述。
2. **分支与多 Agent 并行 Worktree 生命周期**：
   - 分支统一命名为 `feat/<topic>`、`fix/<topic>`、`refactor/<topic>`、`docs/<topic>`（禁止保留 `tmp/*` 分支）。
   - 多 Agent 并行开发时：先划分**互不重叠的文件所有权** → 各自在独立 `git worktree add ../wt-<topic> -b feat/<topic>` 中开发并通过单测与门禁脚本 → 合并回目标分支跑全量回归 → **立即执行 `git worktree remove` 与 `git branch -d` 清理干净**。
3. **行为变更必升版本号**：在评测基准仓库中，任何对 Agent 可见题面、容器环境、超时或判分逻辑的修改均属于行为变更，必须同步递增 `[task].version` 或包 `__version__`。
4. **提交前强制执行自动门禁脚本**：禁止盲目 `git add .`；按路径暂存后必须先跑过自动门禁脚本，拦截 `.env`、大于 `1 MiB` 的大文件、冲突标记、损坏的 YAML/TOML/JSON、泄露密钥、硬编码个人路径与嵌套类反模式。

---

## 4. 已安装的全局工程 Skills 与自动化检查脚本

以下 3 个工程 Skill 已内置于本仓库 `.agents/skills/` 目录（克隆仓库后可直接使用，也可软链至全局 `~/.gemini/config/skills/` 与 `~/.codex/skills/`）：

| Skill 名称 | 触发场景 | 内置模板 / 自动检查脚本 |
| :--- | :--- | :--- |
| **`repo-scaffold`** | 新建仓库、新增子模块/基准族、拆分冲突依赖环境、整理目录与 `AGENTS.md` | `.agents/skills/repo-scaffold/resources/`（含 `AGENTS.md`、`Makefile`、`.pre-commit-config.yaml`、PR 模板） |
| **`clean-python-dev`** | 编写、重构或 Review Python 代码与测试，消除 `Codex` 常见代码坏味道 | `.agents/skills/clean-python-dev/scripts/audit_code_hygiene.py` + `references/codex_antipatterns.md` |
| **`git-collab-workflow`** | Git 暂存提交、分支/Worktree 并行协作、Submodule 管理、PR 与发版打标 | `.agents/skills/git-collab-workflow/scripts/pre_commit_gate.py` |
