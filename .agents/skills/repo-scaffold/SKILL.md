---
name: repo-scaffold
description: >-
  Scaffold or reorganize Python, ML/RL, and Agent/Benchmark repositories using
  clean consumer-driven package layouts, isolated uv dependency contexts, thin
  Makefile facades, and AGENTS.md / PLAN.md / BUILD.md documentation boundaries
  (distilled from typoverflow's RLE-Bench, flow-rl, codemimic, and OfflineRL-Lib).
  Activate this skill when initializing a new repository, adding a new subsystem
  or benchmark family, Untangling dependency conflicts, or cleaning up a messy
  project directory structure.
---

# Repository Architecture & Scaffolding (`repo-scaffold`)

Apply these structural rules whenever creating a new repository, adding a major subsystem, or reorganizing an existing codebase.

---

## 1. Core Placement Rule: "One Rule Places Every File"

Decide every file's location by **what consumes it** and **whether it is code, configuration, data, or a generated artifact**:

```text
<repo-root>/
├── <pkg>/                   # ONE top-level Python package for all first-party code
│   ├── __init__.py          # Exposes __version__ (single version source of truth)
│   ├── cli.py               # Unified CLI entry point ([project.scripts]); thin dispatch
│   ├── types.py             # Shared type aliases & lightweight protocols
│   ├── core/                # Stateless, task-agnostic logic & math (stdlib + numpy/jax/torch only)
│   ├── config/              # Typed @dataclass configuration schemas (no execution logic)
│   ├── <domain>/            # 1 entity / algorithm / service component = 1 module file
│   └── <upstream>_ext/      # Extensions to third-party frameworks via explicit register() hooks
├── configs/ or examples/    # Concrete YAML hyperparameter files & thin scenario launch scripts
├── scripts/                 # Idempotent environment setup or data-prep entry scripts
├── tests/                   # Host-side pytest suites (fast CPU unit tests vs @pytest.mark.heavy)
├── third_party/             # Unmodified git submodules or gitignored vendor clones
├── artifacts/               # Gitignored generated datasets, indexes, and staged payloads
├── outputs/ or runs/        # Gitignored logs, checkpoints, and evaluation trajectories
├── AGENTS.md                # Authoritative repo map, hard invariants, and green commands
├── CLAUDE.md                # Single line: @AGENTS.md
├── Makefile                 # Thin, predictable command facade over Python entry points
├── pyproject.toml           # Minimal core dependencies + isolated optional-dependencies
└── .pre-commit-config.yaml  # Formatting, AST/YAML/TOML validation, and secret detection
```

### Non-Negotiable Layout Invariants
1. **No Dumping-Ground Modules**: Never cram unrelated domains, benchmarks, or algorithms into a single 500+ line grab-bag file. One algorithm, benchmark adapter, or service worker gets one dedicated module (`<pkg>/<domain>/<name>.py`).
2. **Zero Hostname or Prompt/Task IDs in Paths**: Never name directories, modules, configs, or tests after a server hostname (`cc1_*.py`), temporary branch (`tmp_*`), or chat/issue slice (`test_s3_*.py`, `slice_c1.py`). Name everything by timeless domain role.
3. **Never Modify `third_party/` Submodules In-Place**: Keep upstream submodules byte-identical to upstream commits. Put custom loaders, model providers, or patches in `<pkg>/<upstream>_ext/` and wire them via import-path hooks or an explicit `register() -> None` function.
4. **Generated Trees & Artifacts**:
   - Large datasets, model weights, staged container contexts, and run outputs (`artifacts/`, `outputs/`, `runs/`, `jobs/`) MUST be in `.gitignore`.
   - If generated configs or manifests are tracked in Git, mark them as generator-owned: **never hand-edit them**. Edit the generator script, re-emit, and verify stability with `git diff --exit-code` in `make check-*`.

---

## 2. Dependency Context Isolation (`uv` + Optional Extras)

ML, simulator, and agent evaluation stacks frequently have mutually incompatible version pins (e.g., `mujoco` / `numpy` pins vs. training frameworks or `harbor`). Never force conflicting stacks into a single fragile environment:

1. **Minimal Base `[project].dependencies`**:
   - Keep orchestrator CLIs and lightweight HTTP/RPC clients (`client.py`) stdlib-only or minimal (`httpx`, `pyyaml`) so they can be imported anywhere without pulling GPU or simulator wheels.
2. **Split Execution Contexts Explicitly**:
   - Declare per-context extras in `pyproject.toml` (`[project.optional-dependencies]`: `dev`, `sim`, `prep`, `train-client`, etc.).
   - Use **`uv`** (`uv venv` + `uv pip install`), never bare `python3 -m venv` (host Pythons routinely lack `ensurepip` or resolve slow/conflicting indexes).
   - When PyTorch custom wheel indexes conflict with PyPI metadata, install them in separate `uv pip install` steps rather than merging index URLs globally.
   - Document distinct execution contexts (`host`, `sim`, `prep`, `train` container) in `AGENTS.md` / `BUILD.md`, communicating across conflicting Python stacks via clean HTTP/UDS JSON protocols or subprocesses.
3. **Single Source of Truth for Pins**:
   - Each stack's version pins live in **one** file (`requirements.txt` or `sim/<layer>/pins.env`), parity-checked by tests against any `Dockerfile` or generator that consumes them.

---

## 3. Documentation Trio: `AGENTS.md`, `PLAN.md`, and `BUILD.md`

Prevent documentation sprawl (never scatter dozens of `turn_1_report.md` or `phase2_notes.md` files into a repository):

1. **`AGENTS.md` + `CLAUDE.md` (`@AGENTS.md`)**:
   - Keep `AGENTS.md` concise (< 150 lines) and evergreen:
     - **What this is**: 1 short paragraph.
     - **Hard invariants**: 3–5 non-negotiable correctness/isolation rules (e.g., agent-verifier boundary, determinism, single source of truth for thresholds).
     - **Repo layout**: Compact tree explaining what belongs where.
     - **Commands (keep these green)**: Exact `make` / `pytest` / CLI commands.
   - Create `CLAUDE.md` with the exact single line `@AGENTS.md` so Claude Code, Codex, and Antigravity read the same file.
   - **Zero development narrative** in `AGENTS.md`, `README.md`, or code comments (do not write *"Previously we used X, now we changed to Y"*).
2. **`PLAN.md` vs. `BUILD.md`** (for multi-milestone builds):
   - `PLAN.md`: Confirmed I/O contracts, architecture decisions, and milestone checkboxes (`M0`, `M1`, `M2`... following `implement → verify → check box with concise metrics → stop for review`).
   - `BUILD.md`: Authoritative current layout, execution environments, launch commands, and non-obvious operational gotchas.

---

## 4. Scaffolding Steps & Templates

When initializing or standardizing a repository:

1. Copy and adapt templates from [resources/](./resources/):
   - [resources/AGENTS.md.tmpl](./resources/AGENTS.md.tmpl) (+ write `@AGENTS.md` to `CLAUDE.md`)
   - [resources/pre-commit-config.yaml](./resources/pre-commit-config.yaml) -> `.pre-commit-config.yaml`
   - [resources/Makefile.tmpl](./resources/Makefile.tmpl) -> `Makefile`
   - [resources/pull_request_template.md](./resources/pull_request_template.md) -> `.github/pull_request_template.md`
2. Verify directory boundaries:
   - Ensure `.gitignore` covers `.venv*`, `__pycache__/`, `*.pyc`, `dist/`, `build/`, `*.egg-info/`, `.env`, `third_party/` (non-submodule downloads), `artifacts/`, `outputs/`, `runs/`, `jobs/`.
   - Run `make lint` and `make test` to confirm the clean baseline is green before adding features.
