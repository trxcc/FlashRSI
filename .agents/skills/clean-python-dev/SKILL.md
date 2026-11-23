---
name: clean-python-dev
description: >-
  Write, refactor, and review clean, modular, strongly-typed Python code
  following typoverflow's engineering conventions (RLE-Bench, flow-rl,
  OfflineRL-Lib, codemimic) while eliminating common AI/Codex anti-patterns
  (nested classes, code-golfed one-liners, manual type-check walls, hardcoded
  host paths, and session-named modules). Activate this skill whenever writing,
  modifying, or reviewing Python source files, configs, benchmarks, or tests.
---

# Clean Python Engineering (`clean-python-dev`)

Follow these standards whenever authoring, refactoring, or reviewing Python code. Before reading or writing non-trivial code, consult [references/codex_antipatterns.md](./references/codex_antipatterns.md) if refactoring agent-generated code, and always run [scripts/audit_code_hygiene.py](./scripts/audit_code_hygiene.py) before finishing.

---

## 1. Module Design & Naming

1. **One Cohesive Responsibility Per File**:
   - One algorithm (`sdac.py`, `fpopp.py`), benchmark adapter (`harbor.py`, `swebench.py`), or service component (`executor.py`, `worker.py`, `server.py`, `client.py`) per file.
   - Keep files under **~400 lines** (hard ceiling **600 lines**) and functions under **~50 lines** (hard ceiling **80 lines**). Split pure data/math helpers into `core/` or `functional/`.
2. **Timeless Domain Names Only**:
   - Name files, classes, functions, and tests strictly after their **domain concept** (`admet.py`, `test_harbor_clocks.py`).
   - **Never** embed server hostnames (`cc1_*.py`), temporary branch names (`tmp_*`), or prompt/chat/issue labels (`s3`, `c1`, `slice2`, `A4`) in module, class, variable, or test names.
3. **Top-Level Module Docstring (5–12 Lines)**:
   - State: (a) what this module owns, (b) its import/privilege boundary (e.g., *"Import-time deps are stdlib + numpy only"*, *"Verifier-only; never imported inside solver container"*), and (c) honored environment variables if any.

---

## 2. Python Idioms, Types, & Configurations

1. **Imports & Type Annotations**:
   - Start with `from __future__ import annotations`.
   - Group imports strictly (`stdlib` -> `third-party` -> `local package`, sorted via `ruff check --select I` or `isort --profile black`).
   - Annotate all public function signatures and return types (`X | None`, `Sequence[T]`, `Mapping[str, Any]`, plus domain aliases in `<pkg>/types.py`).
   - Lazy-import heavy optional dependencies (`torch`, `mujoco`, `matplotlib`, `harbor`) inside the functions that require them when the module's CLI/metadata surface must stay lightweight.
2. **Flat Top-Level Definitions — Zero Closure Classes or Code Golf**:
   - **Never** define `class` objects inside functions/closures (breaks pickling, static analysis, and unit testing). Pass configuration via `__init__` attributes on a top-level class instead.
   - **Never** write code-golfed one-liners (`values, = cursor.fetchall()`, triple-nested comprehensions, or custom 1-word global assertion wrappers like `_require(cond, msg)`). Write explicit, readable `if not condition: raise ValueError("...")` statements.
3. **Clean `@dataclass` Configs & Single Source of Truth**:
   - Define configuration schemas as clean `@dataclass(frozen=True, slots=True)` types in `<pkg>/config/` (or at module top), separate from YAML hyperparameter instances and execution logic.
   - Keep every numeric constant, timeout, or version pin in **one** canonical place — never copy-paste magic numbers across modules.
   - Avoid 40-line `for field in ...: if type(val) is not int:` boilerplate walls inside `__post_init__`; check only genuine domain invariants (`if self.timeout <= 0: raise ValueError(...)`).
   - **Never** hardcode machine-specific absolute paths (`/home/...`, `/Users/...`, `/scratch/...`). Resolve paths relative to repo root or an explicit environment variable / config field.
4. **Atomic Persistence & State Isolation**:
   - Write generated JSON/manifest/store files atomically via a sibling temporary file + `os.replace()`. In multi-file episode/artifact directories, write the index/metadata marker file **last** as the completeness sentinel.
   - When reusing expensive stateful workers/simulators across evaluations, snapshot and restore mutable Python/C++ state around every reset so state never leaks across trials.
5. **Concise "Why-Only" Comments**:
   - Do **not** narrate obvious code (`# Increment counter`) or leave development history (`# Changed from 900s in Stage 2`).
   - Comment **only** non-obvious physical, protocol, or OS invariants (e.g., UID ownership handover on root-rendered Docker bind mounts, tensor padding alignment at `TP > 1`).

---

## 3. Verification & 4-Probe Service Contract

1. **Pure Functions First**:
   - Extract scoring formulas, losses, data transforms, and parsers as pure stateless functions with fast CPU unit tests (`< 2s`, zero Docker/network/API credentials).
2. **4-Probe Contract for Evaluators, Sandboxes, & Services**:
   Whenever building a verifier, code executor, or reward service, test all four probes before scaling up:
   - **Golden / Oracle probe**: Reference solution scores `≈ 1.0` (expected ceiling).
   - **Zero / NOP probe**: Empty or trivial submission scores `0.0` (or known analytic floor) — proves no state leak gives free reward.
   - **Exception / Garbage probe**: Syntax error or raised exception scores `0.0` and the worker/server survives cleanly.
   - **Timeout / Hang probe**: `while True:` / `sleep(999)` is killed at deadline, leaves zero orphan subprocesses/containers, and distinguishes task failure from `InfraError` (`HTTP 402/429/5xx`, Docker daemon failure).
3. **Mandatory Automated Hygiene Gate**:
   Run the hygiene script on modified files before committing or handing off:
   ```bash
   python3 .agents/skills/clean-python-dev/scripts/audit_code_hygiene.py <repo_or_paths...>
   ```
   Fix all reported errors (`exit code != 0`) and review any bloat warnings before marking work complete.
