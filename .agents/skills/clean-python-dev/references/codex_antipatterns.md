# Common AI / Codex Anti-Patterns vs. Clean Idioms

Coding agents (especially `codex-cli` under tight prompt constraints) frequently take shortcuts that pass immediate tests while degrading long-term maintainability. Eliminate the patterns below on sight.

---

## 1. Hostname / Ticket-Tainted Naming & Dev Narrative

### ❌ Bad (Codex Default)
```python
# flashrsi/benchmarks/cc1_design.py
"""Stage 2 (#156) Slice C2 calibration for cc1."""
ROOT = Path("/home/cc1/flashrsi-suite/envs/eng_and_mol")
```
- Encodes a specific server (`cc1`), chat sequence (`#156`), and temporary slice (`C2`) into permanent package filenames, docstrings, and hardcoded paths. Fails immediately on another machine.

### ✅ Good (`typoverflow` Idiom)
```python
# flashrsi/benchmarks/admet.py
"""TDC ADMET 22-endpoint molecular property benchmark.

Evaluates candidate workflows inside offline containers against scaffold-split
validation folds; test folds remain isolated inside the host scorer.

Environment:
  FLASHRSI_DATA_ROOT  Root directory for prepared benchmark payloads.
"""
from __future__ import annotations
import os
from pathlib import Path

def default_data_root() -> Path:
    if env_root := os.environ.get("FLASHRSI_DATA_ROOT"):
        return Path(env_root).expanduser().resolve()
    return Path(__file__).resolve().parents[2] / "artifacts" / "admet"
```

---

## 2. Classes Nested Inside Functions (Closure Classes)

### ❌ Bad (Codex Default)
```python
def _harbor_benchmark(spec, root, rows, review, *, n_attempts):
    class HarborTrack(Benchmark):
        primary_metric = "reward"
        def load(self):
            for row in rows:  # captures 5 local variables by closure!
                ...
    return HarborTrack()
```
- `HarborTrack` cannot be imported, type-checked, pickled for multiprocessing, or unit-tested in isolation.

### ✅ Good (`typoverflow` Idiom)
```python
class HarborBenchmark(Benchmark):
    """Pinned Harbor task suite evaluated through isolated verifiers."""

    primary_metric = "reward"

    def __init__(
        self,
        spec: SplitSpec,
        data_root: Path,
        rows: Sequence[Mapping[str, Any]],
        *,
        n_attempts: int = 5,
    ) -> None:
        super().__init__()
        self.spec = spec
        self.data_root = data_root
        self._rows = tuple(rows)
        self.n_attempts = n_attempts
```

---

## 3. Code-Golfed One-Liners & Custom `_require` Wrappers

### ❌ Bad (Codex Default)
```python
def _require(c: bool, m: str) -> None:
    if not c: raise ValueError(m)

def load(p): return json.loads(p.read_bytes())

selected = {k: [self.task(n, k) for n in self.selected(k, v)] for k, v in ids.items() if v}
values, = cursor.fetchall()
```
- Cryptic single-letter variables, tuple-unpacking traps (`ValueError: not enough values to unpack` with zero context), and unnecessary indirection (`_require` hides stack frames and confuses static type narrowers like `mypy`/`pyright`).

### ✅ Good (`typoverflow` Idiom)
```python
rows = cursor.fetchall()
if len(rows) != 1:
    raise ValueError(f"Expected exactly 1 row for {task_id!r}, got {len(rows)}")
row = rows[0]

splits: dict[str, list[Task]] = {}
for split_name, task_ids in manifest_ids.items():
    chosen_ids = self.select_ids(split_name, task_ids)
    splits[split_name] = [self.build_task(task_id, split_name) for task_id in chosen_ids]
```

---

## 4. 50-Line Manual `type(x) is not int` Walls in `@dataclass`

### ❌ Bad (Codex Default)
```python
@dataclass(frozen=True)
class WorkspaceSettings:
    solver_timeout: int = 660
    checker_timeout: int = 120
    cpus: int = 2
    # ... 15 more fields ...
    def __post_init__(self):
        for name in ("solver_timeout", "checker_timeout", "cpus", ...):
            val = getattr(self, name)
            if type(val) is not int or val < 1:
                raise ValueError(f"{name} must be positive int")
```
- Verbose, brittle (`type(val) is not int` rejects `np.int64` or subclasses unnecessarily unless checking `bool`), and drowns the actual domain defaults.

### ✅ Good (`typoverflow` Idiom)
Keep `@dataclass` definitions clean and declarative; if parsing untrusted YAML/JSON dicts, share ONE concise reusable converter/validator helper in `config/base.py`:
```python
def require_positive_int(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return value
```

---

## 5. Modifying Upstream Code In-Place Instead of Extension Hooks

### ❌ Bad (Codex Default)
- Copying 5,000 lines of third-party library code into the main package and scattering undocumented edits across 12 files.

### ✅ Good (`typoverflow` Idiom — see `codemimic/slime_ext/qwen3_vl.py`)
- Keep upstream in an unmodified `third_party/` submodule; place project-specific adapters in `<pkg>/<upstream>_ext/` with a clear docstring explaining the upstream gap and a single `register() -> None` entry point.
