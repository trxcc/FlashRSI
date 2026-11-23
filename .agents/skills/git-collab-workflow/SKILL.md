---
name: git-collab-workflow
description: >-
  Enforce clean Git commits, branch naming, pre-commit validation, parallel
  multi-agent git worktree lifecycles, unmodified third-party submodules, PR
  checklists, and version-pinned releases (distilled from typoverflow's
  RLE-Bench, flow-rl, OfflineRL-Lib, WiseRL, and codemimic). Activate this
  skill before committing changes, creating or merging branches/worktrees,
  managing submodules, opening pull requests, or tagging releases.
---

# Git & Multi-Agent Collaboration Workflow (`git-collab-workflow`)

Follow this workflow whenever staging, committing, branching, managing parallel agent worktrees, or preparing pull requests and releases.

---

## 1. Commit Granularity & Message Standard

1. **One Logical Change Per Commit**:
   - Separate structural refactors (`refactor`), feature additions (`feat`), bug fixes (`fix`), and documentation (`docs`) into distinct atomic commits.
   - Every commit must leave lint and fast host tests green.
2. **Commit Subject Format**:
   ```text
   <type>(<optional-domain-scope>): <concise lowercase imperative summary>
   ```
   - **Allowed `<type>`**: `feat`, `fix`, `refactor`, `perf`, `test`, `docs`, `build`, `ci`, `chore`, `milestone`.
   - **Length & Style**: `<= 72` characters, lowercase after `: `, imperative mood (`add`, `fix`, `resolve`, `remove` — not `added` or `fixes`), no trailing period.
   - **Examples**:
     - `feat(summarize): record per-trial status and timing in summary.json`
     - `fix(actor): preserve min/max logstd dtype under fp16`
     - `refactor(agent): separate online and offline policy modules`
     - `chore: bump version to v0.2.0`
   - **Strictly Forbidden in Commit Messages**:
     - Ephemeral chat/board sequence numbers (`#156` unless referencing a real GitHub Issue/PR), prompt slice labels (`A4`, `C1`, `s3`), or server hostnames (`cc1`).
     - Vague grab-bag subjects (`update files`, `fix issues`, `finalize changes`).
3. **Behavioral / Eval Version Bumps**:
   - In benchmark or evaluation repositories, edits to agent-visible instructions, Dockerfiles, timeouts, or scoring logic are behavioral changes — bump `[task].version` or package `__version__` in the same commit/PR.

---

## 2. Mandatory Pre-Commit Gate

Before running `git commit`, **always** stage only intended files (`git add <paths>`, never blind `git add -A` or `git add .` without checking `git status --short` first!) and run [scripts/pre_commit_gate.py](./scripts/pre_commit_gate.py):

```bash
python3 .agents/skills/git-collab-workflow/scripts/pre_commit_gate.py \
  --repo "<repo_root>" \
  --message "feat(scope): concise summary"
```

What the gate enforces automatically:
1. `git config user.name` and `user.email` are configured (avoids `Author identity unknown` failures on remote servers).
2. Zero `.env` files, build caches (`__pycache__`, `*.pyc`, `.part`), run logs (`outputs/`, `runs/`, `jobs/`), or files `> 1 MiB` staged.
3. Zero merge-conflict markers (`<<<<<<<`, `=======`, `>>>>>>>`), invalid JSON/TOML/YAML, hardcoded secrets, or hardcoded user/host paths in staged files.
4. `clean-python-dev`'s AST & Ruff/Pyflakes hygiene check passes on all staged `.py` files.
5. Commit message conforms to `<type>(<scope>): <summary>`.

If the repository has `.pre-commit-config.yaml` and `pre-commit` installed, run `pre-commit run` as well.

---

## 3. Branching, Parallel Worktrees, & Submodules

1. **Branch Naming**:
   - Use `<type>/<domain-topic>`: `feat/driftpo`, `fix/ensemble-linear-init`, `refactor/ppo-config`, `docs/quickstart`.
   - **Never** push or leave behind `tmp/*`, `test-*`, or `slice-*` branches.
2. **Parallel Multi-Agent Worktrees (`codex-swarm` / Subagents)**:
   When multiple agents edit a repository concurrently:
   - Partition ownership into **strictly disjoint file sets** before spawning workers.
   - Create isolated worktrees outside the main working tree:
     ```bash
     git worktree add ../wt-<topic> -b feat/<topic>
     ```
   - Each worker runs `pre_commit_gate.py` + unit tests inside its own worktree and commits cleanly onto `feat/<topic>`.
   - Coordinator / integrator merges `feat/<topic>` into the target branch, runs the cross-module test suite + generator drift check (`git diff --exit-code`), and **immediately cleans up**:
     ```bash
     git worktree remove ../wt-<topic>
     git branch -d feat/<topic>
     ```
   - Verify `git worktree list` shows only the main checkout and `git status --short` is empty before concluding.
3. **Third-Party Code via Unmodified `git submodule`**:
   - Vendor external frameworks under `third_party/<name>` via `git submodule add`, pinned to a clean upstream commit.
   - Never commit project-specific hacks inside a submodule; place adapters in `<pkg>/<name>_ext/` and wire via explicit hooks.

---

## 4. Pull Request & Release Checklist

1. **Pull Request Description** (keep concise — 1 short paragraph + honest verification tier):
   - **`## What`**: One paragraph explaining the problem and architectural change.
   - **`## Verification`**: State **which tier was actually executed** — never conflate mocked smoke checks with full hardware/model evaluations:
     - *Tier 1*: Fast host lint + CPU unit/contract tests (`make lint && make test`).
     - *Tier 2*: Generator stability (`git diff --exit-code`) + 4-probe fixture smoke (Golden / NOP / Exception / Timeout).
     - *Tier 3*: Real container / simulator / GPU / live model E2E run.
2. **Tagging & Releases (`vX.Y.Z`)**:
   - Assert single-source `__version__` in `<pkg>/__init__.py` (or `pyproject.toml`) matches tag `vX.Y.Z`.
   - Run `python -m build --sdist` and `twine check dist/*` before pushing tags; never push to remotes unless explicitly requested by the user.
