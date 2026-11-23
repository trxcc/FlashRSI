# AGENTS.md — FlashRSI

Instructions for coding agents (Claude Code, Codex, Antigravity) working in this repository. Read before editing.

## What this is

FlashRSI is a reproducible framework for evaluating and evolving agent harnesses and multi-step program-search algorithms across isolated benchmarks (Terminal-Bench / Harbor, SWE-bench, EngDesign & Frontier-Eng, TDC ADMET, ClawGym & Harvey Workspace, and KernelBench).

## Hard invariants — do not violate

1. **Solver–verifier isolation**: Solver containers and workspaces never mount verifier scripts, reference solutions, test patches, or raw provider API keys. Verifiers treat candidate patches, programs, and workspace deliverables as untrusted input.
2. **Sealed split protection**: Configs and splits marked `*_sealed` (`allow_sealed_test: false`) must refuse access before starting any container or model request. Never feed test trajectories, scores, or rubric feedback into evolution loops.
3. **Official benchmark settings**: Preserve each benchmark's official timeouts, turn caps (or uncapped turns), container resource quotas, seed sets, and primary metrics unless explicitly running a named ablation.
4. **InfraError vs. task zero**: Provider billing/rate-limit/5xx errors (`HTTP 402`, `429`, `5xx`), network drops, and Docker daemon failures must surface as `InfraError` — never record them as task `0.0` scores.
5. **Engineering & Git hygiene**: Follow `.agents/skills/repo-scaffold/SKILL.md`, `.agents/skills/clean-python-dev/SKILL.md`, and `.agents/skills/git-collab-workflow/SKILL.md`. Use timeless domain names (no hostnames or prompt/slice IDs in new filenames), flat top-level classes, repo-relative/env-configured paths, and concise "why-only" comments.

## Repo layout

```text
flashrsi/
  core/              Task, Trajectory, Benchmark, Agent, Algorithm, and InfraError primitives
  agents/            Target agent harnesses (claude_code, codex, program_search, pi_agent, ...)
  algorithms/        Pluggable evolution algorithms (ace, gepa, ahe, meta_harness, rrsi, ...)
  benchmarks/        Benchmark loaders, split locks, and grading contracts
  integrations/      Container sandboxes, model relays, Harbor/SWE bridges, GPU worker daemon
configs/suite/       Native benchmark YAML profiles and algorithm registration guide
docs/                Architecture, benchmark, algorithm, and engineering guides
scripts/             Standalone operational daemons and utilities
tests/               Host-side pytest suites
.agents/skills/      Workspace engineering skills and automated hygiene/pre-commit gates
```

## Commands (keep these green)

- `python3 .agents/skills/clean-python-dev/scripts/audit_code_hygiene.py <changed_paths...>` — static AST, path, secret, and modularity check
- `ruff check <changed_paths...>` — fast Python lint check
- `pytest -q` — run host test suite (`pytest -q tests/test_agent_container_relays.py tests/test_runtime_limits.py tests/test_sandbox_hardening.py tests/test_suite_unification.py tests/test_suite_production_configs.py` for core suite contracts)
- `python3 .agents/skills/git-collab-workflow/scripts/pre_commit_gate.py --message "<type>(<scope>): <summary>"` — mandatory gate before `git commit`
- `flashrsi run --config configs/suite/clawgym.yaml --smoke` — run one reviewed training task with native grading
