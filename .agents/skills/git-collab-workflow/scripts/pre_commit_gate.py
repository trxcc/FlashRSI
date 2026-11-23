#!/usr/bin/env python3
"""Pre-commit & commit-message gate for clean Git collaboration.

Checks before `git commit`:
1. Git author identity (`user.name` and `user.email`) is configured.
2. Staged file set is non-empty and excludes `.env*`, caches, logs, and files > 1 MiB.
3. Staged text files contain no conflict markers and valid JSON / TOML / YAML syntax.
4. Staged Python & config files pass `audit_code_hygiene.py`.
5. Optional `--message` adheres to `<type>(<optional-scope>): <summary>` without
   session/hostname breadcrumbs.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tomllib
from pathlib import Path

COMMIT_RE = re.compile(
    r"^(feat|fix|refactor|perf|test|docs|build|ci|chore|milestone)"
    r"(?:\([a-z0-9_./-]+\))?: [a-z0-9`\"'\[\(].{2,68}$"
)
BANNED_MSG_RE = re.compile(
    r"(?i)\b(?:cc[1]|slice[-_ ]?[0-9a-z]+|turn[-_ ]?\d+|wip|temp|tmp)\b"
)
CONFLICT_RE = re.compile(r"^(?:<{7}\s|={7}$|>{7}\s)", re.MULTILINE)
FORBIDDEN_PATH_RE = re.compile(
    r"(?:^|/)(?:\.env(?:\..+)?|__pycache__|.*\.pyc|.*\.part|outputs|runs|jobs)(?:/|$)"
)
ENV_EXAMPLE_ALLOW = {".env.example", ".env.sample", ".env.template"}


def git_out(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=repo, text=True).strip()


def validate_identity(repo: Path) -> list[str]:
    errors: list[str] = []
    for key in ("user.name", "user.email"):
        proc = subprocess.run(
            ["git", "config", "--get", key], cwd=repo, capture_output=True, text=True
        )
        if proc.returncode != 0 or not proc.stdout.strip():
            errors.append(
                f"Git {key} is not configured. Set `git config user.name` and `git config user.email`."
            )
    return errors


def validate_message(msg: str) -> list[str]:
    errors: list[str] = []
    subject = msg.strip().splitlines()[0]
    if len(subject) > 72:
        errors.append(f"Commit subject is {len(subject)} chars (> 72): {subject!r}")
    if subject.endswith("."):
        errors.append("Commit subject must not end with a period.")
    if not COMMIT_RE.match(subject):
        errors.append(
            f"Commit subject {subject!r} must match '<type>(<optional-scope>): <lowercase imperative summary>'."
        )
    if BANNED_MSG_RE.search(subject):
        errors.append(
            f"Commit subject {subject!r} contains ephemeral host/slice/temp token; describe the domain change instead."
        )
    return errors


def validate_structured_syntax(path: Path, text: str) -> list[str]:
    try:
        if path.suffix == ".json":
            json.loads(text)
        elif path.suffix == ".toml":
            tomllib.loads(text)
        elif path.suffix in {".yaml", ".yml"}:
            try:
                import yaml  # type: ignore

                yaml.safe_load(text)
            except ImportError:
                pass
    except Exception as exc:  # noqa: BLE001
        return [f"{path}: invalid {path.suffix} syntax: {exc}"]
    return []


def _staged_files(repo: Path) -> list[tuple[str, Path]]:
    # Inspect staged changes (Added / Copied / Modified / Renamed)
    raw = subprocess.check_output(
        ["git", "diff", "--cached", "--name-status", "-z"], cwd=repo
    ).decode("utf-8", errors="replace")
    tokens = [t for t in raw.split("\0") if t]

    staged_files: list[tuple[str, Path]] = []
    idx = 0
    while idx < len(tokens):
        status = tokens[idx]
        if status.startswith(("R", "C")):
            rel = tokens[idx + 2]
            idx += 3
        else:
            rel = tokens[idx + 1]
            idx += 2
        if not status.startswith("D"):
            staged_files.append((status, repo / rel))

    return staged_files


def _check_files(
    repo: Path,
    staged_files: list[tuple[str, Path]],
    max_kb: int,
    allow_tainted_existing_names: bool,
) -> list[str]:
    errors: list[str] = []
    hygiene_added: list[str] = []
    hygiene_modified: list[str] = []

    for status, path in staged_files:
        rel_posix = path.relative_to(repo).as_posix()
        if path.name not in ENV_EXAMPLE_ALLOW and FORBIDDEN_PATH_RE.search(rel_posix):
            errors.append(f"{rel_posix}: forbidden secret/cache/output file staged")
            continue
        if not path.is_file():
            continue
        size_kb = path.stat().st_size / 1024.0
        if size_kb > max_kb:
            errors.append(
                f"{rel_posix}: staged file is {size_kb:.1f} KiB (> {max_kb} KiB limit)"
            )

        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue

        if CONFLICT_RE.search(text):
            errors.append(f"{rel_posix}: unresolved Git merge conflict marker detected")
        errors.extend(validate_structured_syntax(path, text))

        if status.startswith("A"):
            hygiene_added.append(str(path))
        else:
            hygiene_modified.append(str(path))

    errors.extend(
        _check_hygiene(hygiene_added, hygiene_modified, allow_tainted_existing_names)
    )
    return errors


def _check_hygiene(
    added: list[str],
    modified: list[str],
    allow_tainted_existing_names: bool,
) -> list[str]:
    errors: list[str] = []
    hygiene_script = (
        Path(__file__).resolve().parents[2]
        / "clean-python-dev"
        / "scripts"
        / "audit_code_hygiene.py"
    )
    if hygiene_script.is_file():
        if added:
            proc = subprocess.run(
                [sys.executable, str(hygiene_script), *added],
                capture_output=True,
                text=True,
            )
            if proc.returncode != 0:
                errors.extend(
                    line
                    for line in proc.stderr.splitlines()
                    if line.startswith("[ERROR]")
                )
        if modified:
            cmd = [sys.executable, str(hygiene_script), *modified]
            if allow_tainted_existing_names:
                cmd.append("--skip-filename-check")
            proc = subprocess.run(cmd, capture_output=True, text=True)
            if proc.returncode != 0:
                errors.extend(
                    line
                    for line in proc.stderr.splitlines()
                    if line.startswith("[ERROR]")
                )

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run pre-commit and commit-message checks."
    )
    parser.add_argument("--repo", default=".", help="Git repository root")
    parser.add_argument("--message", help="Proposed commit message to validate")
    parser.add_argument(
        "--max-kb", type=int, default=1024, help="Max staged file size in KiB"
    )
    parser.add_argument(
        "--allow-tainted-existing-names",
        action="store_true",
        help="Skip filename token check on modified pre-existing files",
    )
    args = parser.parse_args()

    repo = Path(args.repo).resolve()
    errors: list[str] = []

    errors.extend(validate_identity(repo))
    if args.message:
        errors.extend(validate_message(args.message))

    staged_files = _staged_files(repo)
    if not staged_files:
        errors.append("No staged files found (`git diff --cached` is empty).")
    errors.extend(
        _check_files(repo, staged_files, args.max_kb, args.allow_tainted_existing_names)
    )

    if errors:
        for err in errors:
            print(f"[BLOCKED] {err}", file=sys.stderr)
        print(f"\nPre-commit gate FAILED with {len(errors)} issue(s).", file=sys.stderr)
        return 1

    print(f"Pre-commit gate PASSED ({len(staged_files)} staged file(s) verified).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
