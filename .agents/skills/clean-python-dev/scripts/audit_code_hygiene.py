#!/usr/bin/env python3
"""Static code-hygiene auditor for Python & config files.

Catches recurring AI/Codex anti-patterns before commit or review:
1. Syntax errors & Ruff / Pyflakes static issues (when installed).
2. Classes defined inside functions/closures (breaks pickling & unit testing).
3. Hardcoded user/server absolute paths (/Users/<name>, /home/<name>, /scratch/<name>).
4. Session/hostname-tainted filenames (host, slice_, turn_, tmp_).
5. Ephemeral ticket and phase text in comments.
6. Leaked credentials or private keys.
7. Oversized functions (> 80 lines) and monolithic modules (> 600 lines) [warnings].
"""

from __future__ import annotations

import argparse
import ast
import re
import shutil
import subprocess
import sys
from pathlib import Path

HARDCODED_PATH_RE = re.compile(
    r"""(?x)
    (?<![A-Za-z0-9_])
    (?:
        /Users/[A-Za-z0-9_.-]+/
      | /home/(?!user\b|ubuntu\b|runner\b|app\b|sandbox\b)[A-Za-z0-9_.-]+/
      | /scratch/[A-Za-z0-9_.-]+/
    )
    """
)

SECRET_RE = re.compile(
    r"""(?x)
    (?:
        \bsk-[A-Za-z0-9]{20,}\b
      | \bgh[pousr]_[A-Za-z0-9]{20,}\b
      | -----BEGIN\s+(?:RSA\s+|OPENSSH\s+|EC\s+)?PRIVATE\s+KEY-----
    )
    """
)

DEV_NARRATIVE_RE = re.compile(
    r"(?i)#.*\b(?:dispatch\s*#\d+|slice\s+[a-z]?\d+|stage\s+[12]\s+fix|changed\s+from\s+\d+)"
)

TAINTED_NAME_RE = re.compile(
    r"(?i)(?:^|_)(?:cc[1]|tmp|temp|slice[0-9a-z]*|turn_?\d+|patch_v\d+)(?:_|\.|$)"
)

TEXT_SUFFIXES = {".py", ".yaml", ".yml", ".toml", ".sh", ".json"}
SKIP_DIRS = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    "node_modules",
    "third_party",
    "build",
    "dist",
    ".pytest_cache",
    ".ruff_cache",
}


def collect_files(targets: list[str]) -> list[Path]:
    found: list[Path] = []
    for raw in targets:
        target = Path(raw).resolve()
        if target.is_file():
            if target.suffix in TEXT_SUFFIXES:
                found.append(target)
            continue
        if not target.is_dir():
            continue
        # Prefer git ls-files when inside a git repo so gitignored trees are skipped
        try:
            out = subprocess.check_output(
                [
                    "git",
                    "ls-files",
                    "--cached",
                    "--others",
                    "--exclude-standard",
                    "-z",
                    "--",
                    str(target),
                ],
                cwd=target,
                stderr=subprocess.DEVNULL,
            ).decode("utf-8", errors="replace")
            for rel in out.split("\0"):
                if not rel:
                    continue
                p = (target / rel).resolve()
                if (
                    p.is_file()
                    and p.suffix in TEXT_SUFFIXES
                    and not any(part in SKIP_DIRS for part in p.parts)
                ):
                    found.append(p)
            continue
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass

        for p in target.rglob("*"):
            if (
                p.is_file()
                and p.suffix in TEXT_SUFFIXES
                and not any(part in SKIP_DIRS for part in p.parts)
            ):
                found.append(p)
    return sorted(set(found))


def audit_python_ast(
    path: Path, source: str, max_fn_lines: int, max_file_lines: int
) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    lines = source.splitlines()
    if len(lines) > max_file_lines:
        warnings.append(
            f"{path}:1: module has {len(lines)} lines (> {max_file_lines}); split by responsibility"
        )

    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError as exc:
        errors.append(f"{path}:{exc.lineno or 1}: SyntaxError: {exc.msg}")
        return errors, warnings

    def visit_block(node: ast.AST, inside_fn: str | None = None) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                end_lineno = getattr(child, "end_lineno", child.lineno)
                span = end_lineno - child.lineno + 1
                if span > max_fn_lines:
                    warnings.append(
                        f"{path}:{child.lineno}: function '{child.name}' is {span} lines (> {max_fn_lines}); extract helpers"
                    )
                visit_block(child, inside_fn=child.name)
            elif isinstance(child, ast.ClassDef):
                line_text = (
                    lines[child.lineno - 1] if child.lineno <= len(lines) else ""
                )
                if inside_fn is not None and "noqa: nested-class" not in line_text:
                    errors.append(
                        f"{path}:{child.lineno}: class '{child.name}' defined inside function '{inside_fn}'; move to module top level"
                    )
                visit_block(child, inside_fn=inside_fn)
            else:
                visit_block(child, inside_fn=inside_fn)

    visit_block(tree)
    return errors, warnings


def audit_text_rules(
    path: Path, source: str, check_filenames: bool
) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []

    if check_filenames and TAINTED_NAME_RE.search(path.stem):
        errors.append(
            f"{path}:1: filename '{path.name}' contains host/session/temp token; use a timeless domain name"
        )

    for idx, line in enumerate(source.splitlines(), start=1):
        if "noqa: hygiene" in line:
            continue
        if SECRET_RE.search(line):
            errors.append(
                f"{path}:{idx}: potential hardcoded secret or private key detected"
            )
        if HARDCODED_PATH_RE.search(line):
            errors.append(
                f"{path}:{idx}: hardcoded user/host absolute path detected; use repo-relative or env var"
            )
        if path.suffix == ".py" and DEV_NARRATIVE_RE.search(line):
            warnings.append(
                f"{path}:{idx}: remove ephemeral ticket/slice narrative from code comments"
            )

    return errors, warnings


def run_external_linter(py_files: list[Path]) -> list[str]:
    if not py_files:
        return []
    ruff = shutil.which("ruff")
    if ruff:
        proc = subprocess.run(
            [
                ruff,
                "check",
                "--select",
                "E9,F63,F7,F82,F401",
                "--output-format",
                "concise",
                *[str(p) for p in py_files],
            ],
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            return [line for line in proc.stdout.splitlines() if line.strip()]
    return []


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit Python and config files for code hygiene."
    )
    parser.add_argument(
        "targets", nargs="*", default=["."], help="Files or directories to check"
    )
    parser.add_argument("--max-fn-lines", type=int, default=80)
    parser.add_argument("--max-file-lines", type=int, default=600)
    parser.add_argument("--skip-filename-check", action="store_true")
    parser.add_argument("--strict-warnings", action="store_true")
    args = parser.parse_args()

    files = collect_files(args.targets)
    all_errors: list[str] = []
    all_warnings: list[str] = []
    py_files: list[Path] = []

    for path in files:
        try:
            source = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        errs, warns = audit_text_rules(
            path, source, check_filenames=not args.skip_filename_check
        )
        all_errors.extend(errs)
        all_warnings.extend(warns)
        if path.suffix == ".py":
            py_files.append(path)
            ast_errs, ast_warns = audit_python_ast(
                path, source, args.max_fn_lines, args.max_file_lines
            )
            all_errors.extend(ast_errs)
            all_warnings.extend(ast_warns)

    all_errors.extend(run_external_linter(py_files))

    for w in all_warnings:
        print(f"[WARN]  {w}")
    for e in all_errors:
        print(f"[ERROR] {e}", file=sys.stderr)

    if all_errors or (args.strict_warnings and all_warnings):
        print(
            f"\nHygiene audit FAILED: {len(all_errors)} error(s), {len(all_warnings)} warning(s) across {len(files)} file(s).",
            file=sys.stderr,
        )
        return 1

    print(
        f"Hygiene audit PASSED: {len(files)} file(s) checked ({len(all_warnings)} warning(s))."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
