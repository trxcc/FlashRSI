"""Stage packaged Python dependencies for container harnesses.

The host supplements the portable runtime kit with its installed dependencies.
The ready marker is written after the tarball and dependency pins are complete.
"""

from __future__ import annotations

import tarfile
import tempfile
from collections.abc import Iterable
from contextlib import ExitStack
from importlib.metadata import distribution
from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name


def extend_runtime_kit(
    source: Path,
    dependencies: Iterable[str],
    resources: ExitStack,
) -> Path:
    """Supplement the portable kit with the installed native harness packages."""
    available = set((source / "requirements.txt").read_text().splitlines())
    pending = [Requirement(name) for name in ("packaging", *dependencies)]
    packages = {}
    seen = {}
    while pending:
        request = pending.pop()
        name = canonicalize_name(request.name)
        checked = seen.setdefault(name, set())
        extras = ({"", *request.extras}) - checked
        if not extras:
            continue
        checked.update(extras)
        if name not in packages:
            packages[name] = distribution(name)
        for entry in packages[name].requires or ():
            requirement = Requirement(entry)
            if requirement.marker is None or any(
                requirement.marker.evaluate({"extra": extra}) for extra in extras
            ):
                pending.append(requirement)

    pins = {
        f"{package.metadata['Name']}=={package.version}"
        for package in packages.values()
    }
    if pins <= available:
        return source
    target = Path(
        resources.enter_context(
            tempfile.TemporaryDirectory(prefix="flashrsi-runtime-kit-")
        )
    )
    (target / "python312.tar.gz").symlink_to((source / "python312.tar.gz").resolve())
    (target / "requirements.txt").write_text("\n".join(sorted(available | pins)) + "\n")
    with (
        tarfile.open(source / "site-packages.tar.gz", "r:gz") as original,
        tarfile.open(
            target / "site-packages.tar.gz", "w:gz", compresslevel=1
        ) as archive,
    ):
        copied = set()
        for member in original:
            archive.addfile(
                member, original.extractfile(member) if member.isfile() else None
            )
            copied.add(member.name)
        for package in packages.values():
            for member in package.files or ():
                if (
                    member.is_absolute()
                    or ".." in member.parts
                    or member.suffix == ".pyc"
                ):
                    continue
                path = package.locate_file(member)
                if (
                    str(member) not in copied
                    and path.is_file()
                    and not path.is_symlink()
                ):
                    archive.add(path, arcname=str(member), recursive=False)
                    copied.add(str(member))
    (target / ".ready").touch()
    return target
