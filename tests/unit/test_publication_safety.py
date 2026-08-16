"""Repository-wide publication safety checks."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path


def _tracked_text_files(repository_root: Path) -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=repository_root,
        capture_output=True,
        check=True,
    )
    paths = result.stdout.decode("utf-8").split("\0")
    return [repository_root / path for path in paths if path]


def test_tracked_text_does_not_expose_personal_filesystem_paths() -> None:
    repository_root = Path(__file__).resolve().parents[2]
    separator = "/"
    forbidden = (
        "C:" + separator + "Users",
        "C:" + "\\" + "Users",
        "file:" + separator * 3 + "C:" + separator,
        separator + "Users" + separator,
        separator + "home" + separator,
    )
    violations: list[str] = []
    for path in _tracked_text_files(repository_root):
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if any(pattern in text for pattern in forbidden):
            violations.append(path.relative_to(repository_root).as_posix())
    assert not violations, f"Tracked files expose personal filesystem paths: {violations}"


def test_tracked_text_does_not_contain_common_secret_formats() -> None:
    repository_root = Path(__file__).resolve().parents[2]
    expressions = (
        "gh" + r"p_[A-Za-z0-9]{20,}",
        "github" + r"_pat_[A-Za-z0-9_]{20,}",
        "sk" + r"-[A-Za-z0-9]{20,}",
        "AKIA" + r"[0-9A-Z]{16}",
        "-----BEGIN " + r"(?:RSA|OPENSSH|EC) PRIVATE KEY-----",
    )
    secret_pattern = re.compile("|".join(expressions))
    violations: list[str] = []
    for path in _tracked_text_files(repository_root):
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if secret_pattern.search(text):
            violations.append(path.relative_to(repository_root).as_posix())
    assert not violations, f"Tracked files match a secret format: {violations}"
