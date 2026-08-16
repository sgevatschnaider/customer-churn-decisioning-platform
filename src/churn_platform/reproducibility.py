"""Exact run-lineage metadata for public, reproducible executions."""

from __future__ import annotations

import hashlib
import os
import platform
import re
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from churn_platform.config import PROJECT_ROOT, project_path

COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")
PUBLIC_LOCAL_TRACKING_LOCATION = "<local-mlruns>"


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 digest of a local file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_commit() -> str:
    """Resolve the public source commit, with an explicit run override when supplied."""
    explicit = os.getenv("CHURN_PLATFORM_SOURCE_COMMIT")
    if explicit:
        if not COMMIT_PATTERN.fullmatch(explicit):
            raise ValueError("CHURN_PLATFORM_SOURCE_COMMIT must be a 40-character Git SHA")
        return explicit
    try:
        worktree = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        if worktree.stdout.strip():
            return "uncommitted-local-run"
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        commit = result.stdout.strip()
        return commit if COMMIT_PATTERN.fullmatch(commit) else "uncommitted-local-run"
    except (OSError, subprocess.CalledProcessError):
        return "uncommitted-local-run"


def configuration_hashes() -> dict[str, str]:
    """Hash every executable YAML configuration used by the pipeline."""
    return {path.name: sha256_file(path) for path in sorted(project_path("configs").glob("*.yaml"))}


def dependency_lock_identifier(path: str | Path = "requirements.lock") -> str:
    """Identify the deterministic dependency resolution used for a run."""
    lock_path = project_path(path)
    return sha256_file(lock_path) if lock_path.exists() else "missing-lockfile"


def public_repository_path(path: str | Path) -> str:
    """Return a portable repository path without exposing the host filesystem."""
    candidate = Path(path)
    if not candidate.is_absolute():
        return str(path).replace("\\", "/")
    try:
        return candidate.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return "<external-file>"


def public_tracking_reference(tracking_uri: str) -> dict[str, str]:
    """Describe an MLflow backend without publishing local paths or hostnames."""
    parsed = urlparse(tracking_uri)
    if parsed.scheme == "file":
        return {
            "tracking_backend": "local-file-store",
            "tracking_location": PUBLIC_LOCAL_TRACKING_LOCATION,
        }
    if parsed.scheme in {"http", "https"}:
        return {
            "tracking_backend": "tracking-server",
            "tracking_location": "<configured-tracking-server>",
        }
    return {
        "tracking_backend": "configured-backend",
        "tracking_location": "<runtime-configured>",
    }


def build_run_metadata(source: str, dataset_path: str | Path) -> dict[str, Any]:
    """Build environment, source, data, and configuration lineage for one execution."""
    dataset = Path(dataset_path)
    return {
        "source_commit": source_commit(),
        "execution_timestamp_utc": datetime.now(UTC).isoformat(),
        "configuration_hashes": configuration_hashes(),
        "dataset_sha256": sha256_file(dataset),
        "dataset_path": public_repository_path(dataset),
        "source_mode": source,
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "dependency_lock_identifier": dependency_lock_identifier(),
    }
