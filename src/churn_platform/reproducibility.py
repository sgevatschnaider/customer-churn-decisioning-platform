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

from churn_platform.config import PROJECT_ROOT, project_path

COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")


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


def build_run_metadata(source: str, dataset_path: str | Path) -> dict[str, Any]:
    """Build environment, source, data, and configuration lineage for one execution."""
    dataset = Path(dataset_path)
    return {
        "source_commit": source_commit(),
        "execution_timestamp_utc": datetime.now(UTC).isoformat(),
        "configuration_hashes": configuration_hashes(),
        "dataset_sha256": sha256_file(dataset),
        "dataset_path": str(
            dataset.relative_to(PROJECT_ROOT) if dataset.is_relative_to(PROJECT_ROOT) else dataset
        ),
        "source_mode": source,
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "dependency_lock_identifier": dependency_lock_identifier(),
    }
