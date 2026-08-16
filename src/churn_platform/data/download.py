"""Reproducible acquisition of the official UCI Online Retail dataset."""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
import urllib.request
import zipfile
from datetime import UTC, datetime
from pathlib import Path

from churn_platform.config import project_path

LOGGER = logging.getLogger(__name__)


class DownloadError(RuntimeError):
    """Raised when acquisition or archive integrity validation fails."""


def sha256_file(path: str | Path) -> str:
    """Calculate a streaming SHA-256 checksum."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_dataset(
    url: str,
    zip_path: str | Path,
    xlsx_path: str | Path,
    expected_sha256: str | None = None,
) -> dict[str, str | int]:
    """Download, validate, and extract the UCI workbook without redistributing it."""
    archive = project_path(zip_path)
    workbook = project_path(xlsx_path)
    archive.parent.mkdir(parents=True, exist_ok=True)
    workbook.parent.mkdir(parents=True, exist_ok=True)

    if not archive.exists():
        LOGGER.info("Downloading official UCI archive from %s", url)
        temporary = archive.with_suffix(".download")
        request = urllib.request.Request(url, headers={"User-Agent": "churn-platform/1.0"})
        try:
            with (
                urllib.request.urlopen(request, timeout=120) as response,
                temporary.open("wb") as destination,
            ):
                shutil.copyfileobj(response, destination)
            temporary.replace(archive)
        except Exception as exc:
            temporary.unlink(missing_ok=True)
            raise DownloadError(f"Could not download {url}: {exc}") from exc

    checksum = sha256_file(archive)
    if expected_sha256 and checksum.lower() != expected_sha256.lower():
        raise DownloadError(
            f"Checksum mismatch for {archive}: expected {expected_sha256}, observed {checksum}"
        )

    try:
        with zipfile.ZipFile(archive) as bundle:
            bad_member = bundle.testzip()
            if bad_member:
                raise DownloadError(f"Corrupt member in UCI archive: {bad_member}")
            candidates = [name for name in bundle.namelist() if name.lower().endswith(".xlsx")]
            if len(candidates) != 1:
                raise DownloadError(f"Expected one XLSX file, found {len(candidates)}")
            with bundle.open(candidates[0]) as source, workbook.open("wb") as destination:
                shutil.copyfileobj(source, destination)
    except zipfile.BadZipFile as exc:
        raise DownloadError(f"Invalid ZIP archive: {archive}") from exc

    manifest = {
        "source_url": url,
        "archive_sha256": checksum,
        "archive_bytes": archive.stat().st_size,
        "workbook_sha256": sha256_file(workbook),
        "workbook_bytes": workbook.stat().st_size,
        "downloaded_at_utc": datetime.now(UTC).isoformat(),
    }
    manifest_path = archive.parent / "download_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    LOGGER.info("Validated UCI archive with SHA-256 %s", checksum)
    return manifest
