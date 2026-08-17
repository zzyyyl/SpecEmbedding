#!/usr/bin/env python3
"""Build and verify a deterministic, explicitly allowlisted anonymous archive."""

from __future__ import annotations

import argparse
import getpass
import hashlib
import json
import os
import re
import socket
import subprocess
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ALLOWLIST = REPOSITORY_ROOT / "reproducibility" / "anonymous-allowlist.txt"
DEFAULT_OUTPUT = REPOSITORY_ROOT / "dist" / "specembedding-anonymous-supplement.zip"
ARCHIVE_ROOT = PurePosixPath("specembedding-supplement")
ARCHIVE_MANIFEST = PurePosixPath("ANONYMOUS_MANIFEST.json")
FIXED_ZIP_TIME = (1980, 1, 1, 0, 0, 0)

FORBIDDEN_TOP_LEVEL = {
    ".git",
    ".codegraph",
    ".codex",
    ".idea",
    ".venv",
    ".vscode",
    "analysis",
    "artifacts",
    "build",
    "checkpoints",
    "checkpoints_align",
    "checkpoints_rerank",
    "data",
    "dist",
    "docs",
    "logs",
    "outputs",
    "paper",
    "release",
    "rerank_cache",
    "results",
    "wandb",
}
FORBIDDEN_TOP_LEVEL_PREFIXES = (
    "artifact",
    "cache",
    "checkpoint",
    "log",
    "output",
    "result",
)
FORBIDDEN_PARTS = {
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
}
FORBIDDEN_FILENAMES = {
    "build-manifest.yaml",
    "runtime-snapshot.yaml",
    "status.json",
}
FORBIDDEN_SUFFIXES = {
    ".7z",
    ".bin",
    ".bz2",
    ".ckpt",
    ".csv",
    ".feather",
    ".gz",
    ".h5",
    ".hdf5",
    ".ipynb",
    ".joblib",
    ".log",
    ".npy",
    ".npz",
    ".onnx",
    ".parquet",
    ".pdf",
    ".pickle",
    ".pkl",
    ".pt",
    ".pth",
    ".safetensors",
    ".tar",
    ".tsv",
    ".xz",
    ".zip",
}
TEXT_SUFFIXES = {
    "",
    ".cfg",
    ".ini",
    ".json",
    ".lock",
    ".md",
    ".py",
    ".sh",
    ".txt",
    ".toml",
    ".yaml",
    ".yml",
}
STATIC_CONTENT_PATTERNS = (
    (
        "private home-directory path",
        re.compile(r"(?i)(?<![A-Za-z0-9])/(?:home|Users)/[A-Za-z0-9._-]+"),
    ),
    (
        "Windows user-directory path",
        re.compile(r"(?i)\b[A-Z]:\\Users\\[A-Za-z0-9._-]+"),
    ),
    (
        "private data root",
        re.compile(r"(?i)(?<![A-Za-z0-9])/data[0-9]+/"),
    ),
    (
        "named home-directory shorthand",
        re.compile(r"(?i)(?<![A-Za-z0-9])~[A-Za-z0-9._-]+[/\\]"),
    ),
    ("SSH Git remote", re.compile(r"(?i)\bgit" + r"@[A-Za-z0-9.-]+:")),
    (
        "personal email address",
        re.compile(r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b"),
    ),
    (
        "public project-hosting link",
        re.compile(
            r"(?i)https?://[^\s)>\]]*(?:github\.com|gitlab\.com|git\.ustc\.edu\.cn|"
            r"figshare\.com|huggingface\.co|hf\.co)[^\s)>\]]*"
        ),
    ),
    (
        "public Figshare DOI",
        re.compile(r"(?i)https?://doi\.org/10\.6084/m9\.figshare[^\s)>\]]*"),
    ),
    (
        "ORCID identifier",
        re.compile(r"(?i)\b(?:https?://orcid\.org/)?[0-9]{4}-[0-9]{4}-[0-9]{4}-[0-9X]{4}\b"),
    ),
    (
        "GPU UUID",
        re.compile(r"(?i)\bGPU-[0-9A-F]{8}(?:-[0-9A-F]{4}){3}-[0-9A-F]{12}\b"),
    ),
    (
        "private key material",
        re.compile(r"(?i)-----BEGIN (?:OPENSSH |RSA |EC )?PRIVATE KEY-----"),
    ),
    (
        "credential-like token",
        re.compile(r"\b(?:ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|AKIA[0-9A-Z]{16})\b"),
    ),
)
COMMON_RUNTIME_IDENTITIES = {"root", "runner", "user", "unknown", "localhost"}


class AnonymousArchiveError(RuntimeError):
    """Raised when the allowlist or archive violates the anonymity contract."""


@dataclass(frozen=True)
class AllowedFile:
    source_relative: PurePosixPath
    archive_relative: PurePosixPath
    source_path: Path
    data: bytes
    mode: int


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_relative(value: str, *, label: str) -> PurePosixPath:
    raw_parts = value.split("/")
    if (
        "\\" in value
        or any(ord(character) < 32 for character in value)
        or any(":" in part for part in raw_parts)
        or any(part in {"", ".", ".."} for part in raw_parts)
    ):
        raise AnonymousArchiveError(f"Unsafe {label}: {value!r}")
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts:
        raise AnonymousArchiveError(f"Unsafe {label}: {value!r}")
    return path


def _validate_archive_destination(path: PurePosixPath) -> None:
    if (
        path.is_absolute()
        or not path.parts
        or any(part in {"", ".", ".."} for part in path.parts)
        or "\\" in path.as_posix()
        or any(":" in part for part in path.parts)
    ):
        raise AnonymousArchiveError(f"Unsafe archive destination: {path}")
    lowered = tuple(part.lower() for part in path.parts)
    top_level = lowered[0]
    if top_level in FORBIDDEN_TOP_LEVEL or any(
        top_level.startswith(prefix) for prefix in FORBIDDEN_TOP_LEVEL_PREFIXES
    ):
        raise AnonymousArchiveError(f"Forbidden top-level archive path: {path}")
    if any(part in FORBIDDEN_PARTS for part in lowered):
        raise AnonymousArchiveError(f"Forbidden archive path component: {path}")
    filename = lowered[-1]
    if filename in FORBIDDEN_FILENAMES or "artifact_manifest" in filename:
        raise AnonymousArchiveError(f"Forbidden archive inventory file: {path}")
    suffix = path.suffix.lower()
    if suffix in FORBIDDEN_SUFFIXES:
        raise AnonymousArchiveError(f"Forbidden archive file type: {path}")
    if suffix not in TEXT_SUFFIXES:
        raise AnonymousArchiveError(f"Unsupported archive file type: {path}")


def load_allowlist(repository_root: Path, allowlist_path: Path) -> list[AllowedFile]:
    root = repository_root.resolve()
    entries: list[AllowedFile] = []
    destinations: set[PurePosixPath] = set()

    for line_number, raw_line in enumerate(
        allowlist_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        fields = [field.strip() for field in line.split(" -> ", maxsplit=1)]
        source_relative = _safe_relative(fields[0], label="allowlist source")
        archive_relative = _safe_relative(
            fields[1] if len(fields) == 2 else fields[0],
            label="archive destination",
        )
        _validate_archive_destination(archive_relative)
        if archive_relative in destinations:
            raise AnonymousArchiveError(
                f"Duplicate archive destination at allowlist line {line_number}: {archive_relative}"
            )

        source_path = (root / Path(*source_relative.parts)).resolve()
        try:
            source_path.relative_to(root)
        except ValueError as exc:
            raise AnonymousArchiveError(
                f"Allowlist source escapes repository at line {line_number}: {source_relative}"
            ) from exc
        if not source_path.is_file() or source_path.is_symlink():
            raise AnonymousArchiveError(
                f"Allowlist source is not a regular file at line {line_number}: {source_relative}"
            )

        mode = 0o755 if os.access(source_path, os.X_OK) else 0o644
        entries.append(
            AllowedFile(
                source_relative=source_relative,
                archive_relative=archive_relative,
                source_path=source_path,
                data=source_path.read_bytes(),
                mode=mode,
            )
        )
        destinations.add(archive_relative)

    if not entries:
        raise AnonymousArchiveError("The anonymous supplement allowlist is empty")
    return sorted(entries, key=lambda item: item.archive_relative.as_posix())


def runtime_deny_tokens(repository_root: Path) -> list[tuple[str, str]]:
    values: list[tuple[str, str]] = []
    for label, value in (
        ("build username", getpass.getuser()),
        ("build hostname", socket.gethostname()),
    ):
        normalized = value.strip()
        if normalized and normalized.casefold() not in COMMON_RUNTIME_IDENTITIES:
            values.append((label, normalized))

    git_commands = (
        ("Git origin URL", ["git", "remote", "get-url", "origin"]),
        ("Git configured author name", ["git", "config", "--get", "user.name"]),
        ("Git configured author email", ["git", "config", "--get", "user.email"]),
        (
            "Git history author or committer identity",
            ["git", "log", "--format=%an%n%ae%n%cn%n%ce", "-n", "200"],
        ),
    )
    for label, command in git_commands:
        completed = subprocess.run(
            command,
            cwd=repository_root,
            text=True,
            capture_output=True,
            check=False,
        )
        if completed.returncode != 0:
            continue
        for line in completed.stdout.splitlines():
            normalized = line.strip()
            if normalized and normalized.casefold() not in COMMON_RUNTIME_IDENTITIES:
                values.append((label, normalized))

    unique: list[tuple[str, str]] = []
    seen: set[str] = set()
    for label, value in values:
        folded = value.casefold()
        if folded not in seen:
            seen.add(folded)
            unique.append((label, value))
    return unique


def _token_pattern(token: str) -> re.Pattern[str]:
    escaped = re.escape(token)
    if token.isalnum():
        return re.compile(rf"(?i)(?<![A-Za-z0-9]){escaped}(?![A-Za-z0-9])")
    return re.compile(escaped, flags=re.IGNORECASE)


def scan_entries(
    entries: list[AllowedFile],
    deny_tokens: list[tuple[str, str]],
) -> None:
    violations: list[str] = []
    token_patterns = [(label, _token_pattern(token)) for label, token in deny_tokens if token]

    for entry in entries:
        path_text = entry.archive_relative.as_posix()
        for label, pattern in token_patterns:
            if pattern.search(path_text):
                violations.append(f"{path_text}: archive path contains {label}")

        if entry.archive_relative.suffix.lower() not in TEXT_SUFFIXES:
            continue
        try:
            text = entry.data.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise AnonymousArchiveError(
                f"Allowlisted text file is not UTF-8: {entry.archive_relative}"
            ) from exc

        for label, pattern in STATIC_CONTENT_PATTERNS:
            if pattern.search(text):
                violations.append(f"{path_text}: contains {label}")
        for label, pattern in token_patterns:
            if pattern.search(text):
                violations.append(f"{path_text}: contains {label}")

    if violations:
        formatted = "\n".join(f"  - {item}" for item in sorted(set(violations)))
        raise AnonymousArchiveError(f"Anonymous supplement identity scan failed:\n{formatted}")


def build_manifest(entries: list[AllowedFile]) -> bytes:
    payload = {
        "schema_version": 1,
        "artifact": "anonymous source supplement",
        "contains_git_history": False,
        "contains_models_or_data": False,
        "files": [
            {
                "path": entry.archive_relative.as_posix(),
                "bytes": len(entry.data),
                "sha256": sha256_bytes(entry.data),
                "mode": f"{entry.mode:04o}",
            }
            for entry in entries
        ],
    }
    return (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()


def _write_zip_member(archive: zipfile.ZipFile, name: str, data: bytes, mode: int) -> None:
    info = zipfile.ZipInfo(name, date_time=FIXED_ZIP_TIME)
    info.create_system = 3
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = (0o100000 | mode) << 16
    archive.writestr(info, data, compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)


def build_archive(
    *,
    repository_root: Path,
    allowlist_path: Path,
    output_path: Path,
    deny_tokens: list[tuple[str, str]],
) -> dict:
    entries = load_allowlist(repository_root, allowlist_path)
    scan_entries(entries, deny_tokens)
    manifest_data = build_manifest(entries)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(f"{output_path.suffix}.tmp")
    members = [
        (ARCHIVE_ROOT / entry.archive_relative, entry.data, entry.mode)
        for entry in entries
    ]
    members.append((ARCHIVE_ROOT / ARCHIVE_MANIFEST, manifest_data, 0o644))

    with zipfile.ZipFile(temporary, mode="w") as archive:
        for path, data, mode in sorted(members, key=lambda item: item[0].as_posix()):
            _write_zip_member(archive, path.as_posix(), data, mode)
    os.replace(temporary, output_path)
    return verify_archive(output_path, deny_tokens=deny_tokens)


def verify_archive(
    archive_path: Path,
    *,
    deny_tokens: list[tuple[str, str]],
) -> dict:
    with zipfile.ZipFile(archive_path, mode="r") as archive:
        infos = archive.infolist()
        names = [info.filename for info in infos]
        if names != sorted(names) or len(names) != len(set(names)):
            raise AnonymousArchiveError("Archive members must be unique and sorted")
        if any(info.is_dir() or info.date_time != FIXED_ZIP_TIME for info in infos):
            raise AnonymousArchiveError("Archive contains a directory entry or non-deterministic timestamp")
        if archive.comment or any(info.comment or info.extra for info in infos):
            raise AnonymousArchiveError("Archive contains an unexpected ZIP comment or extra metadata")

        root_prefix = f"{ARCHIVE_ROOT.as_posix()}/"
        if any(not name.startswith(root_prefix) for name in names):
            raise AnonymousArchiveError("Archive member is outside the anonymous package root")

        manifest_name = (ARCHIVE_ROOT / ARCHIVE_MANIFEST).as_posix()
        if manifest_name not in names:
            raise AnonymousArchiveError("Archive manifest is missing")
        manifest = json.loads(archive.read(manifest_name))
        expected = {
            (ARCHIVE_ROOT / item["path"]).as_posix(): item
            for item in manifest["files"]
        }
        actual_sources = set(names) - {manifest_name}
        if actual_sources != set(expected):
            raise AnonymousArchiveError("Archive members do not match ANONYMOUS_MANIFEST.json")

        scan_items = []
        for name in names:
            data = archive.read(name)
            if name != manifest_name:
                record = expected[name]
                if len(data) != record["bytes"] or sha256_bytes(data) != record["sha256"]:
                    raise AnonymousArchiveError(f"Archive member hash mismatch: {name}")
            relative = _safe_relative(
                name.removeprefix(root_prefix),
                label="archive member",
            )
            if (ARCHIVE_ROOT / relative).as_posix() != name:
                raise AnonymousArchiveError(f"Non-canonical archive member path: {name}")
            if name != manifest_name:
                _validate_archive_destination(relative)
            scan_items.append(
                AllowedFile(relative, relative, archive_path, data, 0o644)
            )
        scan_entries(scan_items, deny_tokens)

    return {
        "archive": str(archive_path),
        "files": len(expected),
        "bytes": archive_path.stat().st_size,
        "sha256": sha256_file(archive_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build or verify the deterministic anonymous source supplement."
    )
    parser.add_argument("--allowlist", type=Path, default=DEFAULT_ALLOWLIST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--verify-only",
        type=Path,
        help="Verify an existing archive instead of rebuilding it.",
    )
    parser.add_argument(
        "--deny-token",
        action="append",
        default=[],
        help="Additional identity token that must not appear in archive paths or text.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    deny_tokens = runtime_deny_tokens(REPOSITORY_ROOT)
    deny_tokens.extend(
        (f"explicit deny token {index}", token)
        for index, token in enumerate(args.deny_token, start=1)
    )
    if args.verify_only:
        result = verify_archive(args.verify_only.resolve(), deny_tokens=deny_tokens)
    else:
        result = build_archive(
            repository_root=REPOSITORY_ROOT,
            allowlist_path=args.allowlist.resolve(),
            output_path=args.output.resolve(),
            deny_tokens=deny_tokens,
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
