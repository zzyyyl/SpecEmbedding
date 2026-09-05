"""Download and verify the NPLIB1 bundle referenced by the JESTR repository."""

import argparse
import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

JESTR_COMMIT = "a5619c18a85a49171d60ead079684ae667cc0dd0"


@dataclass(frozen=True)
class SourceFile:
    filename: str
    url: str
    size_bytes: int
    checksum_kind: str
    checksum: str
    source_record: str


SOURCE_FILES = (
    SourceFile(
        filename="split.pkl",
        url=(
            "https://raw.githubusercontent.com/HassounLab/JESTR1/"
            f"{JESTR_COMMIT}/data/NPLIB1/split.pkl"
        ),
        size_bytes=216_240,
        checksum_kind="git_blob_sha1",
        checksum="ba259183fcc045857f8389cec4936dbe296ca22d",
        source_record=(
            "https://github.com/HassounLab/JESTR1/tree/"
            f"{JESTR_COMMIT}/data/NPLIB1"
        ),
    ),
    SourceFile(
        filename="data_dict.pkl",
        url="https://zenodo.org/api/records/11237532/files/data_dict.pkl/content",
        size_bytes=117_212_575,
        checksum_kind="md5",
        checksum="2b1d3c6f8c90b385a5c7ee50fa47dca3",
        source_record="https://doi.org/10.5281/zenodo.11237532",
    ),
    SourceFile(
        filename="mol_dict.pkl",
        url="https://zenodo.org/api/records/11237413/files/mol_dict.pkl/content",
        size_bytes=3_367_180_389,
        checksum_kind="md5",
        checksum="99090d074351468937e06e291f8be2e9",
        source_record="https://doi.org/10.5281/zenodo.11237413",
    ),
    SourceFile(
        filename="cand_dict_large.pkl",
        url="https://zenodo.org/api/records/11237561/files/cand_dict_large.pkl/content",
        size_bytes=49_745_029,
        checksum_kind="md5",
        checksum="3e9f0caefffb3e855da3c7d5ab579409",
        source_record="https://doi.org/10.5281/zenodo.11237561",
    ),
    SourceFile(
        filename="cand_dict_train_updated.pkl",
        url=(
            "https://zenodo.org/api/records/11237582/files/"
            "cand_dict_train_updated.pkl/content"
        ),
        size_bytes=888_629_969,
        checksum_kind="md5",
        checksum="c278b7e2966768de04aef525b66f10f4",
        source_record="https://doi.org/10.5281/zenodo.11237582",
    ),
)


def file_digest(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_blob_sha1(path: Path) -> str:
    digest = hashlib.sha1(usedforsecurity=False)
    digest.update(f"blob {path.stat().st_size}\0".encode("ascii"))
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_source_file(path: Path, source: SourceFile) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Downloaded NPLIB1 file not found: {path}")
    actual_size = path.stat().st_size
    if actual_size != source.size_bytes:
        raise ValueError(
            f"Unexpected size for {path}: expected {source.size_bytes}, got {actual_size}"
        )

    if source.checksum_kind == "git_blob_sha1":
        actual_source_checksum = git_blob_sha1(path)
    else:
        actual_source_checksum = file_digest(path, source.checksum_kind)
    if actual_source_checksum != source.checksum:
        raise ValueError(
            f"Checksum mismatch for {path}: expected {source.checksum_kind}:"
            f"{source.checksum}, got {actual_source_checksum}"
        )

    return {
        "filename": source.filename,
        "source_record": source.source_record,
        "download_url": source.url,
        "bytes": actual_size,
        "source_checksum": f"{source.checksum_kind}:{actual_source_checksum}",
        "sha256": file_digest(path, "sha256"),
    }


def download_source(source: SourceFile, output_dir: Path) -> dict:
    output_path = output_dir / source.filename
    partial_path = output_path.with_suffix(output_path.suffix + ".part")
    if output_path.is_file():
        return validate_source_file(output_path, source)

    command = [
        "curl",
        "--location",
        "--fail",
        "--show-error",
        "--retry",
        "5",
        "--continue-at",
        "-",
        "--output",
        str(partial_path),
        source.url,
    ]
    subprocess.run(command, check=True)
    partial_path.replace(output_path)
    return validate_source_file(output_path, source)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download the pinned JESTR NPLIB1 source bundle and verify it."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Destination directory, for example <storage-root>/raw/NPLIB1.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    records = [download_source(source, output_dir) for source in SOURCE_FILES]
    manifest = {
        "schema_version": 1,
        "dataset": "NPLIB1",
        "distribution": "JESTR NPLIB1 bundle",
        "jestr_repository": "https://github.com/HassounLab/JESTR1",
        "jestr_commit": JESTR_COMMIT,
        "license": {
            "dataset_records": "CC-BY-4.0",
            "jestr_repository": "MIT",
        },
        "files": records,
    }
    manifest_path = output_dir / "nplib1_acquisition_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Verified {len(records)} NPLIB1 files; manifest: {manifest_path}")


if __name__ == "__main__":
    main()
