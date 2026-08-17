"""Portable repository and data-path helpers.

Path constants may be imported by data-preparation modules, so this module only
resolves paths.  It deliberately does not create directories.
"""

import os
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def repository_path(*parts: str) -> Path:
    """Return a path below the checked-out repository root."""
    return REPOSITORY_ROOT.joinpath(*parts)


def data_path(environment_variable: str, *parts: str) -> Path:
    """Resolve a dataset path from an explicit env var or the portable data root.

    ``SPECEMBEDDING_DATA_ROOT`` defaults to ``<repository>/data``.  A more
    specific variable such as ``SPECEMBEDDING_GNPS_DIR`` takes precedence.
    """
    configured = os.environ.get(environment_variable)
    if configured:
        return Path(configured).expanduser()

    data_root = Path(
        os.environ.get("SPECEMBEDDING_DATA_ROOT", repository_path("data"))
    ).expanduser()
    return data_root.joinpath(*parts)


LEGACY_DATA_ROOT = data_path("SPECEMBEDDING_LEGACY_DATA_ROOT", "legacy")
MSBERT_ROOT = Path(
    os.environ.get("SPECEMBEDDING_MSBERT_ROOT", LEGACY_DATA_ROOT / "MSBert")
).expanduser()
TSNE_CLUSTER_ROOT = data_path(
    "SPECEMBEDDING_TSNE_CLUSTER_DIR",
    "tsne_cluster_data",
)
