"""External experiment storage and dependency caches; no import-time writes."""

import os
import re
from pathlib import Path

CACHE_DIRECTORIES = {
    "XDG_CACHE_HOME": "cache/xdg",
    "HF_HOME": "cache/huggingface",
    "HF_HUB_CACHE": "cache/huggingface/hub",
    "HUGGINGFACE_HUB_CACHE": "cache/huggingface/hub",
    "HF_DATASETS_CACHE": "cache/huggingface/datasets",
    "HF_ASSETS_CACHE": "cache/huggingface/assets",
    "HF_XET_CACHE": "cache/huggingface/xet",
    "TORCH_HOME": "cache/torch",
    "TORCH_EXTENSIONS_DIR": "cache/torch_extensions",
    "TORCHINDUCTOR_CACHE_DIR": "cache/torchinductor",
    "TRITON_CACHE_DIR": "cache/triton",
    "CUDA_CACHE_PATH": "cache/cuda",
    "NUMBA_CACHE_DIR": "cache/numba",
    "MPLCONFIGDIR": "cache/matplotlib",
    "PYTHONPYCACHEPREFIX": "cache/python",
    "TMPDIR": "tmp",
}


def expanded_path(value: str | Path) -> Path:
    expanded = os.path.expandvars(str(value))
    if re.search(r"\$(?:[A-Za-z_]\w*|\{[^}]+\})", expanded):
        raise ValueError(f"Unresolved environment variable in path: {value}")
    return Path(expanded).expanduser()


def external_storage_root(value: str | Path) -> Path:
    path = expanded_path(value)
    if not path.is_absolute():
        raise ValueError("Storage root must be an absolute path")
    root = path.resolve()
    if root == Path(root.anchor) or root.is_relative_to(Path('/home')) or root.is_relative_to(Path.home().resolve()):
        raise ValueError("Storage root must be outside the home filesystem tree")
    return root


def storage_path(value: str | Path, root: Path) -> Path:
    path = expanded_path(value)
    resolved = (path if path.is_absolute() else root / path).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"Storage path escapes configured root: {value}")
    return resolved


def storage_environment(root: Path) -> dict[str, str]:
    root = external_storage_root(root)
    return {
        "SPECEMBEDDING_STORAGE_ROOT": str(root),
        "SPECEMBEDDING_DATA_ROOT": str(root),
        **{name: str(storage_path(relative, root)) for name, relative in CACHE_DIRECTORIES.items()},
    }


def storage_receipt(output_root: Path) -> dict | None:
    """Pin the launcher's storage contract in a formal queue preflight."""
    configured = os.environ.get("SPECEMBEDDING_STORAGE_ROOT")
    if configured is None:
        return None  # Historical configurations retain their original semantics.
    root = external_storage_root(configured)
    environment = storage_environment(root)
    if any(os.environ.get(key) != value for key, value in environment.items()):
        raise ValueError("Storage environment mismatch; launch through run_with_storage.py")
    return {"root": str(root), "output_root": str(storage_path(output_root, root)), "environment": environment}
