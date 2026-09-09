import os
from pathlib import Path
from typing import Any

import yaml

from SpecEmbedding.utils.storage import expanded_path, external_storage_root, storage_path

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "params.yaml"
PATH_FIELDS = (
    ("general", "save_dir"),
    ("data", "data_path"),
    ("data", "cache_path"),
    ("data", "raw_path"),
    ("data", "candidates_path"),
    ("data", "cid_smiles_path"),
    ("data", "pubchem_cache_path"),
    ("rerank", "prepare", "checkpoint"),
    ("rerank", "prepare", "data_path"),
    ("rerank", "prepare", "save_path"),
    ("rerank", "prepare", "candidate_path"),
    ("rerank", "train", "train_cache"),
    ("rerank", "train", "val_cache"),
    ("rerank", "train", "save_dir"),
    ("rerank", "eval", "cache"),
    ("rerank", "eval", "checkpoint"),
    ("rerank", "eval", "save_dir"),
)


class ConfigObject:
    def __init__(self, data: dict[str, Any]):
        for key, value in data.items():
            if isinstance(value, dict):
                setattr(self, key, ConfigObject(value))
            else:
                setattr(self, key, value)

    def to_dict(self) -> dict[str, Any]:
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, ConfigObject):
                result[key] = value.to_dict()
            else:
                result[key] = value
        return result

    def __getitem__(self, key):
        return getattr(self, key)

def _resolve_path_value(value: str, config_dir: Path) -> str:
    path = expanded_path(value)
    if not path.is_absolute():
        path = config_dir / path
    return str(path.resolve())


def _resolve_known_paths(data: dict[str, Any], config_dir: Path) -> None:
    for field_path in PATH_FIELDS:
        parent = data
        for key in field_path[:-1]:
            parent = parent.get(key)
            if not isinstance(parent, dict):
                break
        else:
            field = field_path[-1]
            value = parent.get(field)
            if isinstance(value, str) and value:
                parent[field] = _resolve_path_value(value, config_dir)

    pretrained_models = data.get("paths", {}).get("pretrained_models", {})
    if isinstance(pretrained_models, dict):
        for key, value in pretrained_models.items():
            if isinstance(value, str) and value:
                pretrained_models[key] = _resolve_path_value(value, config_dir)


def load_config(config_path: str | Path | None = None) -> ConfigObject:
    configured_path = config_path or os.environ.get("SPECEMBEDDING_CONFIG")
    path = Path(configured_path).expanduser() if configured_path else DEFAULT_CONFIG_PATH
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Config file not found at {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a mapping: {path}")
    storage = data.get("storage", {})
    if not isinstance(storage, dict):
        raise ValueError("storage must be a mapping")
    root_value = os.environ.get("SPECEMBEDDING_STORAGE_ROOT", storage.get("root"))
    root = external_storage_root(root_value) if root_value is not None else None
    if root is not None:
        data["storage"] = {**storage, "root": str(root)}
    _resolve_known_paths(data, root or path.parent)
    if root is not None:
        for keys in (("general", "save_dir"), ("rerank", "prepare", "save_path"),
                     ("rerank", "train", "save_dir"), ("rerank", "eval", "save_dir")):
            parent = data
            for key in keys[:-1]:
                parent = parent.get(key, {})
            if parent.get(keys[-1]):
                storage_path(parent[keys[-1]], root)
        for key, value in data.get("data", {}).items():
            if key.endswith("_path") and isinstance(value, str) and value:
                storage_path(value, root)
    return ConfigObject(data)


# Global config instance. Missing or malformed configuration should fail loudly
# instead of leaving callers with an opaque ``NoneType`` error later.
config = load_config()
