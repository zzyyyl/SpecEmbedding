from pathlib import Path
from typing import Any

import yaml


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

def load_config(config_path: str = "params.yaml") -> ConfigObject:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found at {config_path}")
    
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    
    return ConfigObject(data)

# Global config instance
try:
    config = load_config()
except Exception:
    config = None
