import logging
import pickle
from pathlib import Path

from src.data import GNPSProvider, MassBankProvider, MassSpecGymProvider, MoNAProvider, NPLIB1Provider


def get_provider(dataset_type: str, data_path: str | None = None):
    if dataset_type == "massspecgym":
        return MassSpecGymProvider(data_dir=data_path)
    if dataset_type == "massbank":
        return MassBankProvider(data_dir=data_path)
    if dataset_type == "nplib1":
        return NPLIB1Provider(data_dir=data_path)
    if dataset_type == "gnps":
        return GNPSProvider(data_dir=data_path)
    if dataset_type == "mona":
        return MoNAProvider(data_dir=data_path)
    raise ValueError("--dataset_type is invalid")


def load_train_val_data(dataset_type: str, data_path: str | None = None):
    provider = get_provider(dataset_type, data_path)
    train_raw = provider.load_data(mode="train")
    val_raw = provider.load_data(mode="val")
    return train_raw, val_raw


def load_candidates(provider, candidate_type: str, candidate_path: str | None):
    if candidate_path is None:
        return provider.load_candidates(type=candidate_type), candidate_type

    path = Path(candidate_path)
    if not path.exists():
        raise FileNotFoundError(f"Candidate file not found: {path}")

    logging.info("Loading candidates from custom file %s ...", path)
    with open(path, "rb") as f:
        candidates = pickle.load(f)

    if not isinstance(candidates, dict):
        raise TypeError(f"Candidate file must contain dict[str, list[str]], got {type(candidates).__name__}")

    logging.info("Loaded %s candidate sets from custom file.", len(candidates))
    return candidates, path.stem
