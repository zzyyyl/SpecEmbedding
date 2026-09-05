import pickle
from pathlib import Path

from SpecEmbedding.config import config

from .base import DataProvider


class NPLIB1Provider(DataProvider):
    def __init__(self, data_dir=None):
        if data_dir is None:
            data_dir = config.data.data_path
        super(NPLIB1Provider, self).__init__("NPLIB1", data_dir)

    def load_data(self, mode):
        if mode not in {"train", "val", "test"}:
            raise ValueError("NPLIB1 mode must be one of: train, val, test")
        file_path = self.data_dir / f"{mode}.pkl"
        if not file_path.is_file():
            raise FileNotFoundError(f"NPLIB1 {mode} data not found: {file_path}")
        with file_path.open("rb") as handle:
            data = pickle.load(handle)
        if not isinstance(data, (list, tuple)):
            raise TypeError(
                f"NPLIB1 {mode} data must be a sequence, got {type(data).__name__}"
            )
        return data

    def load_candidates(self, type):
        if type not in {"formula", "supplied"}:
            raise ValueError(
                "NPLIB1 provides the 'formula' and 'supplied' candidate protocols; "
                "use a custom --candidate_path for any other protocol"
            )
        file_path = Path(self.data_dir) / f"candidates_{type}.pkl"
        if not file_path.is_file():
            raise FileNotFoundError(f"NPLIB1 {type} candidates not found: {file_path}")
        with file_path.open("rb") as handle:
            candidates = pickle.load(handle)
        if not isinstance(candidates, dict):
            raise TypeError(
                f"NPLIB1 {type} candidates must be dict[str, list[str]], "
                f"got {candidates.__class__.__name__}"
            )
        return candidates
