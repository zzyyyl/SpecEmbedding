import pickle
from pathlib import Path

import torch

from SpecEmbedding.config import config
from SpecEmbedding.models import SiameseModel
from SpecEmbedding.models_align import GINEEncoder, SpecMolAlignModel
from src.data import GNPSProvider, MassBankProvider, MassSpecGymProvider, MoNAProvider, NPLIB1Provider


def get_provider(dataset_type: str, data_path: str):
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


def load_candidates(provider, candidate_type: str, candidate_path: str | None):
    if candidate_path is None:
        return provider.load_candidates(type=candidate_type), candidate_type

    path = Path(candidate_path)
    if not path.exists():
        raise FileNotFoundError(f"Candidate file not found: {path}")

    with open(path, "rb") as f:
        candidates = pickle.load(f)
    if not isinstance(candidates, dict):
        raise TypeError(f"Candidate file must contain dict[str, list[str]], got {type(candidates).__name__}")
    return candidates, path.stem


def build_align_model(
    checkpoint: str,
    device: torch.device,
    mol_norm_type: str = "layernorm",
    mol_norm_eps: float = 1e-5,
) -> SpecMolAlignModel:
    spec_encoder = SiameseModel(
        embedding_dim=config.model.spec_encoder.embedding_dim,
        n_head=config.model.spec_encoder.n_head,
        n_layer=config.model.spec_encoder.n_layer,
        dim_feedward=config.model.spec_encoder.dim_feedward,
        dim_target=config.model.spec_encoder.dim_target,
        feedward_activation=config.model.spec_encoder.feedward_activation,
    )
    mol_encoder = GINEEncoder(
        emb_dim=config.model.mol_encoder.emb_dim,
        n_layers=config.model.mol_encoder.n_layers,
        dropout_rate=config.model.mol_encoder.dropout_rate,
        size_feature_dim=config.model.mol_encoder.size_feature_dim,
        norm_type=mol_norm_type,
        norm_eps=mol_norm_eps,
    )
    model = SpecMolAlignModel(
        spec_encoder=spec_encoder,
        mol_encoder=mol_encoder,
        spec_dim=config.model.spec_encoder.dim_target,
        hidden_dim=config.model.align.final_dim,
        final_dim=config.model.align.final_dim,
        dropout_rate=config.model.align.dropout_rate,
        tau=config.model.align.tau,
    )

    state_dict = torch.load(checkpoint, map_location=device, weights_only=True)
    if "logit_scale" not in state_dict:
        state_dict["logit_scale"] = model.logit_scale.detach().clone()
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device)
    model.eval()
    return model


def resolve_storage_dtype(dtype_name: str) -> torch.dtype:
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype_name}")
