import logging

import torch

from SpecEmbedding.config import config
from SpecEmbedding.models import SiameseModel
from SpecEmbedding.models_align import GINEEncoder, SpecMolAlignModel


def create_align_model(
    mol_norm_type: str,
    mol_norm_eps: float,
    spec_encoder: SiameseModel | None = None,
) -> SpecMolAlignModel:
    if spec_encoder is None:
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
    return SpecMolAlignModel(
        spec_encoder=spec_encoder,
        mol_encoder=mol_encoder,
        spec_dim=config.model.spec_encoder.dim_target,
        hidden_dim=config.model.align.final_dim,
        final_dim=config.model.align.final_dim,
        dropout_rate=config.model.align.dropout_rate,
        tau=config.model.align.tau,
    )


def load_align_model(
    checkpoint: str,
    device: torch.device,
    mol_norm_type: str,
    mol_norm_eps: float,
) -> SpecMolAlignModel:
    model = create_align_model(mol_norm_type=mol_norm_type, mol_norm_eps=mol_norm_eps)
    state_dict = torch.load(checkpoint, map_location=device, weights_only=True)
    if "logit_scale" not in state_dict:
        logging.warning(
            "Checkpoint has no learnable logit_scale; initializing it from config.model.align.tau for compatibility."
        )
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


def get_dtype_size(dtype: torch.dtype) -> int:
    return torch.empty((), dtype=dtype).element_size()


def resolve_candidate_chunk_size(
    requested_chunk_size: int,
    memory_fraction: float,
    device: torch.device,
    embedding_dim: int,
    compute_dtype: torch.dtype,
    max_chunk_size: int,
) -> int:
    if requested_chunk_size > 0:
        return min(requested_chunk_size, max_chunk_size)

    if device.type != "cuda":
        return max_chunk_size

    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    dtype_size = get_dtype_size(compute_dtype)
    bytes_per_candidate = embedding_dim * dtype_size + dtype_size
    chunk_size = int(free_bytes * memory_fraction / bytes_per_candidate)
    chunk_size = max(1, min(chunk_size, max_chunk_size))

    logging.info(
        "Auto candidate chunk size: %s "
        "(free_cuda=%.2f GiB, total_cuda=%.2f GiB, memory_fraction=%.2f)",
        chunk_size,
        free_bytes / 1024**3,
        total_bytes / 1024**3,
        memory_fraction,
    )
    return chunk_size
