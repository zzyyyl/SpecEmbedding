import json
import logging
from pathlib import Path

import torch
from rdkit import rdBase

from SpecEmbedding.config import config
from SpecEmbedding.models import SiameseModel
from SpecEmbedding.models_align import GINEEncoder, SpecMolAlignModel
from SpecEmbedding.models_precursor_delta import build_spectrum_encoder
from SpecEmbedding.utils.fulltrain import sha256_file


def create_align_model(
    mol_norm_type: str,
    mol_norm_eps: float,
    spec_encoder: SiameseModel | None = None,
) -> SpecMolAlignModel:
    if spec_encoder is None:
        spec_encoder = build_spectrum_encoder(config.model.spec_encoder.to_dict())

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
    selection_path = Path(checkpoint).parent / "alignment_selection.json"
    if selection_path.exists():
        selection = json.loads(selection_path.read_text())
        # Missing policy is the explicitly identified historical format, never v1.5.
        policy = selection.get("graph_policy", "legacy_raw")
        if policy != config.model.mol_encoder.graph_policy:
            raise ValueError("Alignment checkpoint graph policy differs from the active configuration")
        if policy == "rdkit_sanitized":
            if selection["checkpoint_sha256"] != sha256_file(checkpoint) or selection["model_config"] != config.model.to_dict():
                raise ValueError("Sanitized alignment checkpoint/config fingerprint mismatch")
            if selection["rdkit_version"] != rdBase.rdkitVersion:
                raise ValueError("Sanitized alignment RDKit version differs from training")
    elif config.model.mol_encoder.graph_policy != "legacy_raw":
        raise ValueError("Sanitized alignment loading requires alignment_selection.json provenance")
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
