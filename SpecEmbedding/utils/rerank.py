import logging
import pickle
from collections.abc import Mapping
from pathlib import Path

import torch
import torch.nn.functional as F

from SpecEmbedding.config import config
from SpecEmbedding.models import SiameseModel
from SpecEmbedding.models_align import GINEEncoder, SpecMolAlignModel
from SpecEmbedding.models_rerank import CandidateReranker, PointwiseReranker
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
    mol_norm_type: str,
    mol_norm_eps: float,
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


def _config_value(source, key: str):
    if isinstance(source, Mapping):
        if key in source:
            return source[key]
    elif hasattr(source, key):
        return getattr(source, key)

    raise KeyError(f"Missing reranker model config value: {key}")


def build_reranker(model_config, embedding_dim: int | None = None):
    if embedding_dim is None:
        embedding_dim = _config_value(model_config, "embedding_dim")
    if embedding_dim is None:
        raise ValueError("embedding_dim is required to build a reranker")

    model_type = _config_value(model_config, "model_type")
    if model_type == "pointwise":
        model_cls = PointwiseReranker
    elif model_type == "transformer":
        model_cls = CandidateReranker
    else:
        raise ValueError(f"Unsupported reranker model_type: {model_type}")

    return model_cls(
        embedding_dim=int(embedding_dim),
        hidden_dim=int(_config_value(model_config, "hidden_dim")),
        rank_emb_dim=int(_config_value(model_config, "rank_emb_dim")),
        max_rank=int(_config_value(model_config, "max_rank")),
        n_layers=int(_config_value(model_config, "n_layers")),
        n_heads=int(_config_value(model_config, "n_heads")),
        dropout=float(_config_value(model_config, "dropout")),
        alpha_init=float(_config_value(model_config, "alpha_init")),
    )


def reranker_model_config(args, embedding_dim: int) -> dict:
    return {
        "model_type": args.model_type,
        "embedding_dim": embedding_dim,
        "hidden_dim": args.hidden_dim,
        "rank_emb_dim": args.rank_emb_dim,
        "max_rank": args.max_rank,
        "n_layers": args.n_layers,
        "n_heads": args.n_heads,
        "dropout": args.dropout,
        "alpha_init": args.alpha_init,
    }


def load_reranker(checkpoint_path: str | Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = build_reranker(checkpoint["model_config"])
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model = model.to(device)
    model.eval()
    return model


def masked_scores(scores, candidate_mask):
    return scores.masked_fill(~candidate_mask, torch.finfo(scores.dtype).min)


def masked_argmax(scores, candidate_mask):
    return torch.argmax(masked_scores(scores, candidate_mask), dim=-1)


def listwise_cross_entropy(scores, labels, candidate_mask):
    labeled_mask = labels >= 0
    if not labeled_mask.any():
        return None
    scores = masked_scores(scores[labeled_mask], candidate_mask[labeled_mask])
    labels = labels[labeled_mask]
    return F.cross_entropy(scores, labels)


def pairwise_ranking_loss(scores, labels, candidate_mask, margin: float):
    labeled_mask = labels >= 0
    if not labeled_mask.any():
        return None

    scores = scores[labeled_mask]
    candidate_mask = candidate_mask[labeled_mask]
    labels = labels[labeled_mask]
    row_indices = torch.arange(scores.size(0), device=scores.device)
    pos_scores = scores[row_indices, labels].unsqueeze(1)
    neg_mask = candidate_mask.clone()
    neg_mask[row_indices, labels] = False
    losses = F.softplus(scores - pos_scores + margin)
    losses = losses.masked_select(neg_mask)
    if losses.numel() == 0:
        return None
    return losses.mean()


def compute_rerank_loss(scores, labels, candidate_mask, lambda_pair: float, margin: float):
    ce_loss = listwise_cross_entropy(scores, labels, candidate_mask)
    if ce_loss is None:
        return None
    if lambda_pair <= 0:
        return ce_loss
    pair_loss = pairwise_ranking_loss(scores, labels, candidate_mask, margin)
    if pair_loss is None:
        return ce_loss
    return ce_loss + lambda_pair * pair_loss


def rank_from_scores(scores, label: int, candidate_mask) -> int:
    true_score = scores[label]
    return int((scores.masked_select(candidate_mask) > true_score).sum().item()) + 1


def init_ranking_metrics(top_k):
    return {"total": 0, "hits": {k: 0 for k in top_k}, "mrr_sum": 0.0}


def update_ranking_metrics(metrics, rank: int | None, top_k):
    metrics["total"] += 1
    if rank is None:
        return
    for k in top_k:
        if rank <= k:
            metrics["hits"][k] += 1
    metrics["mrr_sum"] += 1.0 / rank


def summarize_ranking_metrics(metrics, top_k):
    total = max(metrics["total"], 1)
    summary = {f"top{k}": metrics["hits"][k] / total for k in top_k}
    summary["mrr"] = metrics["mrr_sum"] / total
    return summary


def log_ranking_summary(name: str, summary: dict, top_k):
    logging.info("%s RESULTS", name)
    for k in top_k:
        logging.info("  Top-%-2s Accuracy : %.4f%%", k, summary[f"top{k}"] * 100)
    logging.info("  MRR              : %.4f", summary["mrr"])
