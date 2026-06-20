import logging
from collections.abc import Mapping
from pathlib import Path

import torch
import torch.nn.functional as F

from SpecEmbedding.models_rerank import CandidateReranker, PointwiseReranker


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
