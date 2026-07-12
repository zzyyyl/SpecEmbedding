import logging
import re
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


def _config_value_default(source, key: str, default):
    try:
        return _config_value(source, key)
    except KeyError:
        return default


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

    legacy_base_score = bool(_config_value_default(model_config, "use_base_score", True))
    return model_cls(
        embedding_dim=int(embedding_dim),
        hidden_dim=int(_config_value(model_config, "hidden_dim")),
        rank_emb_dim=int(_config_value(model_config, "rank_emb_dim")),
        max_rank=int(_config_value(model_config, "max_rank")),
        n_layers=int(_config_value(model_config, "n_layers")),
        n_heads=int(_config_value(model_config, "n_heads")),
        dropout=float(_config_value(model_config, "dropout")),
        alpha_init=float(_config_value(model_config, "alpha_init")),
        use_base_score_feature=bool(
            _config_value_default(model_config, "use_base_score_feature", legacy_base_score)
        ),
        use_residual_score=bool(
            _config_value_default(model_config, "use_residual_score", legacy_base_score)
        ),
        use_rank_embedding=bool(_config_value_default(model_config, "use_rank_embedding", True)),
        use_product_feature=bool(_config_value_default(model_config, "use_product_feature", True)),
        use_abs_diff_feature=bool(_config_value_default(model_config, "use_abs_diff_feature", True)),
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
        "use_base_score_feature": args.use_base_score_feature,
        "use_residual_score": args.use_residual_score,
        "use_rank_embedding": args.use_rank_embedding,
        "use_product_feature": args.use_product_feature,
        "use_abs_diff_feature": args.use_abs_diff_feature,
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


def parse_rerank_eval_metrics(eval_log: str | Path) -> dict[str, float | int]:
    eval_log = Path(eval_log)
    metrics: dict[str, float | int] = {}
    section = ""
    for line in eval_log.read_text(encoding="utf-8", errors="replace").splitlines():
        total_match = re.search(r"Total queries:\s+(\d+)", line)
        if total_match:
            metrics["total_queries"] = int(total_match.group(1))

        excluded_match = re.search(r"Excluded\s+(\d+)\s+cache queries", line)
        if excluded_match:
            metrics["excluded_queries"] = int(excluded_match.group(1))

        if "Pre-retrieval upper bound:" in line:
            match = re.search(r"Pre-retrieval upper bound:\s+([0-9.]+)%", line)
            if match:
                metrics["upper_bound_pct"] = float(match.group(1))
        elif "BASE RESULTS" in line:
            section = "base"
        elif "RERANK RESULTS" in line:
            section = "rerank"
        elif section:
            top_match = re.search(r"Top-(\d+)\s+Accuracy\s+:\s+([0-9.]+)%", line)
            if top_match:
                metrics[f"{section}_top{top_match.group(1)}_pct"] = float(top_match.group(2))
            mrr_match = re.search(r"MRR\s+:\s+([0-9.]+)", line)
            if mrr_match:
                metrics[f"{section}_mrr_raw"] = float(mrr_match.group(1))
        if "Base   MCES@1" in line:
            metrics["base_mces"] = float(line.rsplit(":", 1)[1].strip())
        elif "Rerank MCES@1" in line:
            metrics["rerank_mces"] = float(line.rsplit(":", 1)[1].strip())
    return metrics
