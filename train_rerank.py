import argparse
import logging
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from SpecEmbedding.config import config
from SpecEmbedding.data.datasets_rerank import RerankCacheDataset, rerank_collate_fn
from SpecEmbedding.models_rerank import CandidateReranker, PointwiseReranker
from SpecEmbedding.trainer.trainer import set_seed
from SpecEmbedding.utils.runtime import configure_runtime_cache, resolve_device, setup_logging, startup_logging

configure_runtime_cache()


def masked_scores(scores, candidate_mask):
    return scores.masked_fill(~candidate_mask, torch.finfo(scores.dtype).min)


def listwise_cross_entropy(scores, labels, candidate_mask):
    labeled_mask = labels >= 0
    if not labeled_mask.any():
        return None
    scores = masked_scores(scores[labeled_mask], candidate_mask[labeled_mask])
    labels = labels[labeled_mask]
    return F.cross_entropy(scores, labels)


def pairwise_ranking_loss(scores, labels, candidate_mask, margin):
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


def compute_loss(scores, labels, candidate_mask, lambda_pair, margin):
    ce_loss = listwise_cross_entropy(scores, labels, candidate_mask)
    if ce_loss is None:
        return None
    if lambda_pair <= 0:
        return ce_loss
    pair_loss = pairwise_ranking_loss(scores, labels, candidate_mask, margin)
    if pair_loss is None:
        return ce_loss
    return ce_loss + lambda_pair * pair_loss


@torch.no_grad()
def evaluate(model, loader, device, top_k):
    model.eval()
    total = 0
    positives = 0
    hits = {k: 0 for k in top_k}
    mrr_sum = 0.0
    loss_sum = 0.0
    loss_count = 0

    for batch in tqdm(loader, desc="Validation", ascii=True):
        spec_emb = batch["spec_emb"].to(device)
        candidate_embs = batch["candidate_embs"].to(device)
        base_scores = batch["base_scores"].to(device)
        base_ranks = batch["base_ranks"].to(device)
        candidate_mask = batch["candidate_mask"].to(device)
        labels = batch["labels"].to(device)

        scores = model(spec_emb, candidate_embs, base_scores, base_ranks, candidate_mask)
        loss = listwise_cross_entropy(scores, labels, candidate_mask)
        if loss is not None:
            loss_sum += loss.item()
            loss_count += 1

        scores = masked_scores(scores, candidate_mask)
        total += labels.numel()
        labeled_rows = (labels >= 0).nonzero(as_tuple=False).flatten()
        positives += labeled_rows.numel()

        for row in labeled_rows.tolist():
            true_score = scores[row, labels[row]]
            rank = int((scores[row] > true_score).sum().item()) + 1
            for k in top_k:
                if rank <= k:
                    hits[k] += 1
            mrr_sum += 1.0 / rank

    metrics = {f"top{k}": hits[k] / total for k in top_k}
    metrics["mrr"] = mrr_sum / total
    metrics["upper_bound"] = positives / total
    metrics["loss"] = loss_sum / max(loss_count, 1)
    return metrics


def build_model(args, embedding_dim):
    model_cls = PointwiseReranker if args.model_type == "pointwise" else CandidateReranker
    return model_cls(
        embedding_dim=embedding_dim,
        hidden_dim=args.hidden_dim,
        rank_emb_dim=args.rank_emb_dim,
        max_rank=args.max_rank,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        dropout=args.dropout,
        alpha_init=args.alpha_init,
    )


def main():
    parser = argparse.ArgumentParser(description="Train a lightweight molecule reranker on offline rerank caches.")
    parser.add_argument("--train_cache", type=str, required=True)
    parser.add_argument("--val_cache", type=str, required=True)
    parser.add_argument("--save_dir", type=str, default="./checkpoints_rerank")
    parser.add_argument("--device", type=str, default=config.general.device)
    parser.add_argument("--model_type", type=str, choices=["transformer", "pointwise"], default="transformer")
    parser.add_argument("--train_k", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--rank_emb_dim", type=int, default=32)
    parser.add_argument("--max_rank", type=int, default=512)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--alpha_init", type=float, default=1.0)
    parser.add_argument("--lambda_pair", type=float, default=0.2)
    parser.add_argument("--margin", type=float, default=0.1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--metric_for_best", type=str, choices=["mrr", "top1", "top5"], default="mrr")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--top_k", type=int, nargs="+", default=config.eval.top_k)
    args = parser.parse_args()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(save_dir / "train_rerank.log")
    startup_logging(args, "Train SpecEmbedding reranker")
    set_seed(config.general.seed)
    device = resolve_device(args.device)

    train_dataset = RerankCacheDataset(
        args.train_cache,
        max_candidates=args.train_k,
        shuffle_candidates=True,
        require_label=True,
    )
    val_dataset = RerankCacheDataset(
        args.val_cache,
        max_candidates=None,
        shuffle_candidates=False,
        require_label=False,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=rerank_collate_fn,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=rerank_collate_fn,
        num_workers=args.num_workers,
    )

    model = build_model(args, embedding_dim=train_dataset.embedding_dim).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    model_config = {
        "model_type": args.model_type,
        "embedding_dim": train_dataset.embedding_dim,
        "hidden_dim": args.hidden_dim,
        "rank_emb_dim": args.rank_emb_dim,
        "max_rank": args.max_rank,
        "n_layers": args.n_layers,
        "n_heads": args.n_heads,
        "dropout": args.dropout,
        "alpha_init": args.alpha_init,
    }

    best_metric = -float("inf")
    best_epoch = 0
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        num_steps = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch} Training", ascii=True)
        for batch in pbar:
            spec_emb = batch["spec_emb"].to(device)
            candidate_embs = batch["candidate_embs"].to(device)
            base_scores = batch["base_scores"].to(device)
            base_ranks = batch["base_ranks"].to(device)
            candidate_mask = batch["candidate_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            scores = model(spec_emb, candidate_embs, base_scores, base_ranks, candidate_mask)
            loss = compute_loss(
                scores=scores,
                labels=labels,
                candidate_mask=candidate_mask,
                lambda_pair=args.lambda_pair,
                margin=args.margin,
            )
            if loss is None:
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
            optimizer.step()

            total_loss += loss.item()
            num_steps += 1
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        train_loss = total_loss / max(num_steps, 1)
        val_metrics = evaluate(model, val_loader, device, sorted(set(args.top_k)))
        current_metric = val_metrics[args.metric_for_best]
        logging.info(
            "Epoch %s: train_loss=%.4f val_loss=%.4f val_mrr=%.4f val_top1=%.4f val_upper=%.4f",
            epoch,
            train_loss,
            val_metrics["loss"],
            val_metrics["mrr"],
            val_metrics.get("top1", 0.0),
            val_metrics["upper_bound"],
        )

        if current_metric > best_metric:
            best_metric = current_metric
            best_epoch = epoch
            patience_counter = 0
            save_path = save_dir / "best_reranker.pth"
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "model_config": model_config,
                    "best_metric": best_metric,
                    "best_epoch": best_epoch,
                    "val_metrics": val_metrics,
                },
                save_path,
            )
            logging.info("Saved best reranker to %s", save_path)
        else:
            patience_counter += 1
            logging.info("EarlyStopping counter: %s out of %s", patience_counter, args.patience)
            if patience_counter >= args.patience:
                break

    torch.save({"state_dict": model.state_dict(), "model_config": model_config}, save_dir / "last_reranker.pth")
    logging.info("Training finished. Best epoch=%s best_%s=%.4f", best_epoch, args.metric_for_best, best_metric)


if __name__ == "__main__":
    main()
