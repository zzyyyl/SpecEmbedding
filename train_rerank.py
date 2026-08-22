import argparse
import logging
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from SpecEmbedding.config import config
from SpecEmbedding.data.datasets_rerank import (
    RerankCacheDataset,
    exclude_query_indices,
    rerank_collate_fn,
)
from SpecEmbedding.trainer.trainer import set_seed
from SpecEmbedding.utils.rerank import (
    build_reranker,
    compute_rerank_loss,
    init_ranking_metrics,
    listwise_cross_entropy,
    rank_from_scores,
    reranker_model_config,
    spectrum_dependency_loss,
    summarize_ranking_metrics,
    update_ranking_metrics,
)
from SpecEmbedding.utils.runtime import configure_runtime_cache, resolve_device, setup_logging, startup_logging


@torch.no_grad()
def evaluate(model, loader, device, top_k):
    model.eval()
    metrics = init_ranking_metrics(top_k)
    positives = 0
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

        for row in range(labels.numel()):
            label = int(labels[row].item())
            if label < 0:
                update_ranking_metrics(metrics, None, top_k)
                continue

            positives += 1
            rank = rank_from_scores(scores[row], label, candidate_mask[row])
            update_ranking_metrics(metrics, rank, top_k)

    summary = summarize_ranking_metrics(metrics, top_k)
    summary["upper_bound"] = positives / max(metrics["total"], 1)
    summary["loss"] = loss_sum / max(loss_count, 1)
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description="Train a lightweight molecule reranker on offline rerank caches.")
    parser.add_argument(
        "--train_cache",
        type=str,
        default=config.rerank.train.train_cache,
        help="Train cache path. Falls back to rerank.train.train_cache.",
    )
    parser.add_argument(
        "--val_cache",
        type=str,
        default=config.rerank.train.val_cache,
        help="Validation cache path. Falls back to rerank.train.val_cache.",
    )
    parser.add_argument("--save_dir", type=str, default=config.rerank.train.save_dir)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--max-train-queries", type=int, default=None)
    parser.add_argument(
        "--device",
        type=str,
        default=config.general.device,
        help='Device to use, for example "cpu", "cuda", "cuda:0", or "cuda:1".',
    )
    parser.add_argument(
        "--model_type",
        type=str,
        choices=["transformer", "pointwise", "relative"],
        default=config.rerank.train.model_type,
        help="Reranker variant. Other hyperparameters are read from rerank.train in params.yaml.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=config.general.seed,
        help="Random seed. Defaults to general.seed in params.yaml.",
    )
    parser.add_argument(
        "--base-score-feature",
        dest="use_base_score_feature",
        action=argparse.BooleanOptionalAction,
        default=bool(config.rerank.train.use_base_score_feature),
        help="Use the base score as an input pair feature.",
    )
    parser.add_argument(
        "--residual-score",
        dest="use_residual_score",
        action=argparse.BooleanOptionalAction,
        default=bool(config.rerank.train.use_residual_score),
        help="Add the learned alpha-scaled base score to the predicted correction.",
    )
    parser.add_argument(
        "--rank-embedding",
        dest="use_rank_embedding",
        action=argparse.BooleanOptionalAction,
        default=bool(config.rerank.train.use_rank_embedding),
        help="Use the embedding of the base retrieval rank.",
    )
    parser.add_argument(
        "--product-feature",
        dest="use_product_feature",
        action=argparse.BooleanOptionalAction,
        default=bool(config.rerank.train.use_product_feature),
        help="Use the elementwise spectrum-candidate product feature.",
    )
    parser.add_argument(
        "--abs-diff-feature",
        dest="use_abs_diff_feature",
        action=argparse.BooleanOptionalAction,
        default=bool(config.rerank.train.use_abs_diff_feature),
        help="Use the absolute spectrum-candidate difference feature.",
    )
    parser.add_argument(
        "--shuffle-candidates",
        action=argparse.BooleanOptionalAction,
        default=bool(config.rerank.train.shuffle_candidates),
        help="Randomize candidate order within each training list.",
    )
    parser.add_argument(
        "--lambda-pair",
        type=float,
        default=float(config.rerank.train.lambda_pair),
        help="Weight of the pairwise ranking loss; use 0 for listwise CE only.",
    )
    parser.add_argument(
        "--lambda-spec",
        type=float,
        default=float(getattr(config.rerank.train, "lambda_spec", 0.0)),
        help="Weight of the mismatched-spectrum dependency loss.",
    )
    parser.add_argument(
        "--spec-margin",
        type=float,
        default=float(getattr(config.rerank.train, "spec_margin", 0.1)),
        help="Margin for the mismatched-spectrum dependency loss.",
    )
    parser.add_argument(
        "--relation-dim",
        type=int,
        default=int(getattr(config.rerank.train, "relation_dim", 64)),
        help="Hidden dimension of the relative candidate branch.",
    )
    parser.add_argument(
        "--pair-chunk-size",
        type=int,
        default=int(getattr(config.rerank.train, "pair_chunk_size", 32)),
        help="Number of query candidates processed per relation chunk.",
    )
    parser.add_argument(
        "--relation-top-k",
        type=int,
        default=int(getattr(config.rerank.train, "relation_top_k", 40)),
        help="Number of coarse-scored candidates sent to the relative branch.",
    )
    parser.add_argument(
        "--relative-module",
        dest="use_relative_module",
        action=argparse.BooleanOptionalAction,
        default=bool(getattr(config.rerank.train, "use_relative_module", True)),
        help="Enable the explicit relative candidate branch for model_type=relative.",
    )
    parser.add_argument(
        "--spectrum-conditioning",
        dest="use_spectrum_conditioning",
        action=argparse.BooleanOptionalAction,
        default=bool(getattr(config.rerank.train, "use_spectrum_conditioning", True)),
        help="Condition relative relations on the spectrum embedding.",
    )
    parser.add_argument(
        "--molecular-relation",
        dest="use_molecular_relation",
        action=argparse.BooleanOptionalAction,
        default=bool(getattr(config.rerank.train, "use_molecular_relation", True)),
        help="Use candidate embedding differences in the relative branch.",
    )
    parser.add_argument(
        "--train-k",
        type=int,
        default=int(config.rerank.train.train_k),
        help="Maximum candidates per training list.",
    )
    parser.add_argument(
        "--exclude-val-query-indices",
        nargs="*",
        type=int,
        default=[],
        help="Zero-based validation cache query indices excluded before model selection.",
    )
    args = parser.parse_args()

    if any(index < 0 for index in args.exclude_val_query_indices):
        parser.error("--exclude-val-query-indices must contain non-negative integers")
    args.exclude_val_query_indices = sorted(set(args.exclude_val_query_indices))

    args.batch_size = int(
        config.rerank.train.batch_size if args.batch_size is None else args.batch_size
    )
    args.epochs = int(config.rerank.train.epochs if args.epochs is None else args.epochs)
    args.lr = float(config.rerank.train.lr)
    args.weight_decay = float(config.rerank.train.weight_decay)
    args.patience = int(config.rerank.train.patience if args.patience is None else args.patience)
    args.hidden_dim = int(config.rerank.train.hidden_dim)
    args.rank_emb_dim = int(config.rerank.train.rank_emb_dim)
    args.max_rank = int(config.rerank.train.max_rank)
    args.n_layers = int(config.rerank.train.n_layers)
    args.n_heads = int(config.rerank.train.n_heads)
    args.dropout = float(config.rerank.train.dropout)
    args.alpha_init = float(config.rerank.train.alpha_init)
    args.margin = float(config.rerank.train.margin)
    args.grad_clip = float(config.rerank.train.grad_clip)
    args.metric_for_best = config.rerank.train.metric_for_best
    args.num_workers = int(config.rerank.train.num_workers)
    args.top_k = [int(k) for k in config.rerank.train.top_k]

    if args.model_type == "relative":
        # Rank is intentionally unavailable to the formal model.  Keep the
        # legacy CLI flag for old checkpoints, but make the new model safe by
        # construction even when the global legacy default is true.
        args.use_rank_embedding = False

    if not args.train_cache:
        parser.error("--train_cache is required unless rerank.train.train_cache is set in params.yaml")
    if not args.val_cache:
        parser.error("--val_cache is required unless rerank.train.val_cache is set in params.yaml")
    if args.metric_for_best not in {"mrr", "top1", "top5"}:
        parser.error("rerank.train.metric_for_best must be one of: mrr, top1, top5")
    if not args.top_k:
        parser.error("rerank.train.top_k must contain at least one value")
    if args.metric_for_best.startswith("top"):
        metric_k = int(args.metric_for_best.removeprefix("top"))
        if metric_k not in args.top_k:
            parser.error(f"rerank.train.metric_for_best={args.metric_for_best} requires {metric_k} in top_k")
    return args


def main():
    configure_runtime_cache()
    args = parse_args()

    if args.train_k <= 0:
        raise ValueError("rerank.train.train_k must be greater than 0")
    if args.seed < 0:
        raise ValueError("--seed must be greater than or equal to 0")
    if args.batch_size <= 0:
        raise ValueError("rerank.train.batch_size must be greater than 0")
    if args.epochs <= 0:
        raise ValueError("rerank.train.epochs must be greater than 0")
    if args.num_workers < 0:
        raise ValueError("rerank.train.num_workers must be greater than or equal to 0")
    if args.hidden_dim <= 0:
        raise ValueError("rerank.train.hidden_dim must be greater than 0")
    if args.n_heads <= 0:
        raise ValueError("rerank.train.n_heads must be greater than 0")
    if args.hidden_dim % args.n_heads != 0:
        raise ValueError("rerank.train.hidden_dim must be divisible by rerank.train.n_heads")
    if args.rank_emb_dim <= 0:
        raise ValueError("rerank.train.rank_emb_dim must be greater than 0")
    if args.max_rank <= 0:
        raise ValueError("rerank.train.max_rank must be greater than 0")
    if args.n_layers < 0:
        raise ValueError("rerank.train.n_layers must be greater than or equal to 0")
    if args.lambda_pair < 0:
        raise ValueError("--lambda-pair must be greater than or equal to 0")
    if args.lambda_spec < 0:
        raise ValueError("--lambda-spec must be greater than or equal to 0")
    if args.spec_margin < 0:
        raise ValueError("--spec-margin must be greater than or equal to 0")
    if args.relation_dim <= 0:
        raise ValueError("--relation-dim must be greater than 0")
    if args.pair_chunk_size <= 0:
        raise ValueError("--pair-chunk-size must be greater than 0")
    if args.max_train_queries is not None and args.max_train_queries <= 0:
        raise ValueError("--max-train-queries must be greater than 0")
    if args.relation_top_k <= 0:
        raise ValueError("--relation-top-k must be greater than 0")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(save_dir / "train_rerank.log")
    startup_logging(args, "Train SpecEmbedding reranker")
    set_seed(args.seed)
    device = resolve_device(args.device)

    train_dataset = RerankCacheDataset(
        args.train_cache,
        max_candidates=args.train_k,
        shuffle_candidates=args.shuffle_candidates,
        require_label=True,
    )
    if args.max_train_queries is not None:
        train_dataset.indices = train_dataset.indices[: args.max_train_queries]
        if not train_dataset.indices:
            raise ValueError("--max-train-queries removed every labeled training query")
    val_dataset = RerankCacheDataset(
        args.val_cache,
        max_candidates=None,
        shuffle_candidates=False,
        require_label=False,
    )
    excluded_val_count = exclude_query_indices(
        val_dataset,
        args.exclude_val_query_indices,
        split_name="validation",
    )
    if excluded_val_count:
        logging.info(
            "Excluded %s validation cache queries before model selection: %s",
            excluded_val_count,
            args.exclude_val_query_indices,
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

    model = build_reranker(args, embedding_dim=train_dataset.embedding_dim).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    model_config = reranker_model_config(args, embedding_dim=train_dataset.embedding_dim)

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
            loss = compute_rerank_loss(
                scores=scores,
                labels=labels,
                candidate_mask=candidate_mask,
                lambda_pair=args.lambda_pair,
                margin=args.margin,
            )
            if loss is None:
                continue
            if args.lambda_spec > 0:
                spec_loss = spectrum_dependency_loss(
                    model=model,
                    spec_emb=spec_emb,
                    candidate_embs=candidate_embs,
                    base_scores=base_scores,
                    base_ranks=base_ranks,
                    candidate_mask=candidate_mask,
                    labels=labels,
                    margin=args.spec_margin,
                    scores=scores,
                )
                if spec_loss is not None:
                    loss = loss + args.lambda_spec * spec_loss
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
                    "seed": args.seed,
                    "training_config": vars(args).copy(),
                },
                save_path,
            )
            logging.info("Saved best reranker to %s", save_path)
        else:
            patience_counter += 1
            logging.info("EarlyStopping counter: %s out of %s", patience_counter, args.patience)
            if patience_counter >= args.patience:
                break

    torch.save(
        {
            "state_dict": model.state_dict(),
            "model_config": model_config,
            "seed": args.seed,
            "training_config": vars(args).copy(),
        },
        save_dir / "last_reranker.pth",
    )
    logging.info("Training finished. Best epoch=%s best_%s=%.4f", best_epoch, args.metric_for_best, best_metric)


if __name__ == "__main__":
    main()
