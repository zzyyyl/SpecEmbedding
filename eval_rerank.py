import argparse
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from SpecEmbedding.config import config
from SpecEmbedding.data.datasets_rerank import (
    RerankCacheDataset,
    exclude_query_indices,
    rerank_collate_fn,
)
from SpecEmbedding.utils.mces import compute_mces
from SpecEmbedding.utils.rerank import (
    init_ranking_metrics,
    load_reranker,
    log_ranking_summary,
    masked_argmax,
    rank_from_scores,
    summarize_ranking_metrics,
    update_ranking_metrics,
)
from SpecEmbedding.utils.runtime import configure_runtime_cache, resolve_device, setup_logging, startup_logging


@torch.no_grad()
def evaluate(model, loader, device, top_k, *, shuffle_spectrum: bool = False):
    base_metrics = init_ranking_metrics(top_k)
    rerank_metrics = init_ranking_metrics(top_k)
    positive_count = 0
    base_mces_pairs = []
    rerank_mces_pairs = []

    for batch in tqdm(loader, desc="Rerank evaluation", ascii=True):
        spec_emb = batch["spec_emb"].to(device)
        candidate_embs = batch["candidate_embs"].to(device)
        base_scores = batch["base_scores"].to(device)
        base_ranks = batch["base_ranks"].to(device)
        candidate_mask = batch["candidate_mask"].to(device)
        labels = batch["labels"].to(device)

        if shuffle_spectrum:
            spec_emb = torch.roll(spec_emb, shifts=1, dims=0)
        rerank_scores = model(spec_emb, candidate_embs, base_scores, base_ranks, candidate_mask)
        base_top1 = masked_argmax(base_scores, candidate_mask).cpu().tolist()
        rerank_top1 = masked_argmax(rerank_scores, candidate_mask).cpu().tolist()

        for row in range(labels.numel()):
            label = int(labels[row].item())
            true_smiles = batch["true_smiles"][row]
            candidate_smiles = batch["candidate_smiles"][row]

            base_mces_pairs.append((candidate_smiles[base_top1[row]], true_smiles))
            rerank_mces_pairs.append((candidate_smiles[rerank_top1[row]], true_smiles))

            if label < 0:
                update_ranking_metrics(base_metrics, None, top_k)
                update_ranking_metrics(rerank_metrics, None, top_k)
                continue

            positive_count += 1
            base_rank = int(base_ranks[row, label].item())
            rerank_rank = rank_from_scores(rerank_scores[row], label, candidate_mask[row])

            update_ranking_metrics(base_metrics, base_rank, top_k)
            update_ranking_metrics(rerank_metrics, rerank_rank, top_k)

    return {
        "base": summarize_ranking_metrics(base_metrics, top_k),
        "rerank": summarize_ranking_metrics(rerank_metrics, top_k),
        "upper_bound": positive_count / max(base_metrics["total"], 1),
        "base_mces_pairs": base_mces_pairs,
        "rerank_mces_pairs": rerank_mces_pairs,
        "total": base_metrics["total"],
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate reranked spectrum-to-molecule retrieval from a rerank cache."
    )
    parser.add_argument(
        "--cache",
        type=str,
        default=config.rerank.eval.cache,
        help="Rerank cache path. Falls back to rerank.eval.cache.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=config.rerank.eval.checkpoint,
        help="Reranker checkpoint path. Falls back to rerank.eval.checkpoint.",
    )
    parser.add_argument("--save_dir", type=str, default=config.rerank.eval.save_dir)
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=None,
        help="Optionally truncate an existing full-pool cache without inserting positives.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=config.general.device,
        help='Device to use, for example "cpu", "cuda", "cuda:0", or "cuda:1".',
    )
    parser.add_argument(
        "--mces",
        dest="compute_mces",
        action=argparse.BooleanOptionalAction,
        default=config.rerank.eval.compute_mces,
        help="Enable MCES@1 calculation. Use --no-mces for quick metric-only evaluation.",
    )
    parser.add_argument(
        "--shuffle-spectrum",
        action="store_true",
        help="Cyclically mismatch spectra within each batch for a spectrum-dependence audit.",
    )
    parser.add_argument(
        "--exclude-query-indices",
        nargs="*",
        type=int,
        default=[],
        help="Zero-based cache query indices excluded from a sensitivity evaluation.",
    )
    args = parser.parse_args()

    args.batch_size = int(config.rerank.eval.batch_size)
    args.num_workers = int(config.rerank.eval.num_workers)
    args.top_k = [int(k) for k in config.rerank.eval.top_k]

    if not args.cache:
        parser.error("--cache is required unless rerank.eval.cache is set in params.yaml")
    if not args.checkpoint:
        parser.error("--checkpoint is required unless rerank.eval.checkpoint is set in params.yaml")
    if not args.top_k:
        parser.error("rerank.eval.top_k must contain at least one value")
    if args.exclude_query_indices and not args.save_dir:
        parser.error("--save_dir is required when --exclude-query-indices is used")
    return args


def main():
    configure_runtime_cache()
    args = parse_args()

    if args.batch_size <= 0:
        raise ValueError("rerank.eval.batch_size must be greater than 0")
    if args.num_workers < 0:
        raise ValueError("rerank.eval.num_workers must be greater than or equal to 0")
    if args.max_candidates is not None and args.max_candidates <= 0:
        raise ValueError("--max-candidates must be greater than 0")

    checkpoint_path = Path(args.checkpoint)
    save_dir = Path(args.save_dir) if args.save_dir else checkpoint_path.parent
    if args.exclude_query_indices and save_dir.resolve() == checkpoint_path.parent.resolve():
        raise ValueError("Sensitivity evaluation --save_dir must differ from the checkpoint directory")
    save_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(save_dir / "eval_rerank.log")
    startup_logging(args, "Evaluate SpecEmbedding reranker")
    device = resolve_device(args.device)

    dataset = RerankCacheDataset(
        args.cache,
        max_candidates=args.max_candidates,
        return_smiles=True,
        require_label=False,
    )
    excluded_count = exclude_query_indices(
        dataset,
        args.exclude_query_indices,
        split_name="evaluation",
    )
    if excluded_count:
        logging.info(
            "Excluded %s cache queries for sensitivity evaluation: %s",
            excluded_count,
            sorted(set(args.exclude_query_indices)),
        )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=rerank_collate_fn,
        num_workers=args.num_workers,
    )
    model = load_reranker(args.checkpoint, device)
    top_k = sorted(set(args.top_k))
    results = evaluate(model, loader, device, top_k, shuffle_spectrum=args.shuffle_spectrum)

    logging.info("=" * 50)
    logging.info("Total queries: %s", results["total"])
    if args.max_candidates is not None:
        logging.info("Evaluation candidate truncation: K=%s (no positive forcing)", args.max_candidates)
    if args.shuffle_spectrum:
        logging.info("Spectrum-dependence audit: spectra cyclically shuffled within each batch")
    logging.info("Pre-retrieval upper bound: %.4f%%", results["upper_bound"] * 100)
    log_ranking_summary("BASE", results["base"], top_k)
    log_ranking_summary("RERANK", results["rerank"], top_k)

    if args.compute_mces:
        base_mces = compute_mces(results["base_mces_pairs"], "Base MCES")
        rerank_mces = compute_mces(results["rerank_mces_pairs"], "Rerank MCES")
        logging.info("Base   MCES@1 : %.4f", base_mces)
        logging.info("Rerank MCES@1 : %.4f", rerank_mces)
    else:
        logging.info("MCES calculation skipped.")
    logging.info("=" * 50)


if __name__ == "__main__":
    main()
