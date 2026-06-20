import argparse
import logging
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pulp
import torch
from myopic_mces.myopic_mces import MCES
from torch.utils.data import DataLoader
from tqdm import tqdm

from SpecEmbedding.config import config
from SpecEmbedding.data.datasets_rerank import RerankCacheDataset, rerank_collate_fn
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

configure_runtime_cache()

mces_solvers = pulp.listSolvers(onlyAvailable=True)
mces_solver = "MOSEK" if "MOSEK" in mces_solvers else mces_solvers[0]


def mces_worker(smiles_pair):
    pred_smiles, true_smiles = smiles_pair
    if pred_smiles == true_smiles:
        return 0.0, False
    try:
        retval = MCES(
            smiles1=pred_smiles,
            smiles2=true_smiles,
            threshold=15,
            always_stronger_bound=True,
            solver=mces_solver,
            solver_options=dict(msg=0),
        )
        return retval[1], False
    except Exception:
        return 0.0, True


@torch.no_grad()
def evaluate(model, loader, device, top_k):
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


def compute_mces(pairs, label):
    if not pairs:
        return None
    max_workers = min(os.cpu_count() or 1, 16)
    logging.info("Computing %s MCES for %s pairs using %s solver...", label, len(pairs), mces_solver)
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(tqdm(executor.map(mces_worker, pairs), total=len(pairs), desc=f"{label} MCES", ascii=True))
    mces = sum(item[0] for item in results) / len(pairs)
    errors = sum(item[1] for item in results)
    if errors:
        logging.warning("%s MCES calculation errors: %s/%s", label, errors, len(pairs))
    return mces


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
    return args


def main():
    args = parse_args()

    if args.batch_size <= 0:
        raise ValueError("rerank.eval.batch_size must be greater than 0")
    if args.num_workers < 0:
        raise ValueError("rerank.eval.num_workers must be greater than or equal to 0")

    checkpoint_path = Path(args.checkpoint)
    save_dir = Path(args.save_dir) if args.save_dir else checkpoint_path.parent
    save_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(save_dir / "eval_rerank.log")
    startup_logging(args, "Evaluate SpecEmbedding reranker")
    device = resolve_device(args.device)

    dataset = RerankCacheDataset(args.cache, return_smiles=True, require_label=False)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=rerank_collate_fn,
        num_workers=args.num_workers,
    )
    model = load_reranker(args.checkpoint, device)
    top_k = sorted(set(args.top_k))
    results = evaluate(model, loader, device, top_k)

    logging.info("=" * 50)
    logging.info("Total queries: %s", results["total"])
    logging.info("Pre-retrieval upper bound: %.4f%%", results["upper_bound"] * 100)
    log_ranking_summary("BASE", results["base"], top_k)
    log_ranking_summary("RERANK", results["rerank"], top_k)

    if args.compute_mces:
        base_mces = compute_mces(results["base_mces_pairs"], "Base")
        rerank_mces = compute_mces(results["rerank_mces_pairs"], "Rerank")
        logging.info("Base   MCES@1 : %.4f", base_mces)
        logging.info("Rerank MCES@1 : %.4f", rerank_mces)
    else:
        logging.info("MCES calculation skipped.")
    logging.info("=" * 50)


if __name__ == "__main__":
    main()
