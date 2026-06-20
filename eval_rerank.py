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
from SpecEmbedding.models_rerank import CandidateReranker, PointwiseReranker
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


def load_reranker(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_config = checkpoint["model_config"]
    model_type = model_config.get("model_type", "transformer")
    model_cls = PointwiseReranker if model_type == "pointwise" else CandidateReranker
    model = model_cls(
        embedding_dim=model_config["embedding_dim"],
        hidden_dim=model_config["hidden_dim"],
        rank_emb_dim=model_config["rank_emb_dim"],
        max_rank=model_config["max_rank"],
        n_layers=model_config["n_layers"],
        n_heads=model_config["n_heads"],
        dropout=model_config["dropout"],
        alpha_init=model_config.get("alpha_init", 1.0),
    )
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model = model.to(device)
    model.eval()
    return model


def init_metrics(top_k):
    return {"total": 0, "hits": {k: 0 for k in top_k}, "mrr_sum": 0.0}


def update_metrics(metrics, rank, top_k):
    metrics["total"] += 1
    if rank is None:
        return
    for k in top_k:
        if rank <= k:
            metrics["hits"][k] += 1
    metrics["mrr_sum"] += 1.0 / rank


def summarize_metrics(metrics, top_k):
    total = max(metrics["total"], 1)
    summary = {f"top{k}": metrics["hits"][k] / total for k in top_k}
    summary["mrr"] = metrics["mrr_sum"] / total
    return summary


def masked_argmax(scores, mask):
    scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
    return torch.argmax(scores, dim=-1)


@torch.no_grad()
def evaluate(model, loader, device, top_k):
    base_metrics = init_metrics(top_k)
    rerank_metrics = init_metrics(top_k)
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
                update_metrics(base_metrics, None, top_k)
                update_metrics(rerank_metrics, None, top_k)
                continue

            positive_count += 1
            base_rank = int(base_ranks[row, label].item())
            true_score = rerank_scores[row, label]
            rerank_rank = int((rerank_scores[row].masked_select(candidate_mask[row]) > true_score).sum().item()) + 1

            update_metrics(base_metrics, base_rank, top_k)
            update_metrics(rerank_metrics, rerank_rank, top_k)

    return {
        "base": summarize_metrics(base_metrics, top_k),
        "rerank": summarize_metrics(rerank_metrics, top_k),
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


def log_summary(name, summary, top_k):
    logging.info("%s RESULTS", name)
    for k in top_k:
        logging.info("  Top-%-2s Accuracy : %.4f%%", k, summary[f"top{k}"] * 100)
    logging.info("  MRR              : %.4f", summary["mrr"])


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate reranked spectrum-to-molecule retrieval from a rerank cache."
    )
    parser.add_argument("--cache", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--device", type=str, default=config.general.device)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--top_k", type=int, nargs="+", default=config.eval.top_k)
    parser.add_argument("--no-mces", action="store_true")
    args = parser.parse_args()

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
    log_summary("BASE", results["base"], top_k)
    log_summary("RERANK", results["rerank"], top_k)

    if not args.no_mces:
        base_mces = compute_mces(results["base_mces_pairs"], "Base")
        rerank_mces = compute_mces(results["rerank_mces_pairs"], "Rerank")
        logging.info("Base   MCES@1 : %.4f", base_mces)
        logging.info("Rerank MCES@1 : %.4f", rerank_mces)
    else:
        logging.info("MCES calculation skipped.")
    logging.info("=" * 50)


if __name__ == "__main__":
    main()
