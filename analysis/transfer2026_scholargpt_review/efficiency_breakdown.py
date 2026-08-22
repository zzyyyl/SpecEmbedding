"""Measure fixed-hardware reranker cost and split coarse/relation work."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from SpecEmbedding.data.datasets_rerank import RerankCacheDataset, rerank_collate_fn
from SpecEmbedding.utils.rerank import load_reranker


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def elapsed_call(function, device: torch.device) -> float:
    synchronize(device)
    started = time.perf_counter()
    function()
    synchronize(device)
    return time.perf_counter() - started


@torch.no_grad()
def benchmark_model(
    model,
    loader: DataLoader,
    device: torch.device,
    max_queries: int,
    split_relative: bool = False,
) -> dict:
    timings: list[float] = []
    queries = 0
    batches = 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for batch in loader:
        if queries >= max_queries:
            break
        tensors = {key: value.to(device) for key, value in batch.items() if torch.is_tensor(value)}
        count = min(int(tensors["spec_emb"].shape[0]), max_queries - queries)
        tensors = {key: value[:count] if value.ndim > 0 else value for key, value in tensors.items()}
        queries += count
        batches += 1
        if split_relative and hasattr(model, "_relative_scores"):
            def run_coarse() -> tuple[torch.Tensor, torch.Tensor]:
                absolute = model.absolute_score(
                    model.absolute_mlp(model._absolute_features(
                        tensors["spec_emb"], tensors["candidate_embs"], tensors["base_scores"]
                    ))
                ).squeeze(-1)
                coarse = absolute + model.alpha * tensors["base_scores"] if model.use_residual_score else absolute
                indices = torch.topk(
                    coarse.masked_fill(~tensors["candidate_mask"], torch.finfo(coarse.dtype).min),
                    k=min(model.relation_top_k, coarse.shape[1]), dim=-1, sorted=False,
                ).indices
                return coarse, indices

            coarse_seconds = elapsed_call(run_coarse, device)
            coarse, indices = run_coarse()

            def run_relation() -> None:
                gather = indices.unsqueeze(-1).expand(-1, -1, tensors["candidate_embs"].shape[-1])
                relation_candidates = torch.gather(tensors["candidate_embs"], 1, gather)
                relation_base = torch.gather(tensors["base_scores"], 1, indices)
                relation_mask = torch.gather(tensors["candidate_mask"], 1, indices)
                model._relative_scores(tensors["spec_emb"], relation_candidates, relation_base, relation_mask)

            relation_seconds = elapsed_call(run_relation, device)
            timings.append((coarse_seconds, relation_seconds))
        else:
            timings.append((elapsed_call(lambda: model(
                tensors["spec_emb"], tensors["candidate_embs"], tensors["base_scores"],
                tensors["base_ranks"], tensors["candidate_mask"],
            ), device),))
    aggregate = {
        "measurement_queries": queries,
        "measurement_batches": batches,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
    }
    if split_relative:
        coarse = sum(value[0] for value in timings) / max(queries, 1)
        relation = sum(value[1] for value in timings) / max(queries, 1)
        aggregate.update({
            "coarse_ms_per_query": coarse * 1000,
            "relation_ms_per_query": relation * 1000,
            "full_ms_per_query": (coarse + relation) * 1000,
            "full_queries_per_second": queries / max(sum(value[0] + value[1] for value in timings), 1e-12),
        })
    else:
        full = sum(value[0] for value in timings) / max(queries, 1)
        aggregate.update({
            "full_ms_per_query": full * 1000,
            "full_queries_per_second": queries / max(sum(value[0] for value in timings), 1e-12),
        })
    if device.type == "cuda":
        aggregate["peak_cuda_memory_mb"] = torch.cuda.max_memory_allocated(device) / (1024**2)
    else:
        aggregate["peak_cuda_memory_mb"] = None
    return aggregate


def evaluate(spec: dict, device: torch.device, max_queries: int, batch_size: int) -> dict:
    cache = Path(spec["cache"])
    dataset = RerankCacheDataset(cache, return_smiles=False, require_label=False)
    dataset.indices = dataset.indices[:max_queries]
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=rerank_collate_fn)
    output = {"cache": str(cache), "candidates": int(dataset[0]["candidate_embs"].shape[0])}
    for name, model_path, split in spec["models"]:
        model = load_reranker(Path(model_path), device)
        output[name] = benchmark_model(model, loader, device, max_queries, split_relative=split)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-queries", type=int, default=2048)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--spec", action="append", required=True, metavar="JSON_SPEC")
    args = parser.parse_args()
    device = torch.device(args.device)
    result = {
        "hardware": {"device": str(device), "torch": torch.__version__, "cuda": torch.cuda.is_available()},
        "batch_size": args.batch_size,
        "max_queries": args.max_queries,
        "measurement": "fixed-hardware forward audit; timings exclude data loading",
        "pools": [json.loads(spec) for spec in args.spec],
    }
    result["results"] = [evaluate(spec, device, args.max_queries, args.batch_size) for spec in result["pools"]]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
