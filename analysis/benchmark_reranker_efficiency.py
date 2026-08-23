"""Benchmark reranker forward cost across candidate-pool cutoffs.

The benchmark intentionally measures model forward work only.  It does not
load spectra, compute metrics, or make an end-to-end deployment claim.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from SpecEmbedding.data.datasets_rerank import RerankCacheDataset, rerank_collate_fn
from SpecEmbedding.models_rerank import RelativeCandidateReranker
from SpecEmbedding.utils.rerank import load_reranker


def parse_checkpoint_spec(value: str) -> tuple[str, Path]:
    name, separator, path = value.partition("=")
    if not separator or not name or not path:
        raise ValueError("checkpoint must use NAME=PATH syntax")
    return name, Path(path)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def timed_call(function, device: torch.device) -> float:
    synchronize(device)
    started = time.perf_counter()
    function()
    synchronize(device)
    return time.perf_counter() - started


def relative_coarse(model, tensors):
    absolute = model.absolute_score(
        model.absolute_mlp(
            model._absolute_features(
                tensors["spec_emb"],
                tensors["candidate_embs"],
                tensors["base_scores"],
            )
        )
    ).squeeze(-1)
    coarse = absolute
    if model.use_residual_score:
        coarse = coarse + model.alpha * tensors["base_scores"]
    masked = coarse.masked_fill(
        ~tensors["candidate_mask"],
        torch.finfo(coarse.dtype).min,
    )
    indices = torch.topk(
        masked,
        k=min(model.relation_top_k, coarse.shape[1]),
        dim=-1,
        largest=True,
        sorted=False,
    ).indices
    return coarse, indices


def relative_relation(model, tensors, relation_indices):
    gather_indices = relation_indices.unsqueeze(-1).expand(
        -1, -1, tensors["candidate_embs"].shape[-1]
    )
    relation_candidates = torch.gather(
        tensors["candidate_embs"], 1, gather_indices
    )
    relation_base_scores = torch.gather(
        tensors["base_scores"], 1, relation_indices
    )
    relation_mask = torch.gather(
        tensors["candidate_mask"], 1, relation_indices
    )
    return model._relative_scores(
        tensors["spec_emb"],
        relation_candidates,
        relation_base_scores,
        relation_mask,
    )


@torch.inference_mode()
def benchmark_model(
    model,
    loader: DataLoader,
    device: torch.device,
    *,
    warmup_batches: int,
    measure_batches: int,
) -> dict:
    if warmup_batches < 0 or measure_batches <= 0:
        raise ValueError("warmup_batches must be >= 0 and measure_batches must be > 0")

    is_relative = isinstance(model, RelativeCandidateReranker) and model.use_relative_module
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    measured_queries = 0
    measured_batches = 0
    coarse_seconds = 0.0
    relation_seconds = 0.0
    full_seconds = 0.0

    for batch_index, batch in enumerate(loader):
        tensors = {
            key: value.to(device)
            for key, value in batch.items()
            if torch.is_tensor(value)
        }

        def run_full() -> None:
            model(
                tensors["spec_emb"],
                tensors["candidate_embs"],
                tensors["base_scores"],
                tensors["base_ranks"],
                tensors["candidate_mask"],
            )

        if batch_index < warmup_batches:
            run_full()
            continue
        if measured_batches >= measure_batches:
            break

        query_count = int(tensors["spec_emb"].shape[0])
        measured_queries += query_count
        measured_batches += 1
        if is_relative:
            coarse_result = {}

            def run_coarse() -> None:
                coarse_result["value"] = relative_coarse(model, tensors)

            coarse_seconds += timed_call(run_coarse, device)
            relation_indices = coarse_result["value"][1]
            relation_seconds += timed_call(
                lambda: relative_relation(model, tensors, relation_indices),
                device,
            )
        else:
            full_seconds += timed_call(run_full, device)

    if measured_batches == 0:
        raise ValueError("loader produced no measured batches")

    if is_relative:
        full_seconds = coarse_seconds + relation_seconds
    result = {
        "model_type": getattr(model, "pair_mode", type(model).__name__),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "measurement_queries": measured_queries,
        "measurement_batches": measured_batches,
        "coarse_ms_per_query": (
            coarse_seconds * 1000 / measured_queries if is_relative else None
        ),
        "relation_ms_per_query": (
            relation_seconds * 1000 / measured_queries if is_relative else None
        ),
        "full_ms_per_query": full_seconds * 1000 / measured_queries,
        "queries_per_second": measured_queries / max(full_seconds, 1e-12),
        "peak_cuda_memory_mb": (
            torch.cuda.max_memory_allocated(device) / (1024**2)
            if device.type == "cuda"
            else None
        ),
    }
    return result


def benchmark_checkpoint(
    cache: Path,
    checkpoint: Path,
    candidate_counts: list[int],
    device: torch.device,
    *,
    batch_size: int,
    warmup_batches: int,
    measure_batches: int,
) -> list[dict]:
    model = load_reranker(checkpoint, device)
    results = []
    for candidate_count in candidate_counts:
        dataset = RerankCacheDataset(
            cache,
            max_candidates=candidate_count,
            return_smiles=False,
            require_label=False,
        )
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=rerank_collate_fn,
        )
        result = benchmark_model(
            model,
            loader,
            device,
            warmup_batches=warmup_batches,
            measure_batches=measure_batches,
        )
        result.update(
            {
                "candidate_count": candidate_count,
                "relation_top_k": getattr(model, "relation_top_k", None),
            }
        )
        results.append(result)
    return results


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Measure reranker forward cost over candidate-pool cutoffs."
    )
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument(
        "--checkpoint",
        action="append",
        required=True,
        metavar="NAME=PATH",
        help="Checkpoint label and path; repeat for multiple models.",
    )
    parser.add_argument(
        "--candidate-counts",
        type=int,
        nargs="+",
        default=[40, 80, 160, 256],
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--warmup-batches", type=int, default=2)
    parser.add_argument("--measure-batches", type=int, default=16)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if any(count <= 0 for count in args.candidate_counts):
        parser.error("--candidate-counts must contain only positive integers")
    if args.batch_size <= 0:
        parser.error("--batch-size must be greater than 0")
    if args.warmup_batches < 0:
        parser.error("--warmup-batches must be greater than or equal to 0")
    if args.measure_batches <= 0:
        parser.error("--measure-batches must be greater than 0")
    return args


def main(argv=None) -> None:
    args = parse_args(argv)
    device = torch.device(args.device)
    checkpoints = [parse_checkpoint_spec(value) for value in args.checkpoint]
    result = {
        "cache": str(args.cache),
        "candidate_counts": args.candidate_counts,
        "batch_size": args.batch_size,
        "warmup_batches": args.warmup_batches,
        "measure_batches": args.measure_batches,
        "device": str(device),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "measurement": "forward-only; excludes DataLoader and data loading",
        "models": {},
    }
    for name, checkpoint in checkpoints:
        result["models"][name] = {
            "checkpoint": str(checkpoint),
            "results": benchmark_checkpoint(
                args.cache,
                checkpoint,
                args.candidate_counts,
                device,
                batch_size=args.batch_size,
                warmup_batches=args.warmup_batches,
                measure_batches=args.measure_batches,
            ),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
