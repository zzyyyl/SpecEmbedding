import argparse
import hashlib
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from SpecEmbedding.config import config
from SpecEmbedding.data.datasets_eval import MolSmilesDataset, SpecSequenceDataset, mol_collate_fn
from SpecEmbedding.data.tokenizer import Tokenizer
from SpecEmbedding.type import TokenizerConfig
from SpecEmbedding.utils.align import load_align_model, resolve_storage_dtype
from SpecEmbedding.utils.providers import get_provider, load_candidates
from SpecEmbedding.utils.runtime import configure_runtime_cache, resolve_device, setup_logging, startup_logging


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_unique_candidate_list(sequences, candidates_dict, limit: int = 0):
    unique_smiles = []
    smiles_to_idx = {}
    matched_sequences = []

    for seq in sequences:
        true_smiles = seq["smiles"]
        candidates = candidates_dict.get(true_smiles)
        if candidates is None:
            continue
        matched_sequences.append(seq)
        for smiles in [true_smiles, *candidates]:
            if smiles not in smiles_to_idx:
                smiles_to_idx[smiles] = len(unique_smiles)
                unique_smiles.append(smiles)
        if limit > 0 and len(matched_sequences) >= limit:
            break

    return matched_sequences, unique_smiles, smiles_to_idx


@torch.no_grad()
def encode_molecules(model, smiles_list, args, device):
    storage_device = device if args.mol_embedding_storage == "cuda" and device.type == "cuda" else torch.device("cpu")
    storage_dtype = resolve_storage_dtype(args.mol_embedding_dtype)
    mol_embs = torch.zeros(
        (len(smiles_list), config.model.align.final_dim),
        dtype=storage_dtype,
        device=storage_device,
    )
    valid_mol_mask = torch.zeros(len(smiles_list), dtype=torch.bool)

    mol_loader = DataLoader(
        MolSmilesDataset(smiles_list),
        batch_size=args.mol_batch_size,
        shuffle=False,
        collate_fn=mol_collate_fn,
        num_workers=args.num_workers,
    )

    for batch in tqdm(mol_loader, desc="Molecule embeddings", ascii=True):
        if batch is None:
            continue
        mol_graph = batch["mol_graph"].to(device)
        indices = torch.tensor(batch["indices"], dtype=torch.long, device=storage_device)
        f_mol = model.encode_mol(mol_graph, normalize=True)
        mol_embs[indices] = f_mol.to(device=storage_device, dtype=storage_dtype)
        valid_mol_mask[torch.tensor(batch["indices"], dtype=torch.long)] = True

    return mol_embs, valid_mol_mask


@torch.no_grad()
def encode_spectra(model, sequences, args, device):
    spec_loader = DataLoader(
        SpecSequenceDataset(sequences),
        batch_size=args.spec_batch_size,
        shuffle=False,
        num_workers=0,
    )
    spec_embs = []
    true_smiles = []
    for batch in tqdm(spec_loader, desc="Spectrum embeddings", ascii=True):
        spec_mz = batch["spec_mz"].to(device)
        spec_intensity = batch["spec_intensity"].to(device)
        spec_mask = batch["spec_mask"].to(device)
        f_spec = model.encode_spec(spec_mz, spec_intensity, spec_mask, normalize=True)
        spec_embs.append(f_spec.cpu())
        true_smiles.extend(batch["smiles"])
    return torch.cat(spec_embs, dim=0), true_smiles


def update_topk(top_scores, top_indices, chunk_scores, chunk_indices, pre_top_k):
    if top_scores is None:
        scores = chunk_scores
        indices = chunk_indices
    else:
        scores = torch.cat([top_scores, chunk_scores], dim=0)
        indices = torch.cat([top_indices, chunk_indices], dim=0)

    keep = min(pre_top_k, scores.numel())
    top_scores, order = torch.topk(scores, k=keep, largest=True, sorted=True)
    top_indices = indices[order]
    return top_scores, top_indices


@torch.no_grad()
def build_query_record(
    spec_index,
    spec_emb,
    true_smiles,
    candidates,
    smiles_to_idx,
    mol_embs,
    valid_mol_mask,
    args,
    device,
):
    true_idx = smiles_to_idx.get(true_smiles)
    true_is_valid = true_idx is not None and bool(valid_mol_mask[true_idx])
    candidates = list(candidates or [])
    positive_in_source_candidates = true_smiles in candidates

    candidate_ids = []
    seen = set()
    candidate_smiles = candidates
    if args.force_include_positive and not positive_in_source_candidates:
        candidate_smiles = [true_smiles, *candidate_smiles]
    for smiles in candidate_smiles:
        idx = smiles_to_idx.get(smiles)
        if idx is None or idx in seen or not bool(valid_mol_mask[idx]):
            continue
        seen.add(idx)
        candidate_ids.append(idx)

    if not candidate_ids:
        return None

    storage_device = mol_embs.device
    query_emb = spec_emb.to(device).unsqueeze(0)
    positive_in_candidate_pool = true_is_valid and true_idx in seen
    true_score = None
    if positive_in_candidate_pool:
        true_emb = mol_embs[torch.tensor([true_idx], device=storage_device)].to(
            device=device,
            dtype=query_emb.dtype,
        )
        true_score = torch.mm(query_emb, true_emb.T).squeeze().item()

    top_scores = None
    top_indices = None
    num_higher_than_true = 0

    for start in range(0, len(candidate_ids), args.candidate_chunk_size):
        chunk_ids = candidate_ids[start : start + args.candidate_chunk_size]
        chunk_indices = torch.tensor(chunk_ids, dtype=torch.long, device=storage_device)
        chunk_embs = mol_embs[chunk_indices].to(device=device, dtype=query_emb.dtype)
        chunk_scores = torch.mm(query_emb, chunk_embs.T).squeeze(0)
        if true_score is not None:
            num_higher_than_true += int((chunk_scores > true_score).sum().item())
        top_scores, top_indices = update_topk(
            top_scores=top_scores,
            top_indices=top_indices,
            chunk_scores=chunk_scores,
            chunk_indices=chunk_indices.to(device),
            pre_top_k=args.pre_top_k,
        )

    top_indices_cpu = top_indices.cpu().long()
    top_scores_cpu = top_scores.cpu().float()
    if true_is_valid:
        positive_positions = (top_indices_cpu == true_idx).nonzero(as_tuple=False).flatten()
    else:
        positive_positions = torch.empty(0, dtype=torch.long)
    positive_in_base_topk = positive_in_source_candidates and positive_positions.numel() > 0
    positive_forced_into_candidate_pool = bool(
        args.force_include_positive
        and not positive_in_source_candidates
        and positive_in_candidate_pool
    )
    positive_forced_into_topk = bool(
        positive_forced_into_candidate_pool and positive_positions.numel() > 0
    )

    if positive_positions.numel() == 0 and args.force_include_positive and true_score is not None:
        true_rank = num_higher_than_true + 1
        forced_idx = torch.tensor([true_idx], dtype=torch.long)
        forced_score = torch.tensor([true_score], dtype=torch.float32)
        if top_indices_cpu.numel() >= args.pre_top_k:
            top_indices_cpu = torch.cat([top_indices_cpu[:-1], forced_idx], dim=0)
            top_scores_cpu = torch.cat([top_scores_cpu[:-1], forced_score], dim=0)
        else:
            top_indices_cpu = torch.cat([top_indices_cpu, forced_idx], dim=0)
            top_scores_cpu = torch.cat([top_scores_cpu, forced_score], dim=0)
        order = torch.argsort(top_scores_cpu, descending=True)
        top_indices_cpu = top_indices_cpu[order]
        top_scores_cpu = top_scores_cpu[order]
        base_ranks = torch.arange(1, top_indices_cpu.numel() + 1, dtype=torch.long)
        forced_position = (top_indices_cpu == true_idx).nonzero(as_tuple=False).flatten()
        base_ranks[forced_position] = true_rank
        positive_forced_into_topk = True
    else:
        base_ranks = torch.arange(1, top_indices_cpu.numel() + 1, dtype=torch.long)

    if true_is_valid:
        positive_positions = (top_indices_cpu == true_idx).nonzero(as_tuple=False).flatten()
    else:
        positive_positions = torch.empty(0, dtype=torch.long)
    label = None if positive_positions.numel() == 0 else int(positive_positions[0].item())

    return {
        "spec_index": spec_index,
        "true_smiles": true_smiles,
        "candidate_indices": top_indices_cpu,
        "base_scores": top_scores_cpu,
        "base_ranks": base_ranks,
        "label": label,
        "positive_in_source_candidates": positive_in_source_candidates,
        "positive_in_candidate_pool": positive_in_candidate_pool,
        "positive_in_base_topk": positive_in_base_topk,
        "positive_forced_into_candidate_pool": positive_forced_into_candidate_pool,
        "positive_forced_into_topk": positive_forced_into_topk,
        "source_candidate_pool_size": len(candidates),
        "candidate_pool_size": len(candidate_ids),
    }


def prune_molecule_embeddings(queries, mol_embs, smiles_list):
    used_indices = sorted({int(idx) for query in queries for idx in query["candidate_indices"].tolist()})
    old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(used_indices)}
    pruned_mol_embs = mol_embs.cpu()[torch.tensor(used_indices, dtype=torch.long)].contiguous()
    pruned_smiles = [smiles_list[idx] for idx in used_indices]

    for query in queries:
        query["candidate_indices"] = torch.tensor(
            [old_to_new[int(idx)] for idx in query["candidate_indices"].tolist()],
            dtype=torch.long,
        )

    return pruned_mol_embs, pruned_smiles


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare offline rerank cache from a frozen SpecMolAlignModel.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=config.rerank.prepare.checkpoint,
        help="Path to aligned model checkpoint. Falls back to rerank.prepare.checkpoint.",
    )
    parser.add_argument(
        "--dataset_type",
        type=str,
        choices=["massbank", "massspecgym", "nplib1", "gnps", "mona"],
        default=config.rerank.prepare.dataset_type,
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "val", "test"],
        default=config.rerank.prepare.split,
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=config.rerank.prepare.data_path or config.data.data_path,
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default=config.rerank.prepare.save_path,
        help="Output .pt rerank cache path. Falls back to rerank.prepare.save_path.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=config.general.device,
        help='Device to use, for example "cpu", "cuda", "cuda:0", or "cuda:1".',
    )
    parser.add_argument(
        "--candidate_type",
        type=str,
        choices=["mass", "formula", "supplied"],
        default=None,
        help=(
            "Candidate protocol. Defaults to 'supplied' for NPLIB1 and to "
            "rerank.prepare.candidate_type otherwise."
        ),
    )
    parser.add_argument("--candidate_path", type=str, default=config.rerank.prepare.candidate_path)
    parser.add_argument(
        "--force_include_positive",
        action=argparse.BooleanOptionalAction,
        default=config.rerank.prepare.force_include_positive,
        help="Force the positive molecule into the saved top-k list. Use only for training cache.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=config.rerank.prepare.limit,
        help="Debug mode: only keep the first N matched spectra.",
    )
    parser.add_argument(
        "--pre_top_k",
        "--topk",
        dest="pre_top_k",
        type=int,
        default=config.rerank.prepare.pre_top_k,
        help="Number of base-retrieved candidates kept for reranking.",
    )
    parser.add_argument(
        "--mol_norm_type",
        type=str,
        choices=["layernorm", "rmsnorm"],
        default=config.rerank.prepare.mol_norm_type or config.model.mol_encoder.norm_type,
        help="Normalization used in the molecule GINE encoder. Must match the alignment checkpoint.",
    )
    parser.add_argument(
        "--mol_norm_eps",
        type=float,
        default=config.rerank.prepare.mol_norm_eps or config.model.mol_encoder.norm_eps,
        help="Epsilon used by molecule encoder normalization. Must match the alignment checkpoint.",
    )
    args = parser.parse_args()

    if args.candidate_type is None:
        args.candidate_type = (
            "supplied"
            if args.dataset_type == "nplib1"
            else config.rerank.prepare.candidate_type
        )

    args.pre_top_k = int(args.pre_top_k)
    args.spec_batch_size = int(config.rerank.prepare.spec_batch_size)
    args.mol_batch_size = int(config.rerank.prepare.mol_batch_size)
    args.candidate_chunk_size = int(config.rerank.prepare.candidate_chunk_size)
    args.mol_embedding_storage = config.rerank.prepare.mol_embedding_storage
    args.mol_embedding_dtype = config.rerank.prepare.mol_embedding_dtype
    args.num_workers = int(config.rerank.prepare.num_workers)

    if not args.checkpoint:
        parser.error("--checkpoint is required unless rerank.prepare.checkpoint is set in params.yaml")
    if not args.save_path:
        parser.error("--save_path is required unless rerank.prepare.save_path is set in params.yaml")
    if args.mol_embedding_storage not in {"cpu", "cuda"}:
        parser.error("rerank.prepare.mol_embedding_storage must be either 'cpu' or 'cuda'")
    if args.mol_embedding_dtype not in {"float32", "float16"}:
        parser.error("rerank.prepare.mol_embedding_dtype must be either 'float32' or 'float16'")
    if args.mol_norm_type not in {"layernorm", "rmsnorm"}:
        parser.error("rerank.prepare.mol_norm_type must be either 'layernorm' or 'rmsnorm'")
    if args.pre_top_k <= 0:
        parser.error("--pre_top_k/--topk must be greater than 0")
    if args.limit < 0:
        parser.error("--limit/rerank.prepare.limit must be greater than or equal to 0")
    return args


def main():
    configure_runtime_cache()
    args = parse_args()

    if args.pre_top_k <= 0:
        raise ValueError("rerank.prepare.pre_top_k must be greater than 0")
    if args.candidate_chunk_size <= 0:
        raise ValueError("rerank.prepare.candidate_chunk_size must be greater than 0")
    if args.num_workers < 0:
        raise ValueError("rerank.prepare.num_workers must be greater than or equal to 0")

    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    setup_logging(save_path.parent / f"prepare_rerank_cache_{args.split}.log")
    startup_logging(args, "Prepare rerank cache")
    device = resolve_device(args.device)
    checkpoint_sha256 = sha256_file(args.checkpoint)

    model = load_align_model(
        checkpoint=args.checkpoint,
        device=device,
        mol_norm_type=args.mol_norm_type,
        mol_norm_eps=args.mol_norm_eps,
    )
    provider = get_provider(args.dataset_type, args.data_path)
    raw_data = provider.load_data(mode=args.split)
    candidate_source_path = (
        Path(args.candidate_path)
        if args.candidate_path
        else provider.data_dir / f"candidates_{args.candidate_type}.pkl"
    )
    candidate_sha256 = sha256_file(candidate_source_path)
    candidates_dict, candidate_label = load_candidates(provider, args.candidate_type, args.candidate_path)
    if not raw_data or not candidates_dict:
        raise RuntimeError("No data or candidates loaded.")

    tokenizer = Tokenizer(
        **TokenizerConfig(
            max_len=config.data.tokenizer.max_len,
            show_progress_bar=config.data.tokenizer.show_progress_bar,
        )
    )
    sequences = tokenizer.tokenize_sequence(raw_data)
    mapped_sequence_count = sum(
        candidates_dict.get(sequence["smiles"]) is not None for sequence in sequences
    )
    matched_sequences, unique_smiles, smiles_to_idx = build_unique_candidate_list(
        sequences,
        candidates_dict,
        limit=args.limit,
    )
    if not matched_sequences:
        raise RuntimeError("No spectra have candidate sets for rerank cache generation.")

    logging.info("Matched %s spectra with candidate sets.", len(matched_sequences))
    logging.info("Unique molecules before top-k pruning: %s", len(unique_smiles))

    mol_embs, valid_mol_mask = encode_molecules(model, unique_smiles, args, device)
    spec_embs, true_smiles_list = encode_spectra(model, matched_sequences, args, device)

    queries = []
    skipped = 0
    for spec_index, (spec_emb, true_smiles) in enumerate(
        tqdm(zip(spec_embs, true_smiles_list), total=len(true_smiles_list), desc="Build top-k", ascii=True)
    ):
        candidates = candidates_dict.get(true_smiles)
        record = build_query_record(
            spec_index=spec_index,
            spec_emb=spec_emb,
            true_smiles=true_smiles,
            candidates=candidates,
            smiles_to_idx=smiles_to_idx,
            mol_embs=mol_embs,
            valid_mol_mask=valid_mol_mask,
            args=args,
            device=device,
        )
        if record is None:
            skipped += 1
            continue
        queries.append(record)

    if not queries:
        raise RuntimeError("No valid rerank queries were generated.")

    pruned_mol_embs, pruned_smiles = prune_molecule_embeddings(queries, mol_embs, unique_smiles)
    spec_embs = spec_embs.contiguous()
    source_positives = sum(query["positive_in_source_candidates"] for query in queries)
    candidate_pool_positives = sum(query["positive_in_candidate_pool"] for query in queries)
    base_positives = sum(query["positive_in_base_topk"] for query in queries)
    labeled_queries = sum(query["label"] is not None for query in queries)
    upper_bound = base_positives / len(queries)
    labeled_fraction = labeled_queries / len(queries)
    source_coverage = source_positives / len(queries)
    candidate_pool_coverage = candidate_pool_positives / len(queries)
    logging.info("Generated %s valid queries; skipped=%s.", len(queries), skipped)
    logging.info(
        "Candidate mapping coverage: %.4f (%s/%s spectra)",
        mapped_sequence_count / max(len(sequences), 1),
        mapped_sequence_count,
        len(sequences),
    )
    logging.info("Supplied candidate positive coverage: %.4f", source_coverage)
    logging.info("Valid candidate-pool positive coverage: %.4f", candidate_pool_coverage)
    logging.info("Pre-top-%s recall upper bound: %.4f", args.pre_top_k, upper_bound)
    logging.info("Labeled query fraction in saved cache: %.4f", labeled_fraction)
    logging.info("Unique molecules after top-k pruning: %s", len(pruned_smiles))

    cache = {
        "spec_embs": spec_embs,
        "mol_embs": pruned_mol_embs,
        "mol_smiles": pruned_smiles,
        "queries": queries,
        "meta": {
            "dataset_type": args.dataset_type,
            "split": args.split,
            "candidate_label": candidate_label,
            "candidate_type": args.candidate_type,
            "candidate_path": args.candidate_path,
            "candidate_source_path": str(candidate_source_path.resolve()),
            "candidate_sha256": candidate_sha256,
            "pre_top_k": args.pre_top_k,
            "limit": args.limit,
            "force_include_positive": args.force_include_positive,
            "checkpoint": args.checkpoint,
            "checkpoint_sha256": checkpoint_sha256,
            "num_queries": len(queries),
            "num_input_spectra": len(sequences),
            "num_candidate_mapped_spectra": mapped_sequence_count,
            "num_selected_spectra": len(matched_sequences),
            "num_skipped_queries": skipped,
            "num_molecules": len(pruned_smiles),
            "candidate_mapping_coverage": mapped_sequence_count / max(len(sequences), 1),
            "source_candidate_coverage": source_coverage,
            "candidate_pool_coverage": candidate_pool_coverage,
            "upper_bound": upper_bound,
            "labeled_fraction": labeled_fraction,
        },
    }
    torch.save(cache, save_path)
    logging.info("Saved rerank cache to %s", save_path)


if __name__ == "__main__":
    main()
