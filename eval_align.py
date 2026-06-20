import argparse
import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from SpecEmbedding.config import config
from SpecEmbedding.data.datasets_eval import MolDataset, SpecDataset, mol_collate_fn
from SpecEmbedding.data.tokenizer import Tokenizer
from SpecEmbedding.trainer.trainer import set_seed
from SpecEmbedding.type import TokenizerConfig
from SpecEmbedding.utils.align import (
    load_align_model,
    resolve_candidate_chunk_size,
    resolve_storage_dtype,
)
from SpecEmbedding.utils.mces import compute_mces
from SpecEmbedding.utils.providers import get_provider, load_candidates
from SpecEmbedding.utils.runtime import resolve_device, setup_logging, startup_logging


def main():
    parser = argparse.ArgumentParser(description="Efficient Evaluate SpecMolAlignModel on Cross-Modal Retrieval.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to best aligned model checkpoint")
    parser.add_argument("--dataset_type", type=str, choices=["massbank", "massspecgym", "nplib1", "gnps", "mona"], default="massspecgym", help="Dataset type")
    parser.add_argument("--data_path", type=str, default=config.data.data_path, help="Base directory containing processed dataset folders")
    parser.add_argument("--device", type=str, default=config.general.device, help='Device to use, for example "cpu", "cuda", "cuda:0", or "cuda:1".')
    parser.add_argument("--candidate_type", type=str, choices=["mass", "formula"], default="mass", help="Candidate set type to use.")
    parser.add_argument("--candidate_path", type=str, default=None, help="Path to a custom candidates pickle. Overrides --candidate_type when provided.")
    parser.add_argument("--mol_embedding_storage", type=str, choices=["cpu", "cuda"], default="cpu", help="Device used to store all molecule embeddings during retrieval.")
    parser.add_argument("--mol_embedding_dtype", type=str, choices=["float32", "float16"], default="float32", help="Dtype used to store all molecule embeddings.")
    parser.add_argument("--candidate_chunk_size", type=int, default=0, help="Maximum number of candidate embeddings moved to GPU at once. Use 0 to choose automatically from free GPU memory.")
    parser.add_argument("--candidate_chunk_memory_fraction", type=float, default=0.5, help="Fraction of free GPU memory used to estimate --candidate_chunk_size when it is 0.")
    parser.add_argument("--mol_norm_type", type=str, choices=["layernorm", "rmsnorm"], default=getattr(config.model.mol_encoder, "norm_type", "layernorm"), help="Normalization used in the molecule GINE encoder. Must match the checkpoint.")
    parser.add_argument("--mol_norm_eps", type=float, default=getattr(config.model.mol_encoder, "norm_eps", 1e-5), help="Epsilon used by molecule encoder normalization. Must match the checkpoint.")
    parser.add_argument("--no-mces", action="store_true", help="Disable MCES structural similaritycalculation")
    args = parser.parse_args()
    if args.candidate_chunk_size < 0:
        raise ValueError("--candidate_chunk_size must be greater than or equal to 0")
    if not 0 < args.candidate_chunk_memory_fraction <= 1:
        raise ValueError("--candidate_chunk_memory_fraction must be in the range (0, 1]")

    checkpoint_path = Path(args.checkpoint)
    setup_logging(checkpoint_path.parent / "eval_align.log")
    startup_logging(args, "Start Cross-Modal Evaluation")
    set_seed(config.general.seed)
    device = resolve_device(args.device)

    logging.info("Initializing SpecMolAlignModel...")
    model = load_align_model(
        checkpoint=args.checkpoint,
        device=device,
        mol_norm_type=args.mol_norm_type,
        mol_norm_eps=args.mol_norm_eps,
    )

    provider = get_provider(args.dataset_type, args.data_path)

    test_raw = provider.load_data(mode='test')
    if not test_raw:
        logging.error("No test data loaded.")
        return

    candidates_dict, candidate_label = load_candidates(
        provider,
        args.candidate_type,
        args.candidate_path,
    )
    if not candidates_dict:
        logging.error("No candidate loaded.")
        return

    logging.info(f"Loaded {len(test_raw)} test spectra.")
    logging.info(f"Loaded {len(candidates_dict)} unique {candidate_label} candidate mapping keys.")

    tokenizer_config = TokenizerConfig(max_len=100, show_progress_bar=False)
    tokenizer = Tokenizer(**tokenizer_config)

    test_sequences = tokenizer.tokenize_sequence(test_raw)
    unique_test_smiles = set(s["smiles"] for s in test_sequences)

    # use unique_test_smiles to filter candidates_dict
    candidates_dict = {k: v for k, v in candidates_dict.items() if k in unique_test_smiles}
    logging.info(f"Filtered candidates_dict to {len(candidates_dict)} entries based on test dataset.")
    if not candidates_dict:
        logging.error("No candidate entries match SMILES in the test dataset.")
        return
    cand_sizes = np.array([len(v) for v in candidates_dict.values()])
    logging.info(
        f"Candidate set sizes: "
        f"avg={cand_sizes.mean():.2f}, "
        f"max={cand_sizes.max()}, "
        f"min={cand_sizes.min()}"
    )
    del cand_sizes

    unique_candidate_list = list(unique_test_smiles.union(*candidates_dict.values()))
    num_unique_mols = len(unique_candidate_list)
    logging.info(f"Total unique candidate SMILES to embed: {num_unique_mols}")

    # Fast map: SMILES -> int Index
    smiles_to_idx = {s: i for i, s in enumerate(unique_candidate_list)}

    logging.info("Computing Molecule Embeddings...")
    mol_dataset = MolDataset(unique_candidate_list)
    mol_loader = DataLoader(
        mol_dataset,
        batch_size=config.eval.calc_batch_size,
        shuffle=False,
        collate_fn=mol_collate_fn,
        num_workers=4
    )

    storage_device = device if args.mol_embedding_storage == "cuda" and device.type == "cuda" else torch.device("cpu")
    storage_dtype = resolve_storage_dtype(args.mol_embedding_dtype)
    logging.info(
        f"Storing molecule embeddings on {storage_device} with dtype={storage_dtype}. "
        "Use --mol_embedding_storage cuda only when the full candidate matrix fits in GPU memory."
    )

    # Pre-allocate a contiguous tensor to store embeddings for all unique candidate molecules.
    # Keep it on CPU by default to avoid allocating tens of GB on GPU for large candidate sets.
    global_mol_embs = torch.zeros(
        (num_unique_mols, config.model.align.final_dim),
        dtype=storage_dtype,
        device=storage_device
    )

    with torch.no_grad():
        for batch in tqdm(mol_loader, desc="Mol Embeddings", ascii=True):
            if batch is None:
                continue
            
            mol_graph = batch['mol_graph'].to(device)
            indices = torch.tensor(batch['indices'], dtype=torch.long, device=storage_device)

            # Forward pass to get molecule representations
            f_mol = model.encode_mol(mol_graph, normalize=True)
            
            # Populate the embedding matrix using batch indices
            global_mol_embs[indices] = f_mol.to(device=storage_device, dtype=storage_dtype)

    candidate_chunk_size = resolve_candidate_chunk_size(
        requested_chunk_size=args.candidate_chunk_size,
        memory_fraction=args.candidate_chunk_memory_fraction,
        device=device,
        embedding_dim=config.model.align.final_dim,
        compute_dtype=torch.float32,
        max_chunk_size=num_unique_mols,
    )
    logging.info(f"Using candidate_chunk_size={candidate_chunk_size}")

    spec_dataset = SpecDataset(test_sequences)
    spec_loader = DataLoader(spec_dataset, batch_size=config.eval.calc_batch_size, shuffle=False)
    
    hits = {k: 0 for k in config.eval.top_k}
    random_hits = {k: 0.0 for k in config.eval.top_k}
    valid_queries = 0
    mrr_sum = 0.0
    mces_pairs = []

    # Initialize histograms for similarity distributions
    sim_bins = np.arange(-1.0, 1.01, 0.01)
    target_sim_counts = np.zeros(len(sim_bins) - 1, dtype=np.int64)
    candidate_sim_counts = np.zeros(len(sim_bins) - 1, dtype=np.int64)

    logging.info("Computing Spectrum Embeddings and Evaluating...")
    with torch.no_grad():
        for batch in tqdm(spec_loader, desc="Evaluation", ascii=True):
            spec_mz = batch['spec_mz'].to(device)
            spec_intensity = batch['spec_intensity'].to(device)
            spec_mask = batch['spec_mask'].to(device)
            true_smiles_batch = batch['smiles']
            
            # Generate spectrum representations [Batch, 512]
            f_spec = model.encode_spec(spec_mz, spec_intensity, spec_mask, normalize=True)
            
            for i in range(len(true_smiles_batch)):
                true_smiles = true_smiles_batch[i]
                cands = candidates_dict.get(true_smiles)
                
                if not cands:
                    continue
                    
                cand_set = set(cands)
                cand_set.add(true_smiles)
                
                # Resolve unique SMILES to their integer indices in the embedding matrix
                cand_indices_list = [smiles_to_idx[c] for c in cand_set if c in smiles_to_idx]
                
                if not cand_indices_list or smiles_to_idx.get(true_smiles) not in cand_indices_list:
                    continue 

                candidate_size = len(cand_indices_list)
                
                query_emb = f_spec[i].unsqueeze(0) # [1, 512]
                true_idx_global = smiles_to_idx[true_smiles]
                target_sim = None
                top1_sim = -float("inf")
                top1_idx_global = None
                
                for start in range(0, len(cand_indices_list), candidate_chunk_size):
                    chunk_indices_list = cand_indices_list[start:start + candidate_chunk_size]
                    cand_indices = torch.tensor(chunk_indices_list, dtype=torch.long, device=storage_device)
                    cand_embs = global_mol_embs[cand_indices].to(device=device, dtype=f_spec.dtype)
                    sims = torch.mm(query_emb, cand_embs.T).squeeze(0)
                    sims_np = sims.cpu().numpy()

                    chunk_top_sim, chunk_top_rel_idx = torch.max(sims, dim=0)
                    chunk_top_sim_value = chunk_top_sim.item()
                    if chunk_top_sim_value > top1_sim:
                        top1_sim = chunk_top_sim_value
                        top1_idx_global = chunk_indices_list[chunk_top_rel_idx.item()]

                    if true_idx_global in chunk_indices_list:
                        true_rel_idx = chunk_indices_list.index(true_idx_global)
                        target_sim = sims_np[true_rel_idx]
                        cand_sims = np.delete(sims_np, true_rel_idx)
                    else:
                        cand_sims = sims_np

                    c_counts, _ = np.histogram(cand_sims, bins=sim_bins)
                    candidate_sim_counts += c_counts

                if target_sim is None or top1_idx_global is None:
                    continue

                # Ranking is descending, so every candidate with a higher similarity precedes the target.
                candidate_rank_offset = 0
                for start in range(0, len(cand_indices_list), candidate_chunk_size):
                    chunk_indices_list = cand_indices_list[start:start + candidate_chunk_size]
                    cand_indices = torch.tensor(chunk_indices_list, dtype=torch.long, device=storage_device)
                    cand_embs = global_mol_embs[cand_indices].to(device=device, dtype=f_spec.dtype)
                    sims = torch.mm(query_emb, cand_embs.T).squeeze(0)
                    sims_np = sims.cpu().numpy()
                    if true_idx_global in chunk_indices_list:
                        true_rel_idx = chunk_indices_list.index(true_idx_global)
                        sims_np = np.delete(sims_np, true_rel_idx)
                    candidate_rank_offset += int(np.sum(sims_np > target_sim))
                
                # Update target histogram
                t_idx = np.digitize(target_sim, sim_bins) - 1
                if 0 <= t_idx < len(target_sim_counts):
                    target_sim_counts[t_idx] += 1

                rank = candidate_rank_offset + 1
                
                # Accumulate Top-K accuracy and MRR
                for k in config.eval.top_k:
                    if rank <= k:
                        hits[k] += 1
                    random_hits[k] += min(k, candidate_size) / candidate_size
                mrr_sum += 1.0 / rank

                # Store Top-1 prediction for downstream structural similarity analysis
                top1_smiles = unique_candidate_list[top1_idx_global]
                mces_pairs.append((top1_smiles, true_smiles))

                valid_queries += 1

    if valid_queries == 0:
        logging.error("No valid queries to evaluate (e.g., all candidates failed to parse).")
        return

    logging.info("="*40)
    logging.info("      EFFICIENT CROSS-MODAL EVALUATION RESULTS")
    logging.info(f"      Total Valid Queries: {valid_queries}")
    logging.info("="*40)
    
    for k in sorted(config.eval.top_k):
        acc = hits[k] / valid_queries
        logging.info(f"Top-{k:<2} Accuracy : {acc:.4%}  ({hits[k]}/{valid_queries})")
        random_acc = random_hits[k] / valid_queries
        logging.info(f"Random Top-{k:<2} Baseline : {random_acc:.4%}")
        
    mrr = mrr_sum / valid_queries
    logging.info(f"Mean Reciprocal Rank (MRR): {mrr:.4f}")

    # Plot Similarity Distribution
    logging.info("Plotting similarity distribution...")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))
    bin_centers = (sim_bins[:-1] + sim_bins[1:]) / 2
    
    target_sim_percent = target_sim_counts / target_sim_counts.sum()
    candidate_sim_percent = candidate_sim_counts / candidate_sim_counts.sum()

    plt.bar(bin_centers, target_sim_percent, width=0.01, alpha=0.5, label='Targets', color='blue')
    plt.bar(bin_centers, candidate_sim_percent, width=0.01, alpha=0.3, label='Candidates', color='gray')

    mean_target_sim = (bin_centers * target_sim_percent).sum()
    mean_candidate_sim = (bin_centers * candidate_sim_percent).sum()
    plt.axvline(mean_target_sim, color='red', linestyle='--', label=f'M.Target Sim={mean_target_sim:.2f}')
    plt.axvline(mean_candidate_sim, color='green', linestyle='--', label=f'M.Candidate Sim={mean_candidate_sim:.2f}')

    # Apply Gaussian smoothing for better visualization
    from scipy.ndimage import gaussian_filter1d
    target_smooth = gaussian_filter1d(target_sim_percent, sigma=1)
    candidate_smooth = gaussian_filter1d(candidate_sim_percent, sigma=1)
    plt.plot(bin_centers, target_smooth, color='blue', linestyle='-', label='Target Density')
    plt.plot(bin_centers, candidate_smooth, color='gray', linestyle='-', label='Candidate Density')

    plt.xlabel('Cosine Similarity')
    plt.ylabel('Percentage of Occurrence')
    plt.title(f'Cosine Similarity Distribution ({args.dataset_type}, {candidate_label})')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    plot_path = checkpoint_path.parent / f"similarity_dist_{args.dataset_type}_{candidate_label}.png"
    plt.savefig(plot_path)
    logging.info(f"Similarity distribution plot saved to {plot_path}")

    # Parallel MCES Calculation
    if not args.no_mces:
        top1_mces = compute_mces(mces_pairs, "MCES Calc")
        if top1_mces is not None:
            logging.info(f"Top-1 MCES Similarity : {top1_mces:.4f}")
    else:
        logging.info("MCES Calculation skipped (--no-mces is set).")

    logging.info("="*40)

if __name__ == "__main__":
    main()
