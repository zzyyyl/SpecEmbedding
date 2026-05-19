import argparse
import logging
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pulp
import torch
from myopic_mces.myopic_mces import MCES
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Batch
from tqdm import tqdm

from SpecEmbedding.config import config
from SpecEmbedding.data.graph_utils import smiles_to_graph
from SpecEmbedding.data.tokenizer import Tokenizer
from SpecEmbedding.models import SiameseModel
from SpecEmbedding.models_align import GINEEncoder, SpecMolAlignModel
from SpecEmbedding.trainer.trainer import set_seed
from SpecEmbedding.type import TokenizerConfig
from src.data import (
    GNPSProvider,
    MassBankProvider,
    MassSpecGymProvider,
    MoNAProvider,
    NPLIB1Provider,
)
from train import (
    setup_logging,
    startup_logging,
)

mces_solvers = pulp.listSolvers(onlyAvailable=True)
mces_solver = "MOSEK" if "MOSEK" in mces_solvers else mces_solvers[0]


def get_dtype_size(dtype: torch.dtype) -> int:
    return torch.empty((), dtype=dtype).element_size()


def resolve_candidate_chunk_size(
    requested_chunk_size: int,
    memory_fraction: float,
    device: torch.device,
    embedding_dim: int,
    compute_dtype: torch.dtype,
    max_chunk_size: int,
) -> int:
    if requested_chunk_size > 0:
        return min(requested_chunk_size, max_chunk_size)

    if device.type != "cuda":
        return max_chunk_size

    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    dtype_size = get_dtype_size(compute_dtype)
    bytes_per_candidate = embedding_dim * dtype_size + dtype_size
    chunk_size = int(free_bytes * memory_fraction / bytes_per_candidate)
    chunk_size = max(1, min(chunk_size, max_chunk_size))

    logging.info(
        "Auto candidate chunk size: %s "
        "(free_cuda=%.2f GiB, total_cuda=%.2f GiB, memory_fraction=%.2f)",
        chunk_size,
        free_bytes / 1024**3,
        total_bytes / 1024**3,
        memory_fraction,
    )
    return chunk_size


def mces_worker(smiles_pair):
    s1, s2 = smiles_pair
    if s1 == s2:
        return 0.0, False
    try:
        retval = MCES(
            smiles1=s1,
            smiles2=s2,
            threshold=15,
            always_stronger_bound=True,
            solver=mces_solver,
            solver_options=dict(msg=0)
        )
        return retval[1], False
    except Exception:
        return 0.0, True # Return value and error flag

# ----------------- Spectra Dataset -----------------
class SpecDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences
        
    def __len__(self):
        return len(self.sequences)
        
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        return {
            'spec_mz': torch.tensor(seq['mz'], dtype=torch.float32),
            'spec_intensity': torch.tensor(seq['intensity'], dtype=torch.float32),
            'spec_mask': torch.tensor(seq['mask'], dtype=torch.bool),
            'smiles': seq['smiles']
        }

# ----------------- Mol Dataset -----------------
class MolDataset(Dataset):
    def __init__(self, smiles_list):
        self.smiles_list = smiles_list

    def __len__(self):
        return len(self.smiles_list)

    def __getitem__(self, idx):
        smiles = self.smiles_list[idx]
        return {
            'graph': smiles_to_graph(smiles),
            'original_idx': idx
        }

def mol_collate_fn(batch):
    # Filter out failed parsing
    batch = [b for b in batch if b['graph'] is not None]
    if not batch:
        return None
    graphs = Batch.from_data_list([b['graph'] for b in batch])
    indices = [b['original_idx'] for b in batch]
    return {'mol_graph': graphs, 'indices': indices}

def main():
    parser = argparse.ArgumentParser(description="Efficient Evaluate SpecMolAlignModel on Cross-Modal Retrieval.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to best aligned model checkpoint")
    parser.add_argument("--dataset_type", type=str, choices=["massbank", "massspecgym", "nplib1", "gnps", "mona"], default="massspecgym", help="Dataset type")
    parser.add_argument("--data_path", type=str, default=config.data.data_path, help="Base directory containing processed dataset folders")
    parser.add_argument("--candidate_type", type=str, choices=["mass", "formula"], default="mass", help="Candidate set type to use.")
    parser.add_argument("--mol_embedding_storage", type=str, choices=["cpu", "cuda"], default="cpu", help="Device used to store all molecule embeddings during retrieval.")
    parser.add_argument("--mol_embedding_dtype", type=str, choices=["float32", "float16"], default="float32", help="Dtype used to store all molecule embeddings.")
    parser.add_argument("--candidate_chunk_size", type=int, default=0, help="Maximum number of candidate embeddings moved to GPU at once. Use 0 to choose automatically from free GPU memory.")
    parser.add_argument("--candidate_chunk_memory_fraction", type=float, default=0.5, help="Fraction of free GPU memory used to estimate --candidate_chunk_size when it is 0.")
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
    device = torch.device(config.general.device if torch.cuda.is_available() else "cpu")

    logging.info("Initializing SpecMolAlignModel...")
    spec_encoder = SiameseModel(
        embedding_dim=config.model.spec_encoder.embedding_dim,
        n_head=config.model.spec_encoder.n_head,
        n_layer=config.model.spec_encoder.n_layer,
        dim_feedward=config.model.spec_encoder.dim_feedward,
        dim_target=config.model.spec_encoder.dim_target,
        feedward_activation=config.model.spec_encoder.feedward_activation
    )
    mol_encoder = GINEEncoder(
        emb_dim=config.model.mol_encoder.emb_dim,
        n_layers=config.model.mol_encoder.n_layers,
        dropout_rate=config.model.mol_encoder.dropout_rate,
        size_feature_dim=config.model.mol_encoder.size_feature_dim,
    )
    model = SpecMolAlignModel(
        spec_encoder=spec_encoder,
        mol_encoder=mol_encoder,
        spec_dim=config.model.spec_encoder.dim_target,
        hidden_dim=config.model.align.final_dim,
        final_dim=config.model.align.final_dim,
        dropout_rate=config.model.align.dropout_rate,
        tau=config.model.align.tau
    )
    state_dict = torch.load(args.checkpoint, map_location=device, weights_only=True)
    if "logit_scale" not in state_dict:
        logging.warning(
            "Checkpoint has no learnable logit_scale; initializing it from config.model.align.tau for backward compatibility."
        )
        state_dict["logit_scale"] = model.logit_scale.detach().clone()
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device)
    model.eval()

    if args.dataset_type == "massspecgym":
        provider = MassSpecGymProvider(data_dir=args.data_path)
    elif args.dataset_type == "massbank":
        provider = MassBankProvider(data_dir=args.data_path)
    elif args.dataset_type == "nplib1":
        provider = NPLIB1Provider(data_dir=args.data_path)
    elif args.dataset_type == "gnps":
        provider = GNPSProvider(data_dir=args.data_path)
    elif args.dataset_type == "mona":
        provider = MoNAProvider(data_dir=args.data_path)
    else:
        raise ValueError("--dataset_type is invalid")

    test_raw = provider.load_data(mode='test')
    if not test_raw:
        logging.error("No test data loaded.")
        return

    candidates_dict = provider.load_candidates(type=args.candidate_type)
    if not candidates_dict:
        logging.error("No candidate loaded.")
        return

    logging.info(f"Loaded {len(test_raw)} test spectra.")
    logging.info(f"Loaded {len(candidates_dict)} unique {args.candidate_type} candidate mapping keys.")

    tokenizer_config = TokenizerConfig(max_len=100, show_progress_bar=False)
    tokenizer = Tokenizer(**tokenizer_config)

    test_sequences = tokenizer.tokenize_sequence(test_raw)
    unique_test_smiles = set(s["smiles"] for s in test_sequences)

    # use unique_test_smiles to filter candidates_dict
    candidates_dict = {k: v for k, v in candidates_dict.items() if k in unique_test_smiles}
    logging.info(f"Filtered candidates_dict to {len(candidates_dict)} entries based on test dataset.")
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

    storage_device = torch.device("cuda" if args.mol_embedding_storage == "cuda" and device.type == "cuda" else "cpu")
    storage_dtype = torch.float16 if args.mol_embedding_dtype == "float16" else torch.float32
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
    plt.title(f'Cosine Similarity Distribution ({args.dataset_type}, {args.candidate_type})')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    plot_path = checkpoint_path.parent / f"similarity_dist_{args.dataset_type}_{args.candidate_type}.png"
    plt.savefig(plot_path)
    logging.info(f"Similarity distribution plot saved to {plot_path}")

    # Parallel MCES Calculation
    if not args.no_mces:
        mces_sum = 0.0
        mces_errors = 0
        if mces_pairs:
            logging.info(f"Computing MCES for {len(mces_pairs)} pairs using {mces_solver} solver...")
            # Use a fraction of CPUs to avoid overwhelming the system
            max_workers = min(os.cpu_count() or 1, 16)
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                mces_results = list(tqdm(
                    executor.map(mces_worker, mces_pairs), 
                    total=len(mces_pairs), 
                    desc="MCES Calc", 
                    ascii=True
                ))
            mces_sum = sum(res[0] for res in mces_results)
            mces_errors = sum(res[1] for res in mces_results)

        top1_mces = mces_sum / valid_queries
        logging.info(f"Top-1 MCES Similarity : {top1_mces:.4f}")
        
        if mces_errors > 0:
            logging.warning(f"MCES Calculation Errors: {mces_errors} out of {len(mces_pairs)}")
    else:
        logging.info("MCES Calculation skipped (--no-mces is set).")

    logging.info("="*40)

if __name__ == "__main__":
    main()
