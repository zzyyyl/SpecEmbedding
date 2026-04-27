import argparse
import logging
import os
import pickle
from pathlib import Path
from collections import defaultdict

import pulp
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Batch
from myopic_mces.myopic_mces import MCES
from concurrent.futures import ProcessPoolExecutor

from SpecEmbedding.data.tokenizer import Tokenizer
from SpecEmbedding.type import TokenizerConfig
from SpecEmbedding.models import SiameseModel
from SpecEmbedding.models_align import SpecMolAlignModel, GINEEncoder
from SpecEmbedding.data.graph_utils import smiles_to_graph
from SpecEmbedding.config import config

from src.data import MassSpecGymProvider, MassBankProvider, NPLIB1Provider
from train import (
    setup_logging,
    startup_logging,
    dict_to_spectrum
)

def mces_worker(smiles_pair):
    s1, s2 = smiles_pair
    solver_options=dict(msg=0)
    solvers = pulp.listSolvers(onlyAvailable=True)
    solver = "MOSEK" if "MOSEK" in solvers else solvers[0]
    try:
        # Re-initialize for each worker to be safe with pulp solvers
        retval = MCES(
            smiles1=s1,
            smiles2=s2,
            threshold=15,
            always_stronger_bound=True,
            solver=solver,
            solver_options=solver_options
        )
        return retval[1], False
    except Exception:
        return 0.0, True # Return value and error flag

# ----------------- Spectra Dataset -----------------
class EvalSpecDataset(Dataset):
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

# ----------------- Dynamic Mol Dataset -----------------
# Generate graphs on the fly to avoid 25GB+ memory explosions
class DynamicMolDataset(Dataset):
    def __init__(self, smiles_list):
        self.smiles_list = smiles_list
        
    def __len__(self):
        return len(self.smiles_list)
        
    def __getitem__(self, idx):
        s = self.smiles_list[idx]
        graph = smiles_to_graph(s)
        # If rdkit fails to parse, graph is None
        return {'graph': graph, 'smiles': s, 'original_idx': idx}

def mol_collate_fn(batch):
    # Filter out failed parsing
    batch = [b for b in batch if b['graph'] is not None]
    if not batch: return None
    graphs = Batch.from_data_list([b['graph'] for b in batch])
    smiles = [b['smiles'] for b in batch]
    indices = [b['original_idx'] for b in batch]
    return {'mol_graph': graphs, 'smiles': smiles, 'indices': indices}

def main():
    parser = argparse.ArgumentParser(description="Efficient Evaluate SpecMolAlignModel on Cross-Modal Retrieval.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to best aligned model checkpoint")
    parser.add_argument("--dataset_type", type=str, choices=["massbank", "massspecgym", "nplib1"], default="massspecgym", help="Dataset type")
    parser.add_argument("--data_path", type=str, help="Dataset directory (required for massbank/nplib1)")
    parser.add_argument("--no-mces", action="store_true", help="Disable MCES structural similaritycalculation")

    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    setup_logging(checkpoint_path.parent / "eval_align.log")
    startup_logging(args, "Start Cross-Modal Evaluation")
    device = torch.device(config.general.device if torch.cuda.is_available() else "cpu")

    # 1. Initialize Dual-Encoder Model
    logging.info("Initializing SpecMolAlignModel...")
    spec_encoder = SiameseModel(
        embedding_dim=512, 
        n_head=16, 
        n_layer=4, 
        dim_feedward=512, 
        dim_target=512, 
        feedward_activation="selu"
    )
    mol_encoder = GINEEncoder(emb_dim=128, n_layers=4, dropout_rate=0.2)
    model = SpecMolAlignModel(
        spec_encoder,
        mol_encoder,
        spec_dim=512,
        hidden_dim=512,
        final_dim=512,
        dropout_rate=0.2,
        tau=0.07
    )
    state_dict = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device)
    model.eval()

    # 2. Load Data and Candidates
    if args.dataset_type == "massspecgym":
        provider = MassSpecGymProvider()
        test_raw = provider.load_data(mode='test')
        candidates_dict = provider.load_candidates('mass') # Dictionary: {true_smiles: [cand1, cand2, ...]}
    elif args.dataset_type == "nplib1":
        provider = NPLIB1Provider(data_dir=args.data_path or "data")
        test_raw = provider.load_data(mode='test')
        candidates_dict = provider.load_candidates()
    else:
        if not args.data_path:
            raise ValueError("--data_path is required for massbank dataset")
        provider = MassBankProvider(data_dir=args.data_path)
        test_raw = provider.load_data(mode='test')
        candidates_dict = provider.load_candidates()

    if not test_raw:
        logging.error("No test data loaded.")
        return

    logging.info(f"Loaded {len(test_raw)} test spectra.")
    logging.info(f"Loaded {len(candidates_dict)} unique candidate mapping keys.")
    
    cand_sizes = [len(v) for v in candidates_dict.values()]
    if cand_sizes:
        avg_size = sum(cand_sizes) / len(cand_sizes)
        max_size = max(cand_sizes)
        min_size = min(cand_sizes)
        logging.info(f"Candidate set sizes: avg={avg_size:.2f}, max={max_size}, min={min_size}")

    # 3. Compile unique candidate SMILES
    unique_candidate_smiles = set()
    for s_list in candidates_dict.values():
        unique_candidate_smiles.update(s_list)
    unique_candidate_list = list(unique_candidate_smiles)
    num_unique_mols = len(unique_candidate_list)
    logging.info(f"Total unique candidate SMILES to embed: {num_unique_mols}")
    
    # Fast map: SMILES -> int Index
    smiles_to_idx = {s: i for i, s in enumerate(unique_candidate_list)}

    # 4. Generate Molecule Embeddings
    logging.info("Computing Molecule Embeddings dynamically...")
    mol_dataset = DynamicMolDataset(unique_candidate_list)
    mol_loader = DataLoader(mol_dataset, batch_size=config.eval.calc_batch_size, shuffle=False, collate_fn=mol_collate_fn, num_workers=4)
    
    # Pre-allocate a contiguous tensor to store embeddings for all unique candidate molecules
    global_mol_embs = torch.zeros((num_unique_mols, 512), dtype=torch.float32, device=device)

    with torch.no_grad():
        for batch in tqdm(mol_loader, desc="Mol Embeddings", ascii=True):
            if batch is None: continue
            
            mol_graph = batch['mol_graph'].to(device)
            indices = torch.tensor(batch['indices'], dtype=torch.long, device=device)
            
            # Forward pass to get molecule representations
            f_mol = model.mol_encoder(
                mol_graph.x,
                mol_graph.edge_index,
                mol_graph.edge_attr,
                mol_graph.batch,
            )
            f_mol = model.mol_proj(f_mol)
            f_mol = F.normalize(f_mol, dim=-1)
            
            # Populate the embedding matrix using batch indices
            global_mol_embs[indices] = f_mol

    # 5. Compute Spectra Embeddings and Evaluate
    tokenizer_config = TokenizerConfig(max_len=100, show_progress_bar=False)
    tokenizer = Tokenizer(**tokenizer_config)
    
    test_spectra = dict_to_spectrum(test_raw)
    test_sequences = tokenizer.tokenize_sequence(test_spectra)
    
    spec_dataset = EvalSpecDataset(test_sequences)
    spec_loader = DataLoader(spec_dataset, batch_size=config.eval.calc_batch_size, shuffle=False)
    
    hits = {k: 0 for k in config.eval.top_k}
    valid_queries = 0
    mrr_sum = 0.0
    mces_pairs = []

    logging.info("Computing Spectrum Embeddings and Evaluating...")
    with torch.no_grad():
        for batch in tqdm(spec_loader, desc="Evaluation", ascii=True):
            spec_mz = batch['spec_mz'].to(device)
            spec_intensity = batch['spec_intensity'].to(device)
            spec_mask = batch['spec_mask'].to(device)
            true_smiles_batch = batch['smiles']
            
            # Generate spectrum representations [Batch, 512]
            f_spec = model.spec_encoder(spec_mz, spec_intensity, spec_mask)
            f_spec = model.spec_proj(f_spec)
            f_spec = F.normalize(f_spec, dim=-1)
            
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
                
                # Select the embeddings for the current candidate set
                cand_indices = torch.tensor(cand_indices_list, dtype=torch.long, device=device)
                cand_embs = global_mol_embs[cand_indices] 
                query_emb = f_spec[i].unsqueeze(0) # [1, 512]
                
                # Perform efficient similarity search via matrix multiplication
                sims = torch.mm(query_emb, cand_embs.T).squeeze(0)
                
                # Sort candidates by similarity in descending order
                sorted_sims, sorted_rel_idx = torch.sort(sims, descending=True)
                
                # Identify the rank of the ground truth molecule
                true_idx_global = smiles_to_idx[true_smiles]
                sorted_global_indices = cand_indices[sorted_rel_idx]
                
                # Determine the index (rank) of the true SMILES in the sorted list
                rank_tensor = (sorted_global_indices == true_idx_global).nonzero(as_tuple=True)[0]
                if rank_tensor.numel() == 0:
                    continue
                rank = rank_tensor.item() + 1
                
                # Accumulate Top-K accuracy and MRR
                for k in config.eval.top_k:
                    if rank <= k:
                        hits[k] += 1
                mrr_sum += 1.0 / rank

                # Store Top-1 prediction for downstream structural similarity analysis
                top1_idx_global = sorted_global_indices[0].item()
                top1_smiles = unique_candidate_list[top1_idx_global]
                mces_pairs.append((top1_smiles, true_smiles))

                valid_queries += 1

    # 6. Output Results to log
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
        
    mrr = mrr_sum / valid_queries
    logging.info(f"Mean Reciprocal Rank (MRR): {mrr:.4f}")

    # 7. Parallel MCES Calculation
    if not args.no_mces:
        mces_sum = 0.0
        mces_errors = 0
        if mces_pairs:
            logging.info(f"Computing MCES for {len(mces_pairs)} pairs in parallel...")
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
