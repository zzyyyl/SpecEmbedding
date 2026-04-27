import argparse
import logging
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from SpecEmbedding.trainer.trainer import ModelTester
from SpecEmbedding.data.tokenizer import Tokenizer
from SpecEmbedding.utils.model import search, load_transformer_model, SiameseModel
from SpecEmbedding.config import config

from train import (
    setup_logging,
    startup_logging,
)

def main():
    parser = argparse.ArgumentParser(description="Evaluate SpecEmbedding model using MassSpecGym split logic.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint. If using a pre-configured architecture via --loss_type, this parameter is ignored.")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing the replicated .npy files (e.g. /data1/xp/data/massSpecGymData)")
    parser.add_argument("--loss_type", type=str, default="custom", choices=["custom", "TanimotoLoss", "SupConLoss", "SupConWithTanimotoLoss"], help="Type of model to load. 'custom' means load from --checkpoint directly using the default architecture.")

    args = parser.parse_args()

    # ---------------- Setup Logging ----------------
    checkpoint_path = Path(args.checkpoint)
    setup_logging(checkpoint_path.parent / "eval_massspecgym.log")
    startup_logging(args, "Starting MassSpecGym Evaluation (Replication Mode)")
    set_seed(config.general.seed)
    device = torch.device(config.general.device if torch.cuda.is_available() else "cpu")

    path_dir = Path(args.data_dir)
    replica_suffix = "-replication-{}.npy"

    # ---------------- Load Model ----------------
    if args.loss_type == "custom":
        logging.info(f"Loading custom model from {args.checkpoint}")
        model = SiameseModel(
            embedding_dim=config.model.spec_encoder.embedding_dim, 
            n_head=config.model.spec_encoder.n_head, 
            n_layer=config.model.spec_encoder.n_layer, 
            dim_feedward=config.model.spec_encoder.dim_feedward, 
            dim_target=config.model.spec_encoder.dim_target, 
            feedward_activation=config.model.spec_encoder.feedward_activation
        )
        state_dict = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(state_dict)
        model = model.to(device)
    else:
        logging.info(f"Loading pre-configured model type: {args.loss_type}")
        # When using pre-configured, is_augment is hardcoded as True per the notebook logic
        model = load_transformer_model(device, args.loss_type, is_augment=True)
    
    model.eval()
    
    show_progress_bar = True
    tester = ModelTester(model, device, show_progress_bar)
    tokenizer = Tokenizer(config.data.tokenizer.max_len, show_progress_bar)
    
    replica_df_seq = []
    
    # ---------------- Evaluation Loop ----------------
    replications = config.eval.replications
    for i in range(replications):
        logging.info(f"--- Starting Replication {i + 1}/{replications} ---")
        df_seq = []
        for instrument in ["Orbitrap", "QTOF", "all"]:
            logging.info("-" * 20 + f" {instrument} " + "-" * 20)
            
            query_path = path_dir.joinpath(f"{instrument}-query{replica_suffix.format(i + 1)}")
            ref_path = path_dir.joinpath(f"{instrument}-reference{replica_suffix.format(i + 1)}")
            
            if not query_path.exists() or not ref_path.exists():
                logging.warning(f"Files for replication {i+1}, instrument {instrument} not found. Skipping...")
                continue
                
            df = search(
                desc=instrument,
                tester=tester,
                k_metric=config.eval.top_k,
                tokenizer=tokenizer,
                query_path=query_path,
                ref_path=ref_path,
                loader_batch_size=config.eval.loader_batch_size,
                show_progress_bar=show_progress_bar,
                batch_size=config.eval.calc_batch_size
            )
            df_seq.append(df)
            
        if not df_seq:
            logging.error(f"No valid data found for replication {i+1}. Skipping.")
            continue
            
        # Combine instrument results for this replication
        combined_df = pd.concat(df_seq, axis=0)
        logging.info(f"\nResults for Replication {i+1}:\n{combined_df.to_string()}")
        replica_df_seq.append(combined_df)
        
    if not replica_df_seq:
        logging.error("No evaluations were completed. Please check your --data_dir and files.")
        return

    # ---------------- Calculate Mean & Std ----------------
    logging.info("="*50)
    logging.info("FINAL AGGREGATED RESULTS")
    logging.info("="*50)
    
    # Aggregate data into a 3D numpy array [Replications, Instruments, Metrics]
    data = []
    for item in replica_df_seq:
        data.append([item.values])
    data = np.concatenate(data, axis=0)
    
    # Calculate percentages
    mean_data = np.mean(data, axis=0) * 100
    std_data = np.std(data, axis=0) * 100
    
    instruments = replica_df_seq[0].index.tolist()
    columns = replica_df_seq[0].columns.tolist()
    
    # Create formatted strings
    for i, inst in enumerate(instruments):
        logging.info(f"Instrument: {inst}")
        for j, col in enumerate(columns):
            mean_val = mean_data[i, j]
            std_val = std_data[i, j]
            logging.info(f"  {col:<10}: {mean_val:.3f}% ± {std_val:.3f}%")
            
    logging.info("="*50)

if __name__ == "__main__":
    main()
