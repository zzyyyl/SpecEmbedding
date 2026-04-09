import argparse
import os
import logging
from pathlib import Path

import torch
import pickle
import numpy as np
from torch.optim import AdamW
from torch.utils.data import DataLoader
from matchms import Spectrum

from SpecEmbedding.trainer.trainer import Trainer, set_seed
from SpecEmbedding.trainer.fn import step_train, step_evaluate
from SpecEmbedding.data.datasets import TrainDataset
from SpecEmbedding.data.tokenizer import Tokenizer
from SpecEmbedding.utils.clean import get_classified_tokenset
from SpecEmbedding.utils.model import SiameseModel
from SpecEmbedding.loss import SupConLoss
from SpecEmbedding.type import (
    AugmentationConfig, 
    DataLoaderConfig, 
    TrainerConfig, 
    SchedulerConfig, 
    StepFuncConfig, 
    DescriptionConfig, 
    StorageConfig, 
    TokenizerConfig
)

from data_provider import MSPProvider, MassSpecGymProvider

def setup_logging(log_file):
    logging.basicConfig(
        level=logging.INFO, 
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )

def startup_logging(args, message: str = "Start training"):
    logging.info("=" * 50)
    logging.info(message)
    logging.info("=" * 50)
    logging.info("Parsed Arguments:")
    for k, v in vars(args).items():
        logging.info(f"  {k}: {v}")

    device = torch.device(args.device)
    if device.type == 'cuda':
        logging.info(f"Using device: {device} ({torch.cuda.get_device_name(device)})")
    else:
        logging.info(f"Using device: {device}")

def add_base_argument(parser):
    parser.add_argument("--dataset_type", type=str, choices=["local", "massspecgym"], required=True, help="Dataset type")
    parser.add_argument("--data_path", type=str, help="Path to .msp file (required for local dataset)")
    parser.add_argument("--save_dir", type=str, default="./checkpoints", help="Directory to save model and logs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use")

def dict_to_spectrum(data_list):
    """Convert dictionary records from data_provider to matchms.Spectrum objects."""
    spectra = []
    for item in data_list:
        mz = [p[0] for p in item['peaks']]
        intensity = [p[1] for p in item['peaks']]
        spec = Spectrum(mz=np.array(mz), intensities=np.array(intensity), metadata={
            'smiles': item['smiles'],
            'precursor_mz': item.get('precursor_mz', 0.0)
        })
        spectra.append(spec)
    return spectra

def load_data(dataset_type, data_path):
    if dataset_type == "local":
        if not data_path:
            raise ValueError("--data_path is required for local dataset")
        provider = MSPProvider(data_path)
        train_raw = provider.load_data(mode='train')
        val_raw = provider.load_data(mode='val')
    else:
        provider = MassSpecGymProvider()
        train_raw = provider.load_data(mode='train')
        val_raw = provider.load_data(mode='val')

    if not train_raw:
        logging.error("No training data loaded. Check your data paths or internet connection.")
        return

    logging.info(f"Loaded {len(train_raw)} train records and {len(val_raw)} val records.")
    return train_raw, val_raw

def get_classified_data(dataset_type, data_path, cache_path="data/train_cache"):
    # Tokenize & Classify 处理 (带缓存逻辑)
    cache_dir = Path(cache_path)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"tokenset_{dataset_type}.pkl"

    if cache_file.exists():
        logging.info(f"Loading cached TokenSet from {cache_file}...")
        with open(cache_file, "rb") as f:
            classified_data = pickle.load(f)
    else:
        logging.info("No cache found. Processing dataset (Tokenize & Classify)...")
        # 1. Load data
        train_raw, val_raw = load_data(dataset_type=dataset_type, data_path=data_path)

        # 2. Convert to Spectrum objects and Tokenize
        tokenizer_config = TokenizerConfig(max_len=100, show_progress_bar=True)
        tokenizer = Tokenizer(**tokenizer_config)

        logging.info("Tokenizing spectra sequences...")
        train_spectra = dict_to_spectrum(train_raw)
        val_spectra = dict_to_spectrum(val_raw)

        train_sequences = tokenizer.tokenize_sequence(train_spectra)
        val_sequences = tokenizer.tokenize_sequence(val_spectra)
        logging.info(f"Tokenized {len(train_sequences)} train sequences and {len(val_sequences)} val sequences.")

        # 3. Prepare Dataset
        # Get all unique SMILES to establish consistent labeling
        all_smiles = np.unique([s['smiles'] for s in train_raw] + [s['smiles'] for s in val_raw])
        logging.info(f"Extracted {len(all_smiles)} unique SMILES across train and val sets.")
        
        train_data, train_keys = get_classified_tokenset(all_smiles, train_sequences)
        val_data, val_keys = get_classified_tokenset(all_smiles, val_sequences)

        classified_data = {
            'train_data': train_data,
            'train_keys': train_keys,
            'val_data': val_data,
            'val_keys': val_keys
        }
        logging.info(f"Saving TokenSet cache to {cache_file}...")
        with open(cache_file, "wb") as f:
            pickle.dump(classified_data, f)

    return classified_data

def main():
    parser = argparse.ArgumentParser(description="Train SpecEmbedding model on small molecule data.")
    add_base_argument(parser)
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=7.5e-5, help="Learning rate")
    parser.add_argument("--resume", type=str, help="Path to checkpoint to resume training from")
    
    args = parser.parse_args()
    
    save_path = Path(args.save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    setup_logging(save_path / "train.log")
    startup_logging(args, "Start SpecEmbedding Pre-training")
    set_seed(args.seed)
    device = torch.device(args.device)

    classified_data = get_classified_data(dataset_type=args.dataset_type, data_path=args.data_path)
    train_data = classified_data['train_data']
    train_keys = classified_data['train_keys']
    val_data = classified_data['val_data']
    val_keys = classified_data['val_keys']

    augment_config = AugmentationConfig(
        prob=0.5, 
        removal_max=0.2, 
        removal_intensity=0.3, 
        rate_intensity=0.15
    )
    
    train_dataset = TrainDataset(
        data=train_data, 
        keys=train_keys, 
        n_views=2, 
        is_augment=True, 
        augment_config=augment_config
    )
    val_dataset = TrainDataset(
        data=val_data, 
        keys=val_keys, 
        n_views=2
    )
    
    logging.info(f"Initialized TrainDataset with {len(train_keys)} unique SMILES (Total items: {len(train_dataset)}).")
    logging.info(f"Initialized ValDataset with {len(val_keys)} unique SMILES (Total items: {len(val_dataset)}).")

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # 4. Initialize Model
    model = SiameseModel(
        embedding_dim=512, 
        n_head=16, 
        n_layer=4, 
        dim_feedward=512, 
        dim_target=512, 
        feedward_activation="selu"
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Initialized SiameseModel with {total_params:,} trainable parameters.")

    if args.resume:
        if os.path.exists(args.resume):
            logging.info(f"Resuming training from {args.resume}")
            model.load_state_dict(torch.load(args.resume, map_location=device))
        else:
            logging.error(f"Checkpoint file {args.resume} not found. Starting from scratch.")
    
    # 5. Configuration
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)
    criterion = SupConLoss(device=device, temperature=0.05, base_temperature=0.05)
    
    n_step = args.epochs * len(train_dataloader)
    scheduler_config = SchedulerConfig(warmup_steps=int(n_step * 0.1), total_steps=n_step)
    trainer_config = TrainerConfig(n_epoch=args.epochs, device=device, early_stop=20, show_progress_bar=True)
    
    desc_config = DescriptionConfig(
        train="train, epoch={}, loss={:.4f}",
        val="validation, epoch={}, loss={:.4f}, save the model",
        end="model train end, best model loss={:.4f}"
    )
    stepfunc_config = StepFuncConfig(train=step_train, val=step_evaluate)
    
    storage_config = StorageConfig(
        model=save_path / "model.ckpt",
        lr=save_path / "lr.npy",
        step_loss=save_path / "step_loss.npy",
        loss=save_path / "epoch_loss.npy",
        custom=save_path / "custom_metric.npy"
    )

    # 6. Training
    logging.info(f"Starting training loop. Total epochs: {args.epochs}, steps per epoch: {len(train_dataloader)}, total steps: {n_step}")
    trainer = Trainer(
        model, "mean", train_dataloader, val_dataloader, optimizer, criterion,
        device, stepfunc_config, desc_config, storage_config, trainer_config, scheduler_config
    )
    trainer.train()

if __name__ == "__main__":
    main()
