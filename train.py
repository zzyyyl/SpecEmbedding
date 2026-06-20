import argparse
import logging
import os
import pickle
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from SpecEmbedding.config import config
from SpecEmbedding.data.datasets import TrainDataset
from SpecEmbedding.data.tokenizer import Tokenizer
from SpecEmbedding.loss import SupConLoss
from SpecEmbedding.trainer.fn import step_evaluate, step_train
from SpecEmbedding.trainer.trainer import Trainer, set_seed
from SpecEmbedding.type import (
    AugmentationConfig,
    DescriptionConfig,
    SchedulerConfig,
    StepFuncConfig,
    StorageConfig,
    TokenizerConfig,
    TrainerConfig,
)
from SpecEmbedding.utils.clean import get_classified_tokenset
from SpecEmbedding.utils.model import SiameseModel
from SpecEmbedding.utils.providers import load_train_val_data
from SpecEmbedding.utils.runtime import resolve_device, setup_logging, startup_logging


def add_base_argument(parser):
    parser.add_argument(
        "--dataset_type",
        type=str,
        choices=["massbank", "massspecgym", "nplib1", "gnps", "mona"],
        default=config.data.dataset_type,
        help="Dataset type"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=config.data.data_path,
        help="Base directory containing processed dataset folders"
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default=config.general.save_dir,
        help="Directory to save model and logs"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=config.general.device,
        help='Device to use, for example "cpu", "cuda", "cuda:0", or "cuda:1".'
    )

def load_data(dataset_type, data_path=None):
    train_raw, val_raw = load_train_val_data(dataset_type, data_path)

    if not train_raw:
        logging.error("No training data loaded. Check your data paths or internet connection.")
        return

    logging.info(f"Loaded {len(train_raw)} train records and {len(val_raw)} val records.")
    return train_raw, val_raw

def get_classified_data(dataset_type, data_path=None, cache_path=None):
    if cache_path is None:
        cache_path = config.data.cache_path
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
        tokenizer_config = TokenizerConfig(
            max_len=config.data.tokenizer.max_len, 
            show_progress_bar=config.data.tokenizer.show_progress_bar
        )
        tokenizer = Tokenizer(**tokenizer_config)

        logging.info("Tokenizing spectra sequences...")

        train_sequences = tokenizer.tokenize_sequence(train_raw)
        val_sequences = tokenizer.tokenize_sequence(val_raw)
        logging.info(f"Tokenized {len(train_sequences)} train sequences and {len(val_sequences)} val sequences.")

        # 3. Prepare Dataset
        # Get all unique SMILES to establish consistent labeling
        all_smiles = np.unique([s.get("smiles") for s in train_raw] + [s.get("smiles") for s in val_raw])
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
    parser.add_argument("--batch_size", type=int, default=config.train.pretrain.batch_size, help="Batch size")
    parser.add_argument("--epochs", type=int, default=config.train.pretrain.epochs, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=config.train.pretrain.lr, help="Learning rate")
    parser.add_argument("--resume", type=str, help="Path to checkpoint to resume training from")
    
    args = parser.parse_args()
    
    save_path = Path(args.save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    setup_logging(save_path / "train.log")
    startup_logging(args, "Start SpecEmbedding Pre-training")
    set_seed(config.general.seed)
    device = resolve_device(args.device)

    classified_data = get_classified_data(dataset_type=args.dataset_type, data_path=args.data_path)
    train_data = classified_data['train_data']
    train_keys = classified_data['train_keys']
    val_data = classified_data['val_data']
    val_keys = classified_data['val_keys']

    augment_config = AugmentationConfig(
        prob=config.augmentation.prob, 
        removal_max=config.augmentation.removal_max, 
        removal_intensity=config.augmentation.removal_intensity, 
        rate_intensity=config.augmentation.rate_intensity,
        node_drop_rate=config.augmentation.node_drop_rate,
        edge_mask_rate=config.augmentation.edge_mask_rate
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
        embedding_dim=config.model.spec_encoder.embedding_dim, 
        n_head=config.model.spec_encoder.n_head, 
        n_layer=config.model.spec_encoder.n_layer, 
        dim_feedward=config.model.spec_encoder.dim_feedward, 
        dim_target=config.model.spec_encoder.dim_target, 
        feedward_activation=config.model.spec_encoder.feedward_activation
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
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=config.train.pretrain.weight_decay)
    criterion = SupConLoss(
        device=device, 
        temperature=config.train.pretrain.loss.temperature, 
        base_temperature=config.train.pretrain.loss.base_temperature
    )
    
    n_step = args.epochs * len(train_dataloader)
    scheduler_config = SchedulerConfig(warmup_steps=int(n_step * 0.1), total_steps=n_step)
    trainer_config = TrainerConfig(
        n_epoch=args.epochs, 
        device=device, 
        early_stop=config.train.pretrain.early_stop, 
        show_progress_bar=True
    )
    
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
