import argparse
import os
import logging
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from SpecEmbedding.trainer.trainer_align import TrainerAlign
from SpecEmbedding.trainer.trainer import set_seed
from SpecEmbedding.models_align import SpecMolAlignModel, GINEEncoder
from SpecEmbedding.utils.model import SiameseModel
from SpecEmbedding.data.datasets_align import AlignGraphDataset, align_collate_fn

from train import (
    setup_logging,
    startup_logging,
    add_base_argument,
    get_classified_data
)

import random
import numpy as np

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def train_align(
    train_data: dict,
    train_keys: list,
    val_data: dict,
    val_keys: list,
    spec_encoder: SiameseModel,
    batch_size: int = 256,
    epochs_stage1: int = 20,
    epochs_stage2: int = 30,
    spec_dim: int = 512, # 预训练模型的输出维度
    mol_emb_dim: int = 128,
    mol_n_layers: int = 4,
    align_final_dim: int = 512,
    dropout_rate: float = 0.2,
    tau: float = 0.07,
    lr: float = 5e-5,
    device_name: str = "cuda" if torch.cuda.is_available() else "cpu",
    save_dir: str = "./checkpoints",
    seed: int = 42
):
    device = torch.device(device_name)

    logging.info("1. 初始化数据集与 DataLoader...")
    # 使用自定义的 AlignGraphDataset (继承自 TrainDataset)
    train_dataset = AlignGraphDataset(data=train_data, keys=train_keys, n_views=1, is_augment=True)
    val_dataset = AlignGraphDataset(data=val_data, keys=val_keys, n_views=1, is_augment=False)

    # 为了确保可复现性，设置 Generator 和 worker_init_fn
    g = torch.Generator()
    g.manual_seed(seed)

    # 必须使用 align_collate_fn 来组装 PyG 的 Graph Batch
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=align_collate_fn, 
        num_workers=4,
        worker_init_fn=seed_worker,
        generator=g
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=align_collate_fn, 
        num_workers=4,
        worker_init_fn=seed_worker,
        generator=g
    )

    logging.info("2. 初始化模型...")

    # 实例化新的分子图编码器
    mol_encoder = GINEEncoder(
        emb_dim=mol_emb_dim,
        n_layers=mol_n_layers,
        dropout_rate=dropout_rate,
    )

    # 实例化双塔对齐模型
    model = SpecMolAlignModel(
        spec_encoder=spec_encoder,
        mol_encoder=mol_encoder,
        spec_dim=spec_dim,
        hidden_dim=align_final_dim,
        final_dim=align_final_dim,
        dropout_rate=dropout_rate,
        tau=tau
    )
    trainer = TrainerAlign(
        model,
        train_loader,
        val_loader,
        device,
        save_dir=save_dir
    )

    # ==========================================
    # 阶段一：冻结质谱特征，仅训练分子编码器与投影头
    # ==========================================
    logging.info("\n" + "="*50)
    logging.info("Stage 1: Frozen MS Encoder, Train Mol Encoder & Projection Heads")
    logging.info("="*50)

    for param in model.spec_encoder.parameters():
        param.requires_grad = False

    stage1_params = [
        {'params': model.mol_encoder.parameters(), 'lr': lr * 10}, # 1e-3
        {'params': model.spec_proj.parameters(), 'lr': lr * 10},
        {'params': model.mol_proj.parameters(), 'lr': lr * 10},
        # {'params': [model.logit_scale], 'lr': lr * 10}
    ]

    optimizer1 = optim.AdamW(stage1_params, weight_decay=1e-4)
    trainer.fit(epochs=epochs_stage1, optimizer=optimizer1, stage_name="stage1", patience=5)

    # ==========================================
    # 阶段二：解冻所有参数，端到端微调
    # ==========================================
    logging.info("\n" + "="*50)
    logging.info("Stage 2: Unfreeze All, End-to-End Fine-tuning")
    logging.info("="*50)

    for param in model.spec_encoder.parameters():
        param.requires_grad = True

    stage2_params = [
        {'params': model.spec_encoder.parameters(), 'lr': lr / 10}, # 预训练模型用极小学习率
        {'params': model.mol_encoder.parameters(), 'lr': lr},
        {'params': model.spec_proj.parameters(), 'lr': lr},
        {'params': model.mol_proj.parameters(), 'lr': lr},
        # {'params': [model.logit_scale], 'lr': lr}
    ]

    optimizer2 = optim.AdamW(stage2_params, weight_decay=1e-4)
    scheduler2 = optim.lr_scheduler.CosineAnnealingLR(optimizer2, T_max=epochs_stage2)

    trainer.fit(epochs=epochs_stage2, optimizer=optimizer2, scheduler=scheduler2, stage_name="stage2", patience=5)

    logging.info("\nTwo-stage training completed successfully.")
    return model

def main():
    parser = argparse.ArgumentParser(description="Two-Stage Cross-Modal Alignment Training for SpecEmbedding")
    add_base_argument(parser)
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for alignment training")
    parser.add_argument("--epochs_stage1", type=int, default=20, help="Number of epochs for Stage 1 (Frozen MS Encoder)")
    parser.add_argument("--epochs_stage2", type=int, default=30, help="Number of epochs for Stage 2 (End-to-End Fine-tuning)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Base learning rate")
    parser.add_argument("--pretrained_spec", type=str, help="Path to your pre-trained SpecEmbedding model weights")

    args = parser.parse_args()

    save_path = Path(args.save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    setup_logging(save_path / "align_train.log")
    startup_logging(args)
    set_seed(args.seed)
    device = torch.device(args.device)

    classified_data = get_classified_data(dataset_type=args.dataset_type, data_path=args.data_path)
    train_data = classified_data['train_data']
    train_keys = classified_data['train_keys']
    val_data = classified_data['val_data']
    val_keys = classified_data['val_keys']

    # 检查预训练权重
    pretrained_spec_path = args.pretrained_spec
    if pretrained_spec_path and not os.path.exists(pretrained_spec_path):
        logging.warning(f"Pretrained SpecEmbedding not found at {pretrained_spec_path}.")
        logging.warning("Training will start from scratch (random initialization) for the MS Encoder.")
        pretrained_spec_path = None
    elif not pretrained_spec_path:
        logging.warning("No --pretrained_spec provided. MS Encoder will train from scratch.")

    # 实例化预训练的质谱编码器，参数需与之前的 train.py 参数完全一致
    spec_dim = 512
    spec_encoder = SiameseModel(
        embedding_dim=spec_dim,
        n_head=16,
        n_layer=4,
        dim_feedward=512,
        dim_target=512,
        feedward_activation="selu"
    )

    if pretrained_spec_path:
        logging.info(f"Loading pretrained SpecEmbedding from {pretrained_spec_path}")
        spec_encoder.load_state_dict(torch.load(pretrained_spec_path, map_location=device))

    # 启动两阶段对齐训练流水线
    logging.info("\nStarting the two-stage cross-modal alignment training pipeline...")

    final_model = train_align(
        train_data=train_data,
        train_keys=train_keys,
        val_data=val_data,
        val_keys=val_keys,
        spec_encoder=spec_encoder,
        batch_size=args.batch_size,
        epochs_stage1=args.epochs_stage1,
        epochs_stage2=args.epochs_stage2,
        spec_dim=spec_dim,
        mol_emb_dim=128,
        mol_n_layers=4,
        align_final_dim=512,
        dropout_rate=0.2,
        tau=0.07,
        lr=args.lr,
        device_name=device,
        save_dir=args.save_dir,
        seed=args.seed
    )

    logging.info("\nTraining complete! The final aligned model is returned and ready for evaluation/inference.")

    # 最终保存
    final_model_path = os.path.join(args.save_dir, "final_aligned_model.pth")
    torch.save(final_model.state_dict(), final_model_path)
    logging.info(f"Saved final end-to-end aligned model to {final_model_path}")

if __name__ == "__main__":
    main()
