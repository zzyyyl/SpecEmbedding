import argparse
import logging
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from SpecEmbedding.config import config
from SpecEmbedding.data.datasets_align import AlignGraphDataset, align_collate_fn
from SpecEmbedding.models_align import GINEEncoder, SpecMolAlignModel
from SpecEmbedding.trainer.trainer import set_seed
from SpecEmbedding.trainer.trainer_align import TrainerAlign
from SpecEmbedding.utils.model import SiameseModel
from train import add_base_argument, get_classified_data, setup_logging, startup_logging


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
    batch_size: int = config.train.align.batch_size,
    lr: float = config.train.align.lr,
    save_dir: str = config.general.save_dir,
    graph_cache_size: int = config.train.align.graph_cache_size,
):
    device = torch.device(config.general.device if torch.cuda.is_available() else "cpu")
    seed = config.general.seed
    epochs_stage1 = config.train.align.epochs_stage1
    epochs_stage2 = config.train.align.epochs_stage2

    logging.info("1. 初始化数据集与 DataLoader...")
    # 使用自定义的 AlignGraphDataset (继承自 TrainDataset)
    train_dataset = AlignGraphDataset(
        data=train_data,
        keys=train_keys,
        n_views=1,
        is_augment=True,
        graph_cache_size=graph_cache_size,
    )
    val_dataset = AlignGraphDataset(
        data=val_data,
        keys=val_keys,
        n_views=1,
        is_augment=False,
        graph_cache_size=graph_cache_size,
    )

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
        emb_dim=config.model.mol_encoder.emb_dim,
        n_layers=config.model.mol_encoder.n_layers,
        dropout_rate=config.model.mol_encoder.dropout_rate
    )

    has_pretrained_spec = spec_encoder is not None
    if not spec_encoder:
        spec_encoder = SiameseModel(
            embedding_dim=config.model.spec_encoder.embedding_dim,
            n_head=config.model.spec_encoder.n_head,
            n_layer=config.model.spec_encoder.n_layer,
            dim_feedward=config.model.spec_encoder.dim_feedward,
            dim_target=config.model.spec_encoder.dim_target,
            feedward_activation=config.model.spec_encoder.feedward_activation
        )

    # 实例化双塔对齐模型
    model = SpecMolAlignModel(
        spec_encoder=spec_encoder,
        mol_encoder=mol_encoder,
        spec_dim=config.model.spec_encoder.dim_target,
        hidden_dim=config.model.align.final_dim,
        final_dim=config.model.align.final_dim,
        dropout_rate=config.model.align.dropout_rate,
        tau=config.model.align.tau
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

    if not has_pretrained_spec:
        logging.info("No pretrained model, skipped.")
    else:
        for param in model.spec_encoder.parameters():
            param.requires_grad = False

        stage1_params = [
            {'params': model.mol_encoder.parameters(), 'lr': lr * 10}, # 1e-3
            {'params': model.spec_proj.parameters(), 'lr': lr * 10},
            {'params': model.mol_proj.parameters(), 'lr': lr * 10},
            {'params': [model.logit_scale], 'lr': lr * 10},
        ]

        optimizer1 = optim.AdamW(stage1_params, weight_decay=config.train.align.weight_decay)
        trainer.fit(epochs=epochs_stage1, optimizer=optimizer1, stage_name="stage1", patience=config.train.align.patience)

    # ==========================================
    # 阶段二：解冻所有参数，端到端微调
    # ==========================================
    logging.info("\n" + "="*50)
    logging.info("Stage 2: Unfreeze All, End-to-End Fine-tuning")
    logging.info("="*50)

    for param in model.spec_encoder.parameters():
        param.requires_grad = True

    spec_encoder_lr = lr / 10 if has_pretrained_spec else lr
    logging.info(f"Using spec_encoder lr={spec_encoder_lr:.2e} in stage2.")

    stage2_params = [
        {'params': model.spec_encoder.parameters(), 'lr': spec_encoder_lr},
        {'params': model.mol_encoder.parameters(), 'lr': lr},
        {'params': model.spec_proj.parameters(), 'lr': lr},
        {'params': model.mol_proj.parameters(), 'lr': lr},
        {'params': [model.logit_scale], 'lr': lr},
    ]

    optimizer2 = optim.AdamW(stage2_params, weight_decay=config.train.align.weight_decay)
    scheduler2 = optim.lr_scheduler.CosineAnnealingLR(optimizer2, T_max=epochs_stage2)

    trainer.fit(epochs=epochs_stage2, optimizer=optimizer2, scheduler=scheduler2, stage_name="stage2", patience=config.train.align.patience)

    logging.info("\nTwo-stage training completed successfully.")
    return model

def main():
    parser = argparse.ArgumentParser(description="Two-Stage Cross-Modal Alignment Training for SpecEmbedding")
    add_base_argument(parser)
    parser.add_argument("--batch_size", type=int, default=config.train.align.batch_size, help="Batch size for alignment training")
    parser.add_argument("--lr", type=float, default=config.train.align.lr, help="Base learning rate")
    parser.add_argument("--graph_cache_size", type=int, default=config.train.align.graph_cache_size, help="Lazy LRU molecule graph cache size per DataLoader worker. Use 0 to disable and -1 for unlimited.")
    parser.add_argument("--pretrained_spec", type=str, help="Path to your pre-trained SpecEmbedding model weights")

    args = parser.parse_args()

    save_path = Path(args.save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    setup_logging(save_path / "align_train.log")
    startup_logging(args)
    set_seed(config.general.seed)
    device = torch.device(config.general.device if torch.cuda.is_available() else "cpu")

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

    if pretrained_spec_path:
        logging.info(f"Loading pretrained SpecEmbedding from {pretrained_spec_path}")
        spec_encoder = SiameseModel(
            embedding_dim=config.model.spec_encoder.embedding_dim,
            n_head=config.model.spec_encoder.n_head,
            n_layer=config.model.spec_encoder.n_layer,
            dim_feedward=config.model.spec_encoder.dim_feedward,
            dim_target=config.model.spec_encoder.dim_target,
            feedward_activation=config.model.spec_encoder.feedward_activation
        )
        spec_encoder.load_state_dict(torch.load(pretrained_spec_path, map_location=device))
    else:
        spec_encoder = None

    # 启动两阶段对齐训练流水线
    logging.info("\nStarting the two-stage cross-modal alignment training pipeline...")

    final_model = train_align(
        train_data=train_data,
        train_keys=train_keys,
        val_data=val_data,
        val_keys=val_keys,
        spec_encoder=spec_encoder,
        batch_size=args.batch_size,
        lr=args.lr,
        save_dir=args.save_dir,
        graph_cache_size=args.graph_cache_size,
    )


    logging.info("\nTraining complete! The final aligned model is returned and ready for evaluation/inference.")

    # 最终保存
    final_model_path = os.path.join(args.save_dir, "final_aligned_model.pth")
    torch.save(final_model.state_dict(), final_model_path)
    logging.info(f"Saved final end-to-end aligned model to {final_model_path}")

if __name__ == "__main__":
    main()
