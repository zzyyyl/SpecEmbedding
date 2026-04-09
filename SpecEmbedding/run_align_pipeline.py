import torch
import torch.optim as optim
import logging
from torch.utils.data import DataLoader

from SpecEmbedding.models_align import SpecMolAlignModel, GINEEncoder
# 保持与 train.py 相同的模型导入路径
from SpecEmbedding.utils.model import SiameseModel
from SpecEmbedding.train_align import TrainerAlign
from SpecEmbedding.data.datasets_align import AlignGraphDataset, align_collate_fn

def run_two_stage_training(
    train_data: dict,      
    train_keys: list,      
    val_data: dict,
    val_keys: list,
    pretrained_spec_path: str = None, 
    batch_size: int = 256,
    epochs_stage1: int = 20,
    epochs_stage2: int = 30,
    lr: float = 5e-5,
    device_name: str = "cuda" if torch.cuda.is_available() else "cpu",
    save_dir: str = "./checkpoints"
):
    device = torch.device(device_name)
    
    logging.info("1. 初始化数据集与 DataLoader...")
    # 使用自定义的 AlignGraphDataset (继承自 TrainDataset)
    train_dataset = AlignGraphDataset(data=train_data, keys=train_keys, n_views=1, is_augment=True)
    val_dataset = AlignGraphDataset(data=val_data, keys=val_keys, n_views=1, is_augment=False)
    
    # 必须使用 align_collate_fn 来组装 PyG 的 Graph Batch
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=align_collate_fn, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=align_collate_fn, num_workers=4)
    
    logging.info("2. 初始化模型...")
    # 实例化预训练的质谱编码器，参数需与你之前的 train.py 参数完全一致！
    spec_encoder = SiameseModel(
        embedding_dim=512, 
        n_head=16, 
        n_layer=4, 
        dim_feedward=512, 
        dim_target=512, 
        feedward_activation="selu"
    )
    if pretrained_spec_path:
        logging.info(f"Loading pretrained SpecEmbedding from {pretrained_spec_path}")
        spec_encoder.load_state_dict(torch.load(pretrained_spec_path, map_location=device))
        
    # 实例化新的分子图编码器
    mol_encoder = GINEEncoder(emb_dim=256, n_layers=4)
    
    # 实例化双塔对齐模型
    model = SpecMolAlignModel(spec_encoder, mol_encoder, projection_dim=512)
    trainer = TrainerAlign(model, train_loader, val_loader, device, save_dir=save_dir)
    
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
        {'params': [model.logit_scale], 'lr': lr * 10}
    ]
    
    optimizer1 = optim.AdamW(stage1_params, weight_decay=1e-4)
    trainer.fit(epochs=epochs_stage1, optimizer=optimizer1, stage_name="Stage 1 (Frozen MS)", patience=5)
    
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
        {'params': [model.logit_scale], 'lr': lr}
    ]
    
    optimizer2 = optim.AdamW(stage2_params, weight_decay=1e-4)
    scheduler2 = optim.lr_scheduler.CosineAnnealingLR(optimizer2, T_max=epochs_stage2)
    
    trainer.fit(epochs=epochs_stage2, optimizer=optimizer2, scheduler=scheduler2, stage_name="Stage 2 (End-to-End)", patience=5)
    
    logging.info("\nTwo-stage training completed successfully.")
    return model
