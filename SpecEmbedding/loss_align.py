import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveAlignmentLoss(nn.Module):
    """跨模态双向对比学习损失 (CLIP Loss)"""
    def __init__(self):
        super().__init__()

    def forward(self, f_spec, f_mol, logit_scale):
        batch_size = f_spec.size(0)
        device = f_spec.device

        # 归一化特征向量
        f_spec = F.normalize(f_spec, dim=-1)
        f_mol = F.normalize(f_mol, dim=-1)

        # 计算相似度矩阵 [batch, batch]
        # (batch, dim) @ (dim, batch) -> (batch, batch)
        logits = torch.matmul(f_spec, f_mol.T) * logit_scale
        
        # 对角线元素为正样本对
        labels = torch.arange(batch_size, device=device)
        
        # 计算两个方向的交叉熵损失
        loss_ms2mol = F.cross_entropy(logits, labels) # 质谱找分子
        loss_mol2ms = F.cross_entropy(logits.T, labels) # 分子找质谱
        
        return (loss_ms2mol + loss_mol2ms) / 2
