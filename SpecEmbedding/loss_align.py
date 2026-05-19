import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveAlignmentLoss(nn.Module):
    """Bidirectional multi-positive contrastive loss for spectrum-molecule alignment."""

    def __init__(self):
        super().__init__()

    def _build_positive_mask(self, labels, batch_size, device):
        if labels is None:
            return torch.eye(batch_size, dtype=torch.bool, device=device)

        if torch.is_tensor(labels):
            if labels.ndim != 1 or labels.size(0) != batch_size:
                raise ValueError("labels must be a 1D tensor with the same length as the batch")
            labels = labels.to(device)
            return labels[:, None] == labels[None, :]

        labels = list(labels)
        if len(labels) != batch_size:
            raise ValueError("labels must have the same length as the batch")
        return torch.tensor(
            [[left == right for right in labels] for left in labels],
            dtype=torch.bool,
            device=device,
        )

    def _multi_positive_loss(self, logits, positive_mask):
        log_probs = F.log_softmax(logits, dim=1)
        positive_mask = positive_mask.to(dtype=log_probs.dtype)
        positive_counts = positive_mask.sum(dim=1).clamp_min(1)
        return (-(log_probs * positive_mask).sum(dim=1) / positive_counts).mean()

    def forward(self, f_spec, f_mol, logit_scale, labels=None):
        batch_size = f_spec.size(0)
        device = f_spec.device

        f_spec = F.normalize(f_spec, dim=-1)
        f_mol = F.normalize(f_mol, dim=-1)
        logits = torch.matmul(f_spec, f_mol.T) * logit_scale

        positive_mask = self._build_positive_mask(labels, batch_size, device)
        loss_spec_to_mol = self._multi_positive_loss(logits, positive_mask)
        loss_mol_to_spec = self._multi_positive_loss(logits.T, positive_mask.T)

        return (loss_spec_to_mol + loss_mol_to_spec) / 2
