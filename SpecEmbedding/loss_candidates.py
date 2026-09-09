"""Per-query contrastive supervision from naturally occurring training candidates."""

import torch
from torch.nn import functional as F


def candidate_alignment_loss(f_spec, f_positive, f_negative, negative_ptr, logit_scale):
    """Mean query CE: one positive against that query's distinct-identity negatives.

    ``negative_ptr`` delimits the flattened negative embeddings (CSR layout).
    The audited sampler must exclude *all* positive 2D aliases before encoding.
    Queries with no negatives contribute zero but remain in the mean denominator.
    This term can supplement the existing bidirectional in-batch loss; it does
    not change the model's cosine retrieval score or the evaluation candidate pool.
    """
    features = (f_spec, f_positive, f_negative)
    if (any(not torch.is_tensor(value) or value.ndim != 2 or not value.is_floating_point() for value in features)
            or f_spec.shape != f_positive.shape or f_spec.shape[0] == 0 or f_spec.shape[1] == 0
            or f_negative.shape[1] != f_spec.shape[1]
            or any(value.device != f_spec.device for value in features)):
        raise ValueError("Candidate loss requires compatible nonempty query/positive and flat negative embeddings")
    batch = f_spec.shape[0]
    if (not torch.is_tensor(negative_ptr) or negative_ptr.dtype != torch.long or negative_ptr.ndim != 1
            or negative_ptr.numel() != batch + 1 or negative_ptr[0] != 0
            or negative_ptr[-1] != f_negative.shape[0] or (negative_ptr[1:] < negative_ptr[:-1]).any()):
        raise ValueError("Candidate negative pointers must cover the flattened negatives in query order")
    dtype = torch.float64 if any(value.dtype == torch.float64 for value in features) else torch.float32
    scale = torch.as_tensor(logit_scale, device=f_spec.device, dtype=dtype)
    if scale.ndim != 0 or not torch.isfinite(scale) or scale <= 0:
        raise ValueError("Candidate logit scale must be a finite positive scalar")
    if any(not torch.isfinite(value).all() for value in features):
        raise ValueError("Non-finite candidate embeddings")
    spec, positive, negative = (F.normalize(value.to(dtype), dim=-1) for value in features)
    ptr = negative_ptr.to(f_spec.device)
    counts = ptr[1:] - ptr[:-1]
    owners = torch.repeat_interleave(torch.arange(batch, device=f_spec.device), counts)
    columns = torch.arange(len(negative), device=f_spec.device) - ptr[owners] + 1
    logits = f_spec.new_full((batch, int(counts.max()) + 1), -torch.inf, dtype=dtype)
    logits[:, 0] = (spec * positive).sum(dim=-1) * scale
    logits[owners, columns] = (spec[owners] * negative).sum(dim=-1) * scale
    return F.cross_entropy(logits, torch.zeros(batch, dtype=torch.long, device=f_spec.device))
