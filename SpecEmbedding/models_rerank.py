import torch
import torch.nn as nn


class CandidateReranker(nn.Module):
    """Listwise reranker for hard molecule candidates retrieved by SpecEmbedding."""

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        rank_emb_dim: int,
        max_rank: int,
        n_layers: int,
        n_heads: int,
        dropout: float,
        alpha_init: float,
        use_base_score_feature: bool = True,
        use_residual_score: bool = True,
        use_rank_embedding: bool = True,
        use_product_feature: bool = True,
        use_abs_diff_feature: bool = True,
    ):
        super().__init__()
        if max_rank <= 0:
            raise ValueError("max_rank must be greater than 0")
        if n_layers < 0:
            raise ValueError("n_layers must be greater than or equal to 0")

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.rank_emb_dim = rank_emb_dim
        self.max_rank = max_rank
        self.n_layers = n_layers
        self.use_base_score_feature = use_base_score_feature
        self.use_residual_score = use_residual_score
        self.use_rank_embedding = use_rank_embedding
        self.use_product_feature = use_product_feature
        self.use_abs_diff_feature = use_abs_diff_feature

        self.rank_embedding = nn.Embedding(max_rank + 1, rank_emb_dim, padding_idx=0)
        pair_dim = embedding_dim * 4 + 1 + rank_emb_dim
        self.pair_mlp = nn.Sequential(
            nn.Linear(pair_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        if n_layers > 0:
            layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=n_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.context_encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        else:
            self.context_encoder = None

        self.score_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init), dtype=torch.float32))

    def _build_pair_features(self, spec_emb, candidate_embs, base_scores, base_ranks):
        batch_size, num_candidates, _ = candidate_embs.shape
        spec_expand = spec_emb.unsqueeze(1).expand(batch_size, num_candidates, -1)
        rank_ids = base_ranks.clamp(min=0, max=self.max_rank).long()
        rank_emb = self.rank_embedding(rank_ids)
        product = spec_expand * candidate_embs
        abs_diff = torch.abs(spec_expand - candidate_embs)
        base_score_feature = base_scores.unsqueeze(-1)

        # Ablated blocks are zeroed instead of removed so every feature ablation
        # keeps the same MLP width and parameter count as the full model.
        features = [
            spec_expand,
            candidate_embs,
            product if self.use_product_feature else torch.zeros_like(product),
            abs_diff if self.use_abs_diff_feature else torch.zeros_like(abs_diff),
            base_score_feature
            if self.use_base_score_feature
            else torch.zeros_like(base_score_feature),
            rank_emb if self.use_rank_embedding else torch.zeros_like(rank_emb),
        ]
        return torch.cat(features, dim=-1)

    def forward(self, spec_emb, candidate_embs, base_scores, base_ranks, candidate_mask=None):
        pair_features = self._build_pair_features(
            spec_emb=spec_emb,
            candidate_embs=candidate_embs,
            base_scores=base_scores,
            base_ranks=base_ranks,
        )
        hidden = self.pair_mlp(pair_features)

        if self.context_encoder is not None:
            padding_mask = None
            if candidate_mask is not None:
                padding_mask = ~candidate_mask
            hidden = self.context_encoder(hidden, src_key_padding_mask=padding_mask)

        delta = self.score_head(hidden).squeeze(-1)
        scores = self.alpha * base_scores + delta if self.use_residual_score else delta
        if candidate_mask is not None:
            scores = scores.masked_fill(~candidate_mask, torch.finfo(scores.dtype).min)
        return scores


class PointwiseReranker(CandidateReranker):
    """Context-free reranker used as an ablation baseline."""

    def __init__(self, *args, **kwargs):
        kwargs["n_layers"] = 0
        super().__init__(*args, **kwargs)
