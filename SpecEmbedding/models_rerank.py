import torch
import torch.nn as nn
import torch.nn.functional as F


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


class RelativeCandidateReranker(nn.Module):
    """Spectrum-conditioned relative candidate reranker.

    The model keeps an absolute, pointwise compatibility branch and adds a
    permutation-equivariant relation branch.  For every ordered candidate
    pair, the relation score is constructed as ``g(r_ij) - g(r_ji)``; this
    makes the pair preference antisymmetric by construction while allowing
    the query spectrum to condition both the preference and its aggregation
    weight.  ``base_ranks`` is accepted by the public forward signature for
    cache compatibility but is intentionally ignored.
    """

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        rank_emb_dim: int = 1,
        max_rank: int = 1,
        n_layers: int = 0,
        n_heads: int = 1,
        dropout: float = 0.1,
        alpha_init: float = 1.0,
        use_base_score_feature: bool = True,
        use_residual_score: bool = True,
        use_rank_embedding: bool = False,
        use_product_feature: bool = True,
        use_abs_diff_feature: bool = True,
        relation_dim: int | None = None,
        pair_chunk_size: int = 32,
        relation_top_k: int = 40,
        use_relative_module: bool = True,
        use_spectrum_conditioning: bool = True,
        use_molecular_relation: bool = True,
        use_spectrum_features: bool = True,
        use_molecule_features: bool = True,
        use_antisymmetric: bool = True,
        pair_mode: str | None = None,
    ):
        super().__init__()
        del rank_emb_dim, max_rank, n_layers, n_heads, use_rank_embedding
        if embedding_dim <= 0 or hidden_dim <= 0:
            raise ValueError("embedding_dim and hidden_dim must be greater than 0")
        if pair_chunk_size <= 0:
            raise ValueError("pair_chunk_size must be greater than 0")
        if relation_top_k <= 0:
            raise ValueError("relation_top_k must be greater than 0")
        if pair_mode is None:
            pair_mode = "antisymmetric" if use_antisymmetric else "directed"
        if pair_mode not in {"antisymmetric", "directed"}:
            raise ValueError("pair_mode must be either 'antisymmetric' or 'directed'")

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.use_base_score_feature = use_base_score_feature
        self.use_residual_score = use_residual_score
        self.use_product_feature = use_product_feature
        self.use_abs_diff_feature = use_abs_diff_feature
        self.use_relative_module = use_relative_module
        self.use_spectrum_conditioning = use_spectrum_conditioning
        self.use_molecular_relation = use_molecular_relation
        self.use_spectrum_features = use_spectrum_features
        self.use_molecule_features = use_molecule_features
        self.pair_mode = pair_mode
        # Keep the legacy attribute for callers and old analysis code.
        self.use_antisymmetric = pair_mode == "antisymmetric"
        self.pair_chunk_size = pair_chunk_size
        self.relation_top_k = relation_top_k
        self.relation_dim = relation_dim or max(32, min(128, hidden_dim // 2))

        absolute_dim = embedding_dim * 2
        if use_product_feature:
            absolute_dim += embedding_dim
        if use_abs_diff_feature:
            absolute_dim += embedding_dim
        if use_base_score_feature:
            absolute_dim += 1
        self.absolute_mlp = nn.Sequential(
            nn.Linear(absolute_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.absolute_score = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, max(hidden_dim // 2, 1)),
            nn.GELU(),
            nn.Linear(max(hidden_dim // 2, 1), 1),
        )

        self.spec_relation = nn.Linear(embedding_dim, self.relation_dim)
        self.mol_relation = nn.Linear(embedding_dim, self.relation_dim)
        relation_input_dim = (
            (self.relation_dim if use_spectrum_conditioning else 0)
            + (self.relation_dim * 2 if use_molecular_relation else 0)
            + (1 if use_base_score_feature else 0)
            + (1 if use_molecular_relation else 0)
        )
        if relation_input_dim <= 0:
            raise ValueError("relative relation branch needs at least one relation feature")
        self.preference_mlp = nn.Sequential(
            nn.Linear(relation_input_dim, self.relation_dim),
            nn.LayerNorm(self.relation_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.relation_dim, self.relation_dim),
            nn.GELU(),
            nn.Linear(self.relation_dim, 1),
        )
        weight_input_dim = (
            (self.relation_dim if use_spectrum_conditioning else 0)
            + (self.relation_dim if use_molecular_relation else 0)
            + (1 if use_molecular_relation else 0)
        )
        # Preserve the original checkpoint shape for the default spectrum+
        # molecular relation path.  A base-score-only ablation still needs a
        # scalar weight input when both richer relation sources are disabled.
        if weight_input_dim == 0 and use_base_score_feature:
            weight_input_dim = 1
        if weight_input_dim <= 0:
            raise ValueError("relative weight branch needs at least one relation feature")
        self.weight_mlp = nn.Sequential(
            nn.Linear(weight_input_dim, self.relation_dim),
            nn.GELU(),
            nn.Linear(self.relation_dim, 1),
        )
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init), dtype=torch.float32))
        self.beta = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))

    def _absolute_features(self, spec_emb, candidate_embs, base_scores):
        spec_expand = spec_emb.unsqueeze(1).expand_as(candidate_embs)
        zero_spec = torch.zeros_like(spec_expand)
        zero_molecule = torch.zeros_like(candidate_embs)
        features = [
            spec_expand if self.use_spectrum_features else zero_spec,
            candidate_embs if self.use_molecule_features else zero_molecule,
        ]
        if self.use_product_feature:
            features.append(
                spec_expand * candidate_embs
                if self.use_spectrum_features and self.use_molecule_features
                else zero_spec
            )
        if self.use_abs_diff_feature:
            features.append(
                torch.abs(spec_expand - candidate_embs)
                if self.use_spectrum_features and self.use_molecule_features
                else zero_spec
            )
        if self.use_base_score_feature:
            features.append(base_scores.unsqueeze(-1))
        return torch.cat(features, dim=-1)

    def _relation_features(self, spec_rel, mol_i, mol_j, base_i, base_j):
        delta = mol_i - mol_j
        abs_delta = torch.abs(delta)
        cosine = F.cosine_similarity(mol_i, mol_j, dim=-1).unsqueeze(-1)
        parts = []
        if self.use_spectrum_conditioning:
            parts.append(spec_rel)
        if self.use_molecular_relation:
            parts.extend([delta, abs_delta])
        base_delta = base_i - base_j
        if base_delta.ndim == cosine.ndim - 1:
            base_delta = base_delta.unsqueeze(-1)
        if self.use_base_score_feature:
            parts.append(base_delta)
        if self.use_molecular_relation:
            parts.append(cosine)
        return torch.cat(parts, dim=-1)

    def _relative_scores(self, spec_emb, candidate_embs, base_scores, candidate_mask):
        batch_size, num_candidates, _ = candidate_embs.shape
        spec_rel = self.spec_relation(spec_emb)
        mol_rel = self.mol_relation(candidate_embs)
        relative = torch.zeros(
            (batch_size, num_candidates),
            dtype=candidate_embs.dtype,
            device=candidate_embs.device,
        )
        for start in range(0, num_candidates, self.pair_chunk_size):
            stop = min(start + self.pair_chunk_size, num_candidates)
            mol_i = mol_rel[:, start:stop].unsqueeze(2)
            mol_j = mol_rel.unsqueeze(1)
            base_i = base_scores[:, start:stop].unsqueeze(2)
            base_j = base_scores.unsqueeze(1)
            spec_pair = spec_rel[:, None, None, :].expand(-1, stop - start, num_candidates, -1)
            pair_features = self._relation_features(spec_pair, mol_i, mol_j, base_i, base_j)
            reverse_features = self._relation_features(spec_pair, mol_j, mol_i, base_j, base_i)
            preference = self.preference_mlp(pair_features).squeeze(-1)
            reverse_preference = self.preference_mlp(reverse_features).squeeze(-1)
            if self.pair_mode == "antisymmetric":
                pair_preference = preference - reverse_preference
            else:
                pair_preference = preference

            weight_parts = []
            if self.use_spectrum_conditioning:
                weight_parts.append(spec_pair)
            if self.use_molecular_relation:
                weight_parts.append(mol_i - mol_j)
                weight_parts.append(
                    F.cosine_similarity(mol_i, mol_j, dim=-1).unsqueeze(-1)
                )
            if self.use_base_score_feature:
                if not self.use_spectrum_conditioning and not self.use_molecular_relation:
                    base_delta = base_i - base_j
                    if base_delta.ndim == spec_pair.ndim - 1:
                        base_delta = base_delta.unsqueeze(-1)
                    weight_parts.append(base_delta)
            weight_logits = self.weight_mlp(torch.cat(weight_parts, dim=-1)).squeeze(-1)
            valid = candidate_mask[:, start:stop].unsqueeze(-1) & candidate_mask.unsqueeze(1)
            eye = torch.zeros(
                (stop - start, num_candidates),
                dtype=torch.bool,
                device=candidate_embs.device,
            )
            local = torch.arange(stop - start, device=candidate_embs.device)
            global_indices = torch.arange(start, stop, device=candidate_embs.device)
            eye[local, global_indices] = True
            valid = valid & ~eye.unsqueeze(0)
            weights = F.softplus(weight_logits) * valid.to(weight_logits.dtype)
            weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
            contribution = (weights * pair_preference * valid.to(pair_preference.dtype)).sum(dim=-1)
            relative[:, start:stop] = contribution
        return relative

    def forward(
        self,
        spec_emb,
        candidate_embs,
        base_scores,
        base_ranks=None,
        candidate_mask=None,
    ):
        del base_ranks
        if candidate_mask is None:
            candidate_mask = torch.ones(
                candidate_embs.shape[:2], dtype=torch.bool, device=candidate_embs.device
            )
        absolute = self.absolute_score(self.absolute_mlp(
            self._absolute_features(spec_emb, candidate_embs, base_scores)
        )).squeeze(-1)
        coarse_scores = absolute
        if self.use_residual_score:
            coarse_scores = coarse_scores + self.alpha * base_scores
        scores = coarse_scores
        if self.use_relative_module:
            relation_k = min(self.relation_top_k, candidate_embs.shape[1])
            _, relation_indices = torch.topk(
                coarse_scores.masked_fill(~candidate_mask, torch.finfo(coarse_scores.dtype).min),
                k=relation_k,
                dim=-1,
                largest=True,
                sorted=False,
            )
            gather_emb_indices = relation_indices.unsqueeze(-1).expand(-1, -1, candidate_embs.shape[-1])
            relation_candidates = torch.gather(candidate_embs, 1, gather_emb_indices)
            relation_base_scores = torch.gather(base_scores, 1, relation_indices)
            relation_mask = torch.gather(candidate_mask, 1, relation_indices)
            relation = self._relative_scores(
                spec_emb,
                relation_candidates,
                relation_base_scores,
                relation_mask,
            )
            relative_full = torch.zeros_like(scores)
            relative_full.scatter_(1, relation_indices, relation)
            scores = scores + self.beta * relative_full
        return scores.masked_fill(~candidate_mask, torch.finfo(scores.dtype).min)
