# -*- coding: utf-8 -*-
"""
Enhanced module to embed numeric and categorical metadata into a single vector.

Improvements over a basic MetadataEmbedding:
  1) Auto per-category embedding dim (categorical_embed_dim="auto")
  2) Optional unified normalization (numeric_block_norm=False -> only final LayerNorm)
  3) Stronger handling of numeric missingness (learnable missing embedding addition)
  4) Feature Dropout per categorical field (categorical_feature_dropout>0)

Defaults keep behavior close to prior versions:
  - numeric_block_norm: True (LayerNorm at each numeric MLP stage)
  - layer_norm: True (LayerNorm after concatenation)
  - numeric_missing_embedding: False (classic missing indicator concat only)
  - categorical_feature_dropout: 0.0 (disabled)

Example:
    embedder = MetadataEmbedding(
        numeric_dim=dataset.metadata_numeric_dim,
        categorical_cardinalities=[len(v) for v in dataset.metadata_info["categorical_vocab"].values()],
        categorical_embed_dim="auto",       # 1) auto by vocabulary size
        numeric_block_norm=False,           # 2) disable per-layer LN, keep only final LN
        layer_norm=True,
        numeric_use_missing_indicator=True, # classic missing indicator concat
        numeric_missing_embedding=True,     # 3) add learnable vector for missing pattern
        categorical_feature_dropout=0.1,    # 4) per-field Feature Dropout during training
        output_dim=128,
    )
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence
import math

import torch
import torch.nn as nn


# --------------------------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------------------------


def _make_activation(name: str) -> nn.Module:
    """Create activation module from a name"""
    name = name.lower()
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name == "gelu":
        return nn.GELU()
    if name == "silu":
        return nn.SiLU(inplace=True)
    if name == "tanh":
        return nn.Tanh()
    raise ValueError(f"Unknown activation specified: {name}")


def _auto_embed_dim(cardinality: int) -> int:
    """Heuristic to choose embedding dim from vocabulary size.

    Small vocab -> 8–16 dims; cap at 64 for large vocabularies.
    """
    if cardinality <= 1:
        return 8
    # Increase with a mild power; cap at 64, lower bound 8
    dim = int(round(1.6 * (cardinality**0.56)))
    return max(8, min(64, dim))


# --------------------------------------------------------------------------------------
# Main module
# --------------------------------------------------------------------------------------


class MetadataEmbedding(nn.Module):
    """Convert numeric/categorical metadata into an embedding vector (enhanced).

    Extra parameters:
        categorical_embed_dim: int | Sequence[int] | str = 32
            - legacy: int or Sequence[int]
            - new: "auto" to compute per-category dims from vocabulary size
        numeric_block_norm: bool = True
            - True: LayerNorm at each stage of numeric MLP (legacy behavior)
            - False: disable per-stage LN; keep only final LayerNorm after concatenation
        numeric_missing_embedding: bool = False
            - True: multiply missing mask (B, N) with a learnable table (N, D_num) and
                    add to numeric branch output (B, D_num)
        categorical_feature_dropout: float = 0.0
            - Training-only: drop entire categorical fields with prob p and scale by 1/(1-p)

    Other parameters remain backward compatible.
    """

    def __init__(
        self,
        numeric_dim: int,
        categorical_cardinalities: Optional[Sequence[int]] = None,
        *,
        numeric_hidden_dims: Sequence[int] = (128,),
        numeric_use_missing_indicator: bool = True,
        categorical_embed_dim: int | Sequence[int] | str = 32,  # or "auto"
        activation: str = "relu",
        dropout: float = 0.0,
        layer_norm: bool = True,
        numeric_block_norm: bool = True,  # unify toggle (only final LN when False)
        numeric_missing_embedding: bool = False,  # add learnable missing embedding
        categorical_feature_dropout: float = 0.0,  # per-category Feature Dropout
        output_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.numeric_dim = int(numeric_dim)
        self.numeric_use_missing_indicator = bool(numeric_use_missing_indicator)
        self.categorical_cardinalities = list(categorical_cardinalities or [])
        self.layer_norm_enabled = bool(layer_norm)
        self.numeric_block_norm = bool(numeric_block_norm)
        self.numeric_missing_embedding_enabled = bool(numeric_missing_embedding)
        self.categorical_feature_dropout = float(categorical_feature_dropout)
        if self.categorical_feature_dropout < 0.0 or self.categorical_feature_dropout >= 1.0:
            if self.categorical_feature_dropout != 0.0:
                raise ValueError("categorical_feature_dropout must be in [0.0, 1.0)")

        # --- numeric branch ---
        self.numeric_branch, self.numeric_out_dim = self._build_numeric_branch(
            hidden_dims=numeric_hidden_dims,
            activation=activation,
            dropout=dropout,
        )

        # Missing-value embeddings (match numeric branch output dim)
        if self.numeric_missing_embedding_enabled and self.numeric_out_dim > 0 and self.numeric_dim > 0:
            # Shape: (N_features, D_numeric_out)
            self.numeric_missing_table = nn.Parameter(torch.zeros(self.numeric_dim, self.numeric_out_dim))
            # Light init (small std)
            nn.init.normal_(self.numeric_missing_table, mean=0.0, std=(1.0 / math.sqrt(self.numeric_out_dim)))
        else:
            self.numeric_missing_table = None  # type: ignore[assignment]

        # --- categorical branch ---
        self._categorical_embeddings, self._cat_embed_dims, self.categorical_out_dim = (
            self._build_categorical_branch(categorical_embed_dim)
        )

        # --- fuse ---
        concat_dim = self.numeric_out_dim + self.categorical_out_dim
        self.layer_norm = nn.LayerNorm(concat_dim) if self.layer_norm_enabled and concat_dim > 0 else None
        self.proj = nn.Linear(concat_dim, output_dim) if output_dim is not None else None
        self.output_dim = output_dim or concat_dim

    # ---------------- numeric ----------------
    def _build_numeric_branch(
        self,
        hidden_dims: Sequence[int],
        activation: str,
        dropout: float,
    ) -> tuple[Optional[nn.Sequential], int]:
        """Build the numeric-feature MLP projector"""
        if self.numeric_dim <= 0:
            return None, 0

        layers: List[nn.Module] = []
        in_dim = self.numeric_dim * (2 if self.numeric_use_missing_indicator else 1)
        prev_dim = in_dim

        for hidden in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden))
            if self.numeric_block_norm:
                layers.append(nn.LayerNorm(hidden))
            layers.append(_make_activation(activation))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden

        if not layers:
            # No hidden layers -> identity mapping
            return nn.Sequential(nn.Identity()), prev_dim

        return nn.Sequential(*layers), prev_dim

    # ---------------- categorical ----------------
    def _build_categorical_branch(
        self,
        embed_dim: int | Sequence[int] | str,
    ) -> tuple[Optional[nn.ModuleList], List[int], int]:
        """Build Embedding layers for categorical features"""
        if not self.categorical_cardinalities:
            return None, [], 0

        if isinstance(embed_dim, str) and embed_dim.lower() == "auto":
            embed_dims = [_auto_embed_dim(c) for c in self.categorical_cardinalities]
        elif isinstance(embed_dim, Iterable) and not isinstance(embed_dim, (str, bytes)):
            embed_dims = list(embed_dim)
            if len(embed_dims) != len(self.categorical_cardinalities):
                raise ValueError("Length of categorical_embed_dim must match number of categories")
        else:
            embed_dims = [int(embed_dim)] * len(self.categorical_cardinalities)

        embeddings = nn.ModuleList()
        for size, dim in zip(self.categorical_cardinalities, embed_dims):
            if size <= 0:
                raise ValueError("Vocabulary size must be positive")
            embeddings.append(nn.Embedding(size, dim))

        out_dim = int(sum(embed_dims))
        return embeddings, embed_dims, out_dim

    # ---------------- forward ----------------
    def forward(
        self,
        numeric_values: Optional[torch.Tensor] = None,  # (B, N)
        numeric_missing: Optional[torch.Tensor] = None,  # (B, N)
        categorical_indices: Optional[torch.Tensor] = None,  # (B, C)
    ) -> torch.Tensor:
        """Generate metadata embedding during forward pass"""
        features: List[torch.Tensor] = []

        # Numeric branch
        if self.numeric_branch is not None:
            if numeric_values is None:
                raise ValueError("numeric_values is required when using numeric_branch")
            z = numeric_values.float()
            if self.numeric_use_missing_indicator:
                if numeric_missing is None:
                    raise ValueError("numeric_missing is required when using missing indicators")
                z = torch.cat([z, numeric_missing.float()], dim=-1)

            numeric_feat = self.numeric_branch(z)

            # Add learnable vector for missingness
            if self.numeric_missing_embedding_enabled:
                if numeric_missing is None:
                    raise ValueError("numeric_missing is required when numeric_missing_embedding=True")
                if self.numeric_missing_table is None:
                    raise RuntimeError("numeric_missing_table is not initialized")
                # (B, N) @ (N, D) -> (B, D)
                add_feat = numeric_missing.float() @ self.numeric_missing_table
                numeric_feat = numeric_feat + add_feat

            features.append(numeric_feat)

        # Categorical branch
        if self._categorical_embeddings is not None:
            if categorical_indices is None:
                raise ValueError("categorical_indices is required when using categorical embeddings")
            if categorical_indices.dim() == 1:
                categorical_indices = categorical_indices.unsqueeze(0)

            cat_feats: List[torch.Tensor] = []
            B = categorical_indices.size(0)
            p = self.categorical_feature_dropout
            keep_prob = 1.0 - p if p < 1.0 else 0.0
            scale = (1.0 / keep_prob) if (self.training and 0.0 < p < 1.0) else 1.0

            for idx, emb in zip(categorical_indices.split(1, dim=-1), self._categorical_embeddings):
                idx_flat = idx.squeeze(-1).long()  # (B,)
                e = emb(idx_flat)  # (B, D_j)

                # Feature Dropout per categorical field (train only)
                if self.training and p > 0.0:
                    # (B,1) keep mask. Bernoulli(keep_prob) per-sample field keep/drop
                    if keep_prob <= 0.0:
                        drop_mask = torch.zeros((B, 1), device=e.device, dtype=e.dtype)
                    else:
                        drop_mask = torch.bernoulli(torch.full((B, 1), keep_prob, device=e.device)).to(
                            e.dtype
                        )
                    e = e * drop_mask * scale  # scale by 1/keep_prob to keep expectation

                cat_feats.append(e)

            cat_concat = torch.cat(cat_feats, dim=-1)  # (B, sum D_j)
            features.append(cat_concat)

        if not features:
            raise ValueError("Neither numeric nor categorical branches are enabled")

        merged = torch.cat(features, dim=-1) if len(features) > 1 else features[0]

        if self.layer_norm is not None:
            merged = self.layer_norm(merged)
        if self.proj is not None:
            merged = self.proj(merged)

        return merged

    @property
    def categorical_embeddings(self) -> Optional[nn.ModuleList]:
        """Access categorical embedding layers"""
        if self._categorical_embeddings is None:
            return None
        return self._categorical_embeddings


# --------------------------------------------------------------------------------------
# Quick self-test
# --------------------------------------------------------------------------------------
if __name__ == "__main__":
    """Fetch metadata from AneurysmVesselSegDataset and sanity-check forward"""

    import argparse
    from pathlib import Path

    import torch

    from src.data.components.aneurysm_vessel_seg_dataset import AneurysmVesselSegDataset

    parser = argparse.ArgumentParser(description="MetadataEmbedding sanity check")
    parser.add_argument(
        "--vessel-pred-dir",
        type=Path,
        default=Path("data/nnUNet_inference/predictions_v4"),
        help="Directory of vessel segmentation predictions",
    )
    parser.add_argument(
        "--train-csv",
        type=Path,
        default=Path("data/train.csv"),
        help="Training labels CSV",
    )
    parser.add_argument(
        "--series-limit",
        type=int,
        default=4,
        help="Use only the first N series for a quick test",
    )
    parser.add_argument(
        "--output-dim",
        type=int,
        default=128,
        help="Output embedding dimension",
    )
    args = parser.parse_args()

    dataset = AneurysmVesselSegDataset(
        vessel_pred_dir=args.vessel_pred_dir,
        train_csv=args.train_csv,
        cache_data=False,
    )

    series_indices = list(range(min(len(dataset), max(1, args.series_limit))))
    numeric_dim = dataset.metadata_numeric_dim
    categorical_cardinalities = []
    metadata_info = dataset.metadata_info
    categorical_features = metadata_info.get("categorical_features", [])
    categorical_vocab = metadata_info.get("categorical_vocab", {})
    for feat in categorical_features:
        vocab = categorical_vocab.get(feat, {})
        categorical_cardinalities.append(len(vocab))

    print("Numeric dim:", numeric_dim)
    print("Categorical cardinalities:", categorical_cardinalities)

    model = MetadataEmbedding(
        numeric_dim=numeric_dim,
        categorical_cardinalities=categorical_cardinalities,
        categorical_embed_dim="auto",
        numeric_block_norm=False,  # unify: only final LN
        layer_norm=True,
        numeric_use_missing_indicator=True,
        numeric_missing_embedding=True,  # add missing embedding vector
        categorical_feature_dropout=0.2,  # per-category Dropout (train only)
        numeric_hidden_dims=(128,),
        output_dim=128,
    )
    model.train()

    numeric_vals = []
    numeric_missing = []
    categorical_vals = []
    for idx in series_indices:
        sample = dataset[idx]
        if "metadata_numeric" not in sample:
            continue
        numeric_vals.append(sample["metadata_numeric"].unsqueeze(0))
        numeric_missing.append(sample["metadata_numeric_missing"].unsqueeze(0))
        categorical_vals.append(
            sample.get(
                "metadata_categorical", torch.zeros(1, len(categorical_cardinalities), dtype=torch.int64)
            )
        )

    if not numeric_vals:
        raise RuntimeError("No sample with metadata was found")

    numeric_tensor = torch.cat(numeric_vals, dim=0)
    missing_tensor = torch.cat(numeric_missing, dim=0)
    if categorical_vals:
        categorical_tensor = torch.stack(categorical_vals, dim=0)
    else:
        categorical_tensor = None

    with torch.no_grad():
        embedding = model(
            numeric_values=numeric_tensor,
            numeric_missing=missing_tensor,
            categorical_indices=categorical_tensor,
        )

    print("Input numeric shape:", numeric_tensor.shape)
    if categorical_tensor is not None:
        print("Input categorical shape:", categorical_tensor.shape)
    print("Output embedding shape:", embedding.shape)
