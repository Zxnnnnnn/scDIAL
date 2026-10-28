"""Alternating marker selection and differentiable cell-distance matching."""
from __future__ import annotations

import numpy as np
import torch

from .markers import predict_clusters, score_markers

EPS = 1e-8


def distance_consistency_loss(zc, pairs, target_distances):
    """Only the marker target is detached; student mean normalization is live."""
    pairs = torch.as_tensor(pairs, dtype=torch.long, device=zc.device)
    target = torch.as_tensor(target_distances, dtype=zc.dtype, device=zc.device).detach()
    if pairs.ndim != 2 or pairs.shape[1] != 2 or target.shape != (pairs.shape[0],):
        raise ValueError("Pairs must be P x 2 and targets must have P entries.")
    if not pairs.shape[0] or bool(torch.any(pairs[:, 0] == pairs[:, 1])):
        raise ValueError("At least one non-self pair is required.")
    if not bool(torch.isfinite(target).all()) or float(target.mean()) <= EPS:
        raise ValueError("Marker distances must be finite with a positive mean.")
    distance = (zc[:, pairs[:, 0]] - zc[:, pairs[:, 1]]).square().mean(dim=0)
    student_normalized = distance / (distance.mean() + EPS)
    target_normalized = target / (target.mean() + EPS)
    return (student_normalized - target_normalized).square().mean()


class MarkerConsistency:
    """Owns only predicted partitions, expression and a dedicated pair RNG."""

    def __init__(self, expression, standardized_expression, gene_names, n_clusters,
                 seed, top_l=50, pair_count=4096, warmup=1, refresh_interval=5):
        self.expression = np.asarray(expression, dtype=np.float32)
        self.standardized = np.asarray(standardized_expression, dtype=np.float32)
        self.genes = list(gene_names)
        if self.expression.shape != self.standardized.shape:
            raise ValueError("Scoring and standardized expression must have identical axes.")
        if self.expression.shape[1] != len(self.genes):
            raise ValueError("Expression columns must match gene names.")
        self.n_clusters, self.seed = int(n_clusters), int(seed)
        self.top_l, self.pair_count = int(top_l), int(pair_count)
        self.warmup, self.refresh_interval = int(warmup), int(refresh_interval)
        if min(self.top_l, self.pair_count, self.refresh_interval) <= 0 or self.warmup < 0:
            raise ValueError("Marker counts/interval must be positive and warmup nonnegative.")
        self.pair_seed = self.seed + 104729
        self.rng = np.random.default_rng(self.pair_seed)
        self.cache = None
        self.refresh_history = []

    def should_refresh(self, epoch):
        return epoch > self.warmup and (epoch - self.warmup - 1) % self.refresh_interval == 0

    def refresh(self, outputs, epoch):
        predicted = predict_clusters(outputs["Zc"], self.n_clusters, self.seed)
        ranking = score_markers(self.expression, self.genes, outputs, predicted, self.top_l)
        nonempty = [v for v in ranking["marker_indices"].values() if len(v)]
        union = np.unique(np.concatenate(nonempty)) if nonempty else np.empty(0, dtype=np.int64)
        cache = {
            "version": len(self.refresh_history) + 1, "refresh_epoch": int(epoch),
            "status": "ready", "pair_rng_seed": self.pair_seed,
            "marker_indices_by_cluster": {str(k): v.tolist() for k, v in ranking["marker_indices"].items()},
            "marker_union_indices": union.tolist(), "predicted_labels": predicted.tolist(),
            "gene_index_space": "selected_genes", "pairs": [], "target_distances": [],
        }
        if not len(union):
            cache["status"] = "skipped_empty_marker_set"
        elif self.standardized.shape[0] < 2:
            cache["status"] = "skipped_insufficient_cells"
        else:
            n = self.standardized.shape[0]
            left = self.rng.integers(0, n, size=self.pair_count)
            right = (left + self.rng.integers(1, n, size=self.pair_count)) % n
            pairs = np.column_stack((left, right))
            marker_expr = self.standardized[:, union]
            distances = np.mean((marker_expr[left] - marker_expr[right]) ** 2, axis=1)
            if not np.all(np.isfinite(distances)) or float(distances.mean()) <= EPS:
                cache["status"] = "skipped_degenerate_marker_distances"
            else:
                cache["pairs"] = pairs.tolist()
                cache["target_distances"] = distances.tolist()
                cache["target_distance_mean"] = float(distances.mean())
        self.cache = cache
        self.refresh_history.append({k: v for k, v in cache.items()
                                     if k not in ("pairs", "target_distances", "predicted_labels")})
        return cache

    def loss(self, outputs, epoch):
        if epoch <= self.warmup:
            return outputs["Zc"].new_zeros(()), False
        if self.should_refresh(epoch):
            self.refresh(outputs, epoch)
        if self.cache is None or self.cache["status"] != "ready":
            return outputs["Zc"].new_zeros(()), False
        return distance_consistency_loss(outputs["Zc"], self.cache["pairs"],
                                         self.cache["target_distances"]), True
