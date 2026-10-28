"""Shared, label-free marker ranking for refreshes and final export."""
from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from .fa_metrics import _spearman_matrix, compute_fcs

EPS = 1e-8
MARKER_COLUMNS = [
    "cluster", "gene", "gene_index", "rank", "score", "dominant_fa",
    "fa_gene_weight", "signed_correlation", "absolute_correlation", "fcs",
    "fcs_above_chance", "score_denominator", "dominant_contribution",
]


def as_numpy(value):
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def predict_clusters(zc, n_clusters: int, seed: int) -> np.ndarray:
    """Preserve the original marker pipeline's KMeans partition."""
    features = as_numpy(zc).T.astype(np.float32)
    if features.ndim != 2 or not np.all(np.isfinite(features)):
        raise ValueError("Marker KMeans requires a finite factors-by-cells embedding.")
    return KMeans(n_clusters=int(n_clusters), n_init=20, random_state=int(seed)).fit_predict(features)


def score_markers(
    expression: np.ndarray,
    gene_names: Sequence[str],
    snapshot: Mapping[str, object],
    pred_labels: Sequence[int],
    top_l: int = 50,
):
    """S[c,g] = sum_f W[f,g]*B[c,f]*abs(rho[f,g]) / sum_f W[f,g]*B[c,f].

    B is max(FCS - 0.5, 0). A zero denominator gives score zero.
    ``gene_index`` refers to the supplied, ordered selected-gene universe.
    This function has no true-label or reference-marker input.
    """
    expr = np.asarray(expression, dtype=np.float64)
    genes = [str(g) for g in gene_names]
    zf, hf = as_numpy(snapshot["Zf"]), as_numpy(snapshot["Hf"])
    labels = np.asarray(pred_labels)
    if expr.ndim != 2 or zf.ndim != 2 or hf.ndim != 2:
        raise ValueError("expression, Zf and Hf must be two-dimensional.")
    if expr.shape[1] != len(genes) or zf.shape != (hf.shape[0], len(genes)):
        raise ValueError("Gene names, expression columns and Zf columns must share exactly one order.")
    if hf.shape[1] != expr.shape[0] or labels.shape != (expr.shape[0],):
        raise ValueError("Expression, Hf and predicted labels must share the cell order.")
    if top_l <= 0:
        raise ValueError("top_l must be positive.")
    if not all(np.all(np.isfinite(a)) for a in (expr, zf, hf)):
        raise ValueError("Marker scoring requires finite expression and model outputs.")

    weights = np.abs(zf).astype(np.float64)
    weights = weights / (weights.sum(axis=1, keepdims=True) + EPS)
    signed = _spearman_matrix(expr, hf)
    # Undefined constant-vector correlations contribute no evidence to a score.
    absolute = np.where(np.isfinite(signed), np.abs(signed), 0.0)
    fcs_table, fcs = compute_fcs(hf, as_numpy(snapshot["gate_cf"]), labels)
    finite_fcs = np.where(np.isfinite(fcs), fcs, 0.0)
    above_chance = np.maximum(finite_fcs - 0.5, 0.0)
    denominator = above_chance @ weights
    numerator = above_chance @ (weights * absolute)
    scores = np.divide(numerator, denominator, out=np.zeros_like(numerator),
                       where=denominator > 0)
    clusters = np.unique(labels)
    rows, selected = [], {}
    indices = np.arange(len(genes))
    for ci, cluster in enumerate(clusters):
        order = np.lexsort((indices, -scores[ci]))
        order = order[np.isfinite(scores[ci, order]) & (scores[ci, order] > 0)][:top_l]
        selected[int(cluster)] = order.astype(np.int64)
        for rank, gi in enumerate(order, start=1):
            contribution = (above_chance[ci] * weights[:, gi] * absolute[:, gi]
                            / denominator[ci, gi])
            fa = int(np.argmax(contribution))
            rows.append({
                "cluster": int(cluster), "gene": genes[gi], "gene_index": int(gi),
                "rank": rank, "score": float(scores[ci, gi]), "dominant_fa": fa,
                "fa_gene_weight": float(weights[fa, gi]),
                "signed_correlation": float(signed[fa, gi]),
                "absolute_correlation": float(absolute[fa, gi]),
                "fcs": float(fcs[ci, fa]),
                "fcs_above_chance": float(above_chance[ci, fa]),
                "score_denominator": float(denominator[ci, gi]),
                "dominant_contribution": float(contribution[fa]),
            })
    return {
        "table": pd.DataFrame(rows, columns=MARKER_COLUMNS), "scores": scores,
        "clusters": clusters, "marker_indices": selected,
        "weights": weights, "signed_correlations": signed, "fcs": fcs,
        "fcs_above_chance": above_chance, "score_denominators": denominator,
        "fcs_table": fcs_table,
    }
