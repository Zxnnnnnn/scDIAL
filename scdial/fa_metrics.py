"""Metrics for scDIAL supplementary experiments 2--4.

The functions in this module operate on numpy arrays and ordinary Python
containers.  They deliberately do not import the training stack, so they can
be tested with synthetic artifacts and reused by the experiment runner.

Conventions
-----------
``expression`` is cells x genes, ``Zf`` is genes x factors, ``Hf`` is
factors x cells, and ``gate_cf`` is either a length-n-cell vector or a
1 x n-cell array.  Gene indices are zero based.  Missing/constant vectors in
Spearman or AUROC calculations produce ``nan`` rather than raising.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import roc_auc_score


ArrayLike = Union[np.ndarray, Sequence[float]]


def _as_float_matrix(value: ArrayLike, name: str, ndim: int = 2) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}-dimensional, got shape {arr.shape}.")
    if not np.all(np.isfinite(arr)):
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return arr


def _validate_top_k(top_k: Sequence[int], n_genes: int) -> Tuple[int, ...]:
    values = tuple(sorted({int(k) for k in top_k}))
    if not values or any(k <= 0 for k in values):
        raise ValueError("top_k must contain positive integers.")
    if any(k > n_genes for k in values):
        raise ValueError(f"top_k values cannot exceed the number of genes ({n_genes}).")
    return values


def _validate_gene_names(gene_names: Optional[Sequence[str]], n_genes: int) -> List[str]:
    if gene_names is None:
        return [str(i) for i in range(n_genes)]
    names = [str(item) for item in gene_names]
    if len(names) != n_genes:
        raise ValueError("gene_names must have one entry per gene.")
    return names


def rank_genes_by_fa(
    zf: ArrayLike,
    top_k: Sequence[int],
    gene_names: Optional[Sequence[str]] = None,
) -> Dict[int, Dict[int, List[Union[int, str]]]]:
    """Rank genes for every FA using ``abs(Zf)``.

    Returns ``result[k][factor]`` containing the first *k* gene indices (or
    names when ``gene_names`` is provided).  Ties are resolved by the original
    gene order, making results deterministic across runs.
    """
    zf_arr = _as_float_matrix(zf, "zf")
    n_genes, n_factors = zf_arr.shape
    ks = _validate_top_k(top_k, n_genes)
    names = _validate_gene_names(gene_names, n_genes) if gene_names is not None else None
    score = np.abs(zf_arr)
    result: Dict[int, Dict[int, List[Union[int, str]]]] = {k: {} for k in ks}
    for factor in range(n_factors):
        # lexsort uses the last key as primary; ``-score`` gives descending
        # score while the index key provides a stable tie break.
        order = np.lexsort((np.arange(n_genes), -score[:, factor]))
        for k in ks:
            selected = order[:k].tolist()
            result[k][factor] = [names[i] for i in selected] if names is not None else selected
    return result


def top_gene_indices_by_fa(zf: ArrayLike, top_k: Sequence[int]) -> Dict[int, Dict[int, np.ndarray]]:
    """Return the top-k gene indices for each FA (convenience wrapper)."""
    zf_arr = _as_float_matrix(zf, "zf")
    n_genes, n_factors = zf_arr.shape
    ks = _validate_top_k(top_k, n_genes)
    score = np.abs(zf_arr)
    result: Dict[int, Dict[int, np.ndarray]] = {k: {} for k in ks}
    for factor in range(n_factors):
        order = np.lexsort((np.arange(n_genes), -score[:, factor]))
        for k in ks:
            result[k][factor] = order[:k].astype(np.int64, copy=True)
    return result


def _safe_abs_spearman(x: np.ndarray, y: np.ndarray) -> float:
    if x.size != y.size or x.size < 2:
        return np.nan
    if np.all(x == x[0]) or np.all(y == y[0]):
        return np.nan
    with np.errstate(all="ignore"):
        result = spearmanr(x, y)
        value = getattr(result, "statistic", getattr(result, "correlation", result[0]))
    return float(abs(value)) if np.isfinite(value) else np.nan


def _safe_signed_spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Return a signed Spearman correlation, or ``nan`` for constant input."""
    if x.size != y.size or x.size < 2:
        return np.nan
    if np.all(x == x[0]) or np.all(y == y[0]):
        return np.nan
    with np.errstate(all="ignore"):
        result = spearmanr(x, y)
        value = getattr(result, "statistic", getattr(result, "correlation", result[0]))
    return float(value) if np.isfinite(value) else np.nan


def _spearman_matrix(expression: np.ndarray, activations: np.ndarray) -> np.ndarray:
    """Compute signed Spearman correlations in one BLAS-backed pass.

    The matched baseline evaluates the same gene/factor pairs repeatedly.  A
    rank transform followed by centered matrix multiplication is numerically
    equivalent to calling ``spearmanr`` for each pair and avoids a Python loop
    over hundreds of thousands of 18k-cell vectors.
    """
    expr = np.asarray(expression, dtype=np.float64)
    acts = np.asarray(activations, dtype=np.float64)
    if expr.ndim != 2 or acts.ndim != 2 or expr.shape[0] != acts.shape[1]:
        raise ValueError("expression and activations must be (cells, genes) and (factors, cells).")
    expr_rank = rankdata(expr, axis=0).astype(np.float64, copy=False)
    act_rank = rankdata(acts, axis=1).astype(np.float64, copy=False)
    expr_rank -= expr_rank.mean(axis=0, keepdims=True)
    act_rank -= act_rank.mean(axis=1, keepdims=True)
    expr_norm = np.sqrt(np.sum(expr_rank * expr_rank, axis=0))
    act_norm = np.sqrt(np.sum(act_rank * act_rank, axis=1))
    denominator = act_norm[:, None] * expr_norm[None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        correlations = (act_rank @ expr_rank) / denominator
    correlations[~np.isfinite(correlations)] = np.nan
    return correlations


def _quantile_bins(values: np.ndarray, n_bins: int) -> np.ndarray:
    """Assign values to quantile bins, handling ties and constant vectors."""
    if n_bins <= 0:
        raise ValueError("n_bins must be positive.")
    n = values.size
    if n == 0:
        return np.empty(0, dtype=np.int64)
    # Rank first so quantile bins remain populated when many genes share a
    # value (e.g. sparse detection rates).
    ranks = rankdata(values, method="ordinal") - 1
    return np.floor(ranks * min(int(n_bins), n) / float(n)).astype(np.int64)


def matched_random_gene_sets(
    expression: ArrayLike,
    selected_gene_indices: Sequence[int],
    *,
    n_repeats: int = 100,
    n_bins: int = 20,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Generate expression/detection-matched random gene sets.

    For every selected gene, a replacement is sampled from genes in the same
    expression-mean and detection-rate quantile bins.  Sampling is without
    replacement within a replicate.  If a bin has no unused candidate, the
    nearest non-empty bin is used; the final fallback samples from all unused
    genes.  The returned array has shape ``(n_repeats, len(selected_genes))``.
    """
    expr = _as_float_matrix(expression, "expression")
    n_cells, n_genes = expr.shape
    selected = np.asarray(selected_gene_indices, dtype=np.int64).reshape(-1)
    if selected.size == 0:
        raise ValueError("selected_gene_indices must not be empty.")
    if np.any(selected < 0) or np.any(selected >= n_genes):
        raise ValueError("selected_gene_indices contains an out-of-range index.")
    if len(np.unique(selected)) != selected.size:
        raise ValueError("selected_gene_indices must contain unique indices.")
    if n_repeats <= 0:
        raise ValueError("n_repeats must be positive.")
    if selected.size > n_genes:
        raise ValueError("A random gene set cannot be larger than the gene universe.")
    generator = np.random.default_rng() if rng is None else rng

    means = np.mean(expr, axis=0)
    detection = np.mean(expr > 0, axis=0)
    mean_bin = _quantile_bins(means, n_bins)
    det_bin = _quantile_bins(detection, n_bins)
    bin_ids = mean_bin * max(int(n_bins), 1) + det_bin
    pools: Dict[int, np.ndarray] = {
        int(bin_id): np.flatnonzero(bin_ids == bin_id).astype(np.int64)
        for bin_id in np.unique(bin_ids)
    }
    selected_bins = bin_ids[selected]
    all_indices = np.arange(n_genes, dtype=np.int64)
    output = np.empty((int(n_repeats), selected.size), dtype=np.int64)

    for repeat in range(int(n_repeats)):
        available = np.ones(n_genes, dtype=bool)
        # Null sets should not reuse observed genes when the universe has
        # enough alternatives.  For tiny synthetic universes where a
        # disjoint set of equal size is impossible, retain the observed genes
        # as a documented graceful fallback.
        if n_genes >= 2 * selected.size:
            available[selected] = False
        chosen = []
        # Randomize the order of target genes so duplicate-bin scarcity does
        # not systematically disadvantage later selected genes.
        target_order = generator.permutation(selected.size)
        temp = np.empty(selected.size, dtype=np.int64)
        for target_pos in target_order:
            target_bin = int(selected_bins[target_pos])
            candidate = pools.get(target_bin, np.empty(0, dtype=np.int64))
            candidate = candidate[available[candidate]]
            if candidate.size == 0:
                # Search bins by distance in the two-dimensional bin grid.
                target_mean_bin = target_bin // max(int(n_bins), 1)
                target_det_bin = target_bin % max(int(n_bins), 1)
                nonempty = [
                    b
                    for b, genes in pools.items()
                    if np.any(available[genes])
                ]
                if nonempty:
                    nearest = min(
                        nonempty,
                        key=lambda b: abs((b // n_bins) - target_mean_bin)
                        + abs((b % n_bins) - target_det_bin),
                    )
                    candidate = pools[nearest][available[pools[nearest]]]
            if candidate.size == 0:
                candidate = all_indices[available]
            if candidate.size == 0:
                raise RuntimeError("Unable to sample a unique matched gene set.")
            picked = int(generator.choice(candidate))
            temp[target_pos] = picked
            available[picked] = False
            chosen.append(picked)
        output[repeat] = temp
    return output


def compute_gec(
    expression: ArrayLike,
    zf: ArrayLike,
    hf: ArrayLike,
    top_k: Sequence[int] = (3, 5, 10, 20),
    *,
    n_random: int = 100,
    n_bins: int = 20,
    seed: int = 0,
    gene_names: Optional[Sequence[str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compute GEC and expression/detection-matched random baselines.

    ``gec`` remains the original mean *absolute* Spearman correlation.  The
    observed table additionally contains baseline-standardized fields:

    ``gec_abs_raw``
        Alias for the observed absolute GEC, retained with an explicit name.
    ``gec_abs_null_mean`` / ``gec_abs_null_std``
        Mean and standard deviation of the matched random sets for this
        ``(k, factor)`` pair.
    ``gec_delta`` / ``gec_z``
        Raw and standardised excess over the matched null.  ``gec_z`` is
        undefined (``nan``) when the null standard deviation is effectively
        zero and the observed value differs from the null mean.
    ``gec_percentile`` / ``gec_empirical_p``
        Empirical CDF percentile and one-sided (high-tail) permutation p-value
        of the observed GEC among the matched random sets.

    Signed correlation is kept alongside the absolute metric through
    ``gec_signed``, ``gec_signed_null_mean`` and ``gec_signed_delta``.  The
    ``positive_corr_fraction`` and ``negative_corr_fraction`` columns report
    the fraction of selected gene/FA pairs with finite positive/negative
    signed correlations.  The second returned frame contains one row per
    matched random set and keeps the legacy ``gec`` column.
    """
    expr = _as_float_matrix(expression, "expression")
    zf_arr = _as_float_matrix(zf, "zf")
    hf_arr = _as_float_matrix(hf, "hf")
    n_cells, n_genes = expr.shape
    if zf_arr.shape[0] != n_genes:
        raise ValueError("zf must have one row per expression gene.")
    if hf_arr.shape[1] != n_cells or hf_arr.shape[0] != zf_arr.shape[1]:
        raise ValueError("hf must have shape (n_factors, n_cells) matching zf and expression.")
    ks = _validate_top_k(top_k, n_genes)
    _validate_gene_names(gene_names, n_genes)
    if n_random <= 0:
        raise ValueError("n_random must be positive.")
    indices = top_gene_indices_by_fa(zf_arr, ks)
    # Reuse complete signed/absolute correlation matrices for observed and
    # matched-null sets.  This is substantially faster than repeated
    # scipy.stats calls on large datasets while preserving Spearman's tie
    # handling.
    signed_correlation_matrix = _spearman_matrix(expr, hf_arr)
    correlation_matrix = np.abs(signed_correlation_matrix)
    rng = np.random.default_rng(int(seed))
    observed_rows = []
    baseline_rows = []
    for k in ks:
        for factor in range(zf_arr.shape[1]):
            selected = indices[k][factor]
            abs_values = correlation_matrix[factor, selected]
            signed_values = signed_correlation_matrix[factor, selected]
            finite_abs = np.isfinite(abs_values)
            finite_signed = np.isfinite(signed_values)
            observed = float(np.nanmean(abs_values)) if np.any(finite_abs) else np.nan
            signed_observed = float(np.nanmean(signed_values)) if np.any(finite_signed) else np.nan
            positive_fraction = (
                float(np.mean(signed_values[finite_signed] > 0)) if np.any(finite_signed) else np.nan
            )
            negative_fraction = (
                float(np.mean(signed_values[finite_signed] < 0)) if np.any(finite_signed) else np.nan
            )
            observed_rows.append(
                {
                    "k": int(k), "factor": int(factor), "gec": observed,
                    "gec_abs_raw": observed, "gec_signed": signed_observed,
                    "positive_corr_fraction": positive_fraction,
                    "negative_corr_fraction": negative_fraction,
                    "genes": selected.tolist(),
                }
            )
            random_sets = matched_random_gene_sets(
                expr,
                selected,
                n_repeats=int(n_random),
                n_bins=int(n_bins),
                rng=rng,
            )
            for replicate, genes in enumerate(random_sets):
                corr_values = correlation_matrix[factor, genes]
                signed_values = signed_correlation_matrix[factor, genes]
                finite_signed = np.isfinite(signed_values)
                baseline_rows.append(
                    {
                        "k": int(k),
                        "factor": int(factor),
                        "replicate": int(replicate),
                        "gec": float(np.nanmean(corr_values)) if np.any(np.isfinite(corr_values)) else np.nan,
                        "gec_abs_raw": float(np.nanmean(corr_values)) if np.any(np.isfinite(corr_values)) else np.nan,
                        "gec_signed": float(np.nanmean(signed_values)) if np.any(finite_signed) else np.nan,
                        "positive_corr_fraction": (
                            float(np.mean(signed_values[finite_signed] > 0)) if np.any(finite_signed) else np.nan
                        ),
                        "negative_corr_fraction": (
                            float(np.mean(signed_values[finite_signed] < 0)) if np.any(finite_signed) else np.nan
                        ),
                        "genes": genes.tolist(),
                    }
                )
    observed_frame = pd.DataFrame(observed_rows)
    baseline_frame = pd.DataFrame(baseline_rows)

    # Standardise each observed FA against its own matched-null distribution.
    # Per-FA normalisation prevents factors with intrinsically different null
    # variances from dominating a top-k summary downstream.
    if not observed_frame.empty and not baseline_frame.empty:
        group_cols = ["k", "factor"]
        grouped = baseline_frame.groupby(group_cols, sort=False)
        null_stats = grouped["gec_abs_raw"].agg(
            gec_abs_null_mean="mean", gec_abs_null_std="std", gec_abs_null_count="count"
        ).reset_index()
        signed_stats = grouped["gec_signed"].agg(
            gec_signed_null_mean="mean", gec_signed_null_std="std"
        ).reset_index()
        observed_frame = observed_frame.merge(null_stats, on=group_cols, how="left")
        observed_frame = observed_frame.merge(signed_stats, on=group_cols, how="left")
        null_values = baseline_frame[group_cols + ["gec_abs_raw"]].rename(columns={"gec_abs_raw": "_null"})
        observed_frame["gec_delta"] = observed_frame["gec_abs_raw"] - observed_frame["gec_abs_null_mean"]
        null_std = observed_frame["gec_abs_null_std"].to_numpy(dtype=float)
        delta = observed_frame["gec_delta"].to_numpy(dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            z_values = delta / null_std
        near_zero = np.isfinite(null_std) & (null_std < 1e-8)
        equal_null = near_zero & np.isclose(delta, 0.0, atol=1e-8, rtol=0.0)
        z_values[equal_null] = 0.0
        z_values[near_zero & ~equal_null] = np.nan
        observed_frame["gec_z"] = z_values
        percentiles = []
        empirical_p = []
        for _, row in observed_frame.iterrows():
            mask = (null_values["k"] == row["k"]) & (null_values["factor"] == row["factor"])
            draws = pd.to_numeric(null_values.loc[mask, "_null"], errors="coerce").dropna().to_numpy(dtype=float)
            value = float(row["gec_abs_raw"]) if np.isfinite(row["gec_abs_raw"]) else np.nan
            if draws.size == 0 or not np.isfinite(value):
                percentiles.append(np.nan)
                empirical_p.append(np.nan)
            else:
                percentiles.append(float(np.mean(draws <= value)))
                # +1 correction gives a valid conservative one-sided p-value
                # even when every null draw is below the observed statistic.
                empirical_p.append(float((1.0 + np.sum(draws >= value)) / (draws.size + 1.0)))
        observed_frame["gec_percentile"] = percentiles
        observed_frame["gec_empirical_p"] = empirical_p
        observed_frame["gec_signed_delta"] = (
            observed_frame["gec_signed"] - observed_frame["gec_signed_null_mean"]
        )
    else:
        for column in (
            "gec_abs_null_mean", "gec_abs_null_std", "gec_abs_null_count", "gec_delta", "gec_z",
            "gec_percentile", "gec_empirical_p", "gec_signed_null_mean", "gec_signed_null_std",
            "gec_signed_delta",
        ):
            observed_frame[column] = np.nan
    return observed_frame, baseline_frame


def _coerce_gate(gate_cf: ArrayLike, n_cells: int) -> np.ndarray:
    gate = np.asarray(gate_cf, dtype=np.float64)
    if gate.ndim == 2 and 1 in gate.shape:
        gate = gate.reshape(-1)
    if gate.ndim != 1 or gate.size != n_cells:
        raise ValueError("gate_cf must be a vector with one value per cell (or shape 1 x n_cells).")
    return np.nan_to_num(gate, nan=0.0, posinf=0.0, neginf=0.0)


def compute_fcs(
    hf: ArrayLike,
    gate_cf: ArrayLike,
    cluster_labels: Sequence[int],
) -> Tuple[pd.DataFrame, np.ndarray]:
    """Compute FA-cluster specificity (FCS) as AUROC.

    The per-cell FA score is ``U[f, i] = abs(Hf[f, i] * gate_cf[i])``.  For
    every cluster and FA, FCS is AUROC of this score against membership in the
    cluster.  Clusters with no positive or no negative cells receive ``nan``.
    Returns a long DataFrame and the same values as a cluster-by-factor array
    ordered by ``np.unique(cluster_labels)``.
    """
    hf_arr = _as_float_matrix(hf, "hf")
    labels = np.asarray(cluster_labels)
    if labels.ndim != 1 or labels.size != hf_arr.shape[1]:
        raise ValueError("cluster_labels must be a vector with one label per cell.")
    gate = _coerce_gate(gate_cf, hf_arr.shape[1])
    utility = np.abs(hf_arr * gate.reshape(1, -1))
    clusters = np.unique(labels)
    matrix = np.full((clusters.size, hf_arr.shape[0]), np.nan, dtype=np.float64)
    rows = []
    for row, cluster in enumerate(clusters):
        truth = (labels == cluster).astype(np.int8)
        for factor in range(hf_arr.shape[0]):
            if np.unique(truth).size < 2:
                auc = np.nan
            else:
                with np.errstate(all="ignore"):
                    auc = float(roc_auc_score(truth, utility[factor]))
            matrix[row, factor] = auc
            rows.append({"cluster": cluster.item() if hasattr(cluster, "item") else cluster, "factor": int(factor), "fcs": auc})
    return pd.DataFrame(rows), matrix


def marker_enrichment(
    fa_gene_indices: Mapping[int, Sequence[int]],
    marker_sets: Mapping[Union[int, str], Iterable[Union[int, str]]],
    *,
    n_genes: Optional[int] = None,
    gene_names: Optional[Sequence[str]] = None,
) -> Tuple[pd.DataFrame, np.ndarray, List[Union[int, str]]]:
    """Compute ``E_kf`` marker recall for every cluster and FA.

    ``E_kf = |G_f intersect R_k| / |R_k|`` where ``G_f`` is the FA-associated
    gene set supplied by the caller (typically one of the top-k sets) and
    ``R_k`` is the Wilcoxon marker set for cluster *k*.  The returned matrix is
    ordered by sorted marker-set keys and factor ids.  Both integer indices
    and gene-name sets are accepted, but their representation must match.
    """
    if not fa_gene_indices:
        raise ValueError("fa_gene_indices must contain at least one factor.")
    factors = sorted(int(f) for f in fa_gene_indices)
    if gene_names is not None:
        names = _validate_gene_names(gene_names, len(gene_names))
        selected_sets = {}
        for factor in factors:
            converted = set()
            for item in fa_gene_indices[factor]:
                if isinstance(item, (int, np.integer)):
                    index = int(item)
                    if index < 0 or index >= len(names):
                        raise ValueError("FA gene index is outside gene_names.")
                    converted.add(names[index])
                else:
                    converted.add(str(item))
            selected_sets[int(factor)] = converted
    else:
        selected_sets = {int(f): set(fa_gene_indices[f]) for f in factors}
        if n_genes is not None:
            for values in selected_sets.values():
                if any(not isinstance(i, (int, np.integer)) or int(i) < 0 or int(i) >= n_genes for i in values):
                    raise ValueError("FA gene indices must be valid integer indices.")
    cluster_keys = sorted(marker_sets, key=lambda x: str(x))
    matrix = np.full((len(cluster_keys), len(factors)), np.nan, dtype=np.float64)
    rows = []
    for row, cluster in enumerate(cluster_keys):
        markers = set(marker_sets[cluster])
        if gene_names is not None:
            markers = {
                names[int(i)] if isinstance(i, (int, np.integer)) else str(i)
                for i in markers
            }
        denominator = float(len(markers))
        for col, factor in enumerate(factors):
            overlap = len(selected_sets[factor].intersection(markers))
            score = overlap / denominator if denominator else np.nan
            matrix[row, col] = score
            rows.append(
                {
                    "cluster": cluster,
                    "factor": int(factor),
                    "overlap": int(overlap),
                    "marker_count": int(len(markers)),
                    "gene_count": int(len(selected_sets[factor])),
                    "E": float(score) if np.isfinite(score) else np.nan,
                }
            )
    return pd.DataFrame(rows), matrix, cluster_keys


def compute_tcs(
    fcs_matrix: ArrayLike,
    enrichment_matrix: ArrayLike,
    top_q: Sequence[int] = (1, 2, 4),
    *,
    cluster_keys: Optional[Sequence[Union[int, str]]] = None,
    factor_keys: Optional[Sequence[Union[int, str]]] = None,
) -> pd.DataFrame:
    """Select each cluster's top-q FAs and average their ``E_kf`` values.

    ``TCS_k(q)`` is the mean marker-enrichment score among the q FAs with the
    largest FCS for cluster k.  Ties are resolved by factor order.  The output
    includes selected factor ids and the corresponding mean FCS for auditing.
    """
    fcs = _as_float_matrix(fcs_matrix, "fcs_matrix")
    enrich = _as_float_matrix(enrichment_matrix, "enrichment_matrix")
    if fcs.shape != enrich.shape:
        raise ValueError("fcs_matrix and enrichment_matrix must have identical shapes.")
    n_clusters, n_factors = fcs.shape
    qs = _validate_top_k(top_q, n_factors)
    clusters = list(range(n_clusters)) if cluster_keys is None else list(cluster_keys)
    factors = list(range(n_factors)) if factor_keys is None else list(factor_keys)
    if len(clusters) != n_clusters or len(factors) != n_factors:
        raise ValueError("cluster_keys/factor_keys lengths must match the matrices.")
    rows = []
    for row in range(n_clusters):
        order = np.lexsort((np.arange(n_factors), -np.nan_to_num(fcs[row], nan=-np.inf)))
        for q in qs:
            selected = order[:q]
            values = enrich[row, selected]
            tcs = float(np.nanmean(values)) if np.any(np.isfinite(values)) else np.nan
            mean_fcs = float(np.nanmean(fcs[row, selected])) if np.any(np.isfinite(fcs[row, selected])) else np.nan
            rows.append(
                {
                    "cluster": clusters[row],
                    "q": int(q),
                    "tcs": tcs,
                    "mean_fcs": mean_fcs,
                    "selected_factors": [factors[int(i)] for i in selected],
                }
            )
    return pd.DataFrame(rows)


def permute_tcs(
    fcs_matrix: ArrayLike,
    enrichment_matrix: ArrayLike,
    top_q: Sequence[int] = (1, 2, 4),
    *,
    n_permutations: int = 1000,
    seed: int = 0,
) -> Dict[str, object]:
    """Permutation null for TCS by shuffling FA identities.

    FA columns of the enrichment matrix are jointly permuted for each
    replicate, preserving the observed FCS and enrichment marginals while
    destroying their FA correspondence.  Returns observed TCS, null draws of
    shape ``(n_permutations, n_clusters, len(top_q))``, and one-sided empirical
    p-values ``(null >= observed)`` for every cluster and q.
    """
    fcs = _as_float_matrix(fcs_matrix, "fcs_matrix")
    enrich = _as_float_matrix(enrichment_matrix, "enrichment_matrix")
    if fcs.shape != enrich.shape:
        raise ValueError("fcs_matrix and enrichment_matrix must have identical shapes.")
    if n_permutations <= 0:
        raise ValueError("n_permutations must be positive.")
    qs = _validate_top_k(top_q, fcs.shape[1])
    observed_df = compute_tcs(fcs, enrich, qs)
    n_clusters = fcs.shape[0]
    q_count = len(qs)
    observed = observed_df["tcs"].to_numpy(dtype=np.float64).reshape(n_clusters, q_count)
    null = np.full((int(n_permutations), n_clusters, q_count), np.nan, dtype=np.float64)
    rng = np.random.default_rng(int(seed))
    for rep in range(int(n_permutations)):
        permutation = rng.permutation(fcs.shape[1])
        permuted = enrich[:, permutation]
        perm_df = compute_tcs(fcs, permuted, qs)
        null[rep] = perm_df["tcs"].to_numpy(dtype=np.float64).reshape(n_clusters, q_count)
    p_values = np.full_like(observed, np.nan)
    for row in range(n_clusters):
        for col in range(q_count):
            if np.isfinite(observed[row, col]):
                draws = null[:, row, col]
                draws = draws[np.isfinite(draws)]
                if draws.size:
                    p_values[row, col] = (1.0 + np.sum(draws >= observed[row, col])) / (draws.size + 1.0)
    return {
        "q_values": np.asarray(qs, dtype=np.int64),
        "observed": observed,
        "null": null,
        "p_values": p_values,
        "observed_table": observed_df,
    }


# Alias spelling retained for downstream scripts.
permutation_tcs = permute_tcs


def compute_experiment2(
    expression: ArrayLike,
    gene_names: Sequence[str],
    snapshot: Mapping[str, object],
    top_ks: Sequence[int],
    random_sets: int = 100,
    seed: int = 0,
    dataset: str = "",
) -> Dict[str, pd.DataFrame]:
    """Compute Experiment 2 and return observed/baseline long tables."""
    zf = snapshot.get("Zf")
    hf = snapshot.get("Hf")
    if zf is None or hf is None:
        raise ValueError("snapshot must contain Zf and Hf.")
    zf_arr = np.asarray(zf)
    expr_arr = np.asarray(expression)
    if zf_arr.ndim == 2 and zf_arr.shape[0] != expr_arr.shape[1] and zf_arr.shape[1] == expr_arr.shape[1]:
        zf = zf_arr.T
    observed, baseline = compute_gec(
        expression, zf, hf, top_k=top_ks, n_random=int(random_sets), seed=int(seed), gene_names=gene_names
    )
    observed = observed.copy()
    baseline = baseline.copy()
    observed["dataset"] = str(dataset)
    observed["seed"] = int(seed)
    observed["kind"] = "observed"
    baseline["dataset"] = str(dataset)
    baseline["seed"] = int(seed)
    baseline["kind"] = "random"
    gene_rows = []
    zf_for_genes = np.asarray(zf, dtype=np.float64)
    hf_for_genes = np.asarray(hf, dtype=np.float64)
    expr_for_genes = np.asarray(expression, dtype=np.float64)
    for _, row in observed.iterrows():
        factor = int(row["factor"])
        for rank, gene_index in enumerate(row["genes"], start=1):
            gene_index = int(gene_index)
            gene_rows.append(
                {
                    "dataset": str(dataset), "seed": int(seed), "kind": "gene",
                    "k": int(row["k"]), "factor": factor, "fa": factor,
                    "rank": int(rank), "gene_index": gene_index,
                    "gene": str(gene_names[gene_index]),
                    "correlation": _safe_abs_spearman(expr_for_genes[:, gene_index], hf_for_genes[factor]),
                    "abs_correlation": _safe_abs_spearman(expr_for_genes[:, gene_index], hf_for_genes[factor]),
                    "signed_correlation": _safe_signed_spearman(expr_for_genes[:, gene_index], hf_for_genes[factor]),
                    "weight": float(abs(zf_for_genes[gene_index, factor])),
                }
            )
    gene_table = pd.DataFrame(gene_rows)
    table = pd.concat([observed, baseline, gene_table], ignore_index=True, sort=False)
    # Keep the gene universe and selected sets available to Experiment 4.  The
    # latter are stored in ``observed['genes']`` as integer indices.
    return {
        "observed": observed,
        "baseline": baseline,
        "table": table,
        "gene_table": gene_table,
        "gene_names": [str(gene) for gene in gene_names],
        "zf": np.asarray(zf, dtype=np.float64),
    }


def compute_experiment3(
    snapshot: Mapping[str, object],
    pred_labels: Sequence[int],
    dataset: str = "",
    seed: int = 0,
) -> Dict[str, object]:
    """Compute Experiment 3 from ``U_if=abs(Hf*gate_cf)``."""
    if snapshot.get("Hf") is None or snapshot.get("gate_cf") is None:
        raise ValueError("snapshot must contain Hf and gate_cf.")
    table, matrix = compute_fcs(snapshot["Hf"], snapshot["gate_cf"], pred_labels)
    table = table.copy()
    table["dataset"] = str(dataset)
    table["seed"] = int(seed)
    dominant = (
        table.sort_values(["factor", "fcs"], ascending=[True, False])
        .drop_duplicates("factor")
        .loc[:, ["factor", "cluster"]]
        .rename(columns={"cluster": "dominant_cluster"})
    )
    dominant["fa"] = dominant["factor"]
    dominant["dataset"] = str(dataset)
    dominant["seed"] = int(seed)
    dominant["max_fcs"] = dominant["factor"].map(
        table.groupby("factor", sort=False)["fcs"].max()
    )
    # One run contributes one winning cluster per FA.  The cross-seed
    # proportion is calculated in ``aggregate`` once all run-level winners
    # are available; retain a run-level table here for auditability.
    counts = dominant.loc[:, ["dataset", "seed", "factor", "fa", "dominant_cluster"]].copy()
    counts["fa_count"] = 1
    counts["fa_proportion"] = 1.0
    return {"table": table, "matrix": matrix, "dominant": dominant, "dominant_proportions": counts}


def compute_experiment4(
    exp2: Mapping[str, object],
    exp3: Mapping[str, object],
    reference_markers: pd.DataFrame,
    gene_count: int,
    top_ks: Sequence[int],
    top_qs: Sequence[int],
    gene_names: Optional[Sequence[str]] = None,
    permutations: int = 1000,
    seed: int = 0,
    dataset: str = "",
) -> Dict[str, object]:
    """Compute documented Gene-FA-Cluster enrichment and TCS permutations."""
    observed2 = exp2["observed"]
    if observed2.empty:
        raise ValueError("Experiment 2 observed table is empty.")
    fcs_table = exp3["table"]
    clusters = list(pd.unique(fcs_table["cluster"]))
    factors = sorted(int(x) for x in pd.unique(fcs_table["factor"]))
    fcs_matrix = fcs_table.pivot_table(index="cluster", columns="factor", values="fcs").reindex(index=clusters, columns=factors).to_numpy(dtype=float)
    if reference_markers is None or reference_markers.empty:
        raise ValueError("Reference marker table is empty.")
    marker_col = "pred_cluster" if "pred_cluster" in reference_markers.columns else ("cluster" if "cluster" in reference_markers.columns else "truth_label")
    marker_sets = {cluster: set(frame["gene"].astype(str)) for cluster, frame in reference_markers.groupby(marker_col)}
    if gene_names is None:
        gene_names = exp2.get("gene_names")
    name_to_index = {str(name): int(i) for i, name in enumerate(gene_names or [])}
    if name_to_index:
        marker_sets = {
            cluster: {name_to_index[item] for item in values if item in name_to_index}
            for cluster, values in marker_sets.items()
        }
    # Reference labels may be strings while predicted labels are integer codes.
    if set(clusters).intersection(marker_sets):
        cluster_marker_lookup = {cluster: marker_sets.get(cluster, set()) for cluster in clusters}
    else:
        # Reference labels are often strings while KMeans labels are integer
        # codes.  Match by label text first, then use sorted order as the
        # deterministic fallback used by the training adapter.
        marker_by_text = {str(key): value for key, value in marker_sets.items()}
        cluster_marker_lookup = {cluster: marker_by_text.get(str(cluster), set()) for cluster in clusters}
        if not any(cluster_marker_lookup.values()):
            ordered_markers = [marker_sets[key] for key in sorted(marker_sets, key=str)]
            cluster_marker_lookup = {
                cluster: ordered_markers[index] if index < len(ordered_markers) else set()
                for index, cluster in enumerate(clusters)
            }
    all_rows = []
    enrichment_frames = []
    network_frames = []
    perm_rows = []
    null_rows = []
    # Gene names are not needed here because reference and selected genes use the same names.
    for k in sorted({int(v) for v in top_ks}):
        factor_gene_sets = {}
        for factor in factors:
            mask = (observed2["k"] == k) & (observed2["factor"] == factor)
            if "kind" in observed2.columns:
                mask &= observed2["kind"].astype(str).str.lower().isin(["observed", "real", ""])
            gene_values = observed2.loc[mask, "genes"]
            if gene_values.empty:
                continue
            values = gene_values.iloc[0]
            if isinstance(values, str):
                import ast
                values = ast.literal_eval(values)
            factor_gene_sets[factor] = set(values)
        if not factor_gene_sets:
            continue
        # The observed gene IDs are stable indices; marker genes may be names. Resolve both forms.
        marker_index_sets = {}
        for cluster, markers in cluster_marker_lookup.items():
            marker_index_sets[cluster] = set(markers)
        enrichment_rows = []
        enrich_matrix = np.full((len(clusters), len(factors)), np.nan, dtype=float)
        for ci, cluster in enumerate(clusters):
            markers = marker_index_sets.get(cluster, set())
            for fi, factor in enumerate(factors):
                fa_genes = factor_gene_sets.get(factor, set())
                overlap = len(fa_genes.intersection(markers))
                denom_m = max(len(fa_genes), 1)
                # E_kf = (overlap / |M_f|) / (|R_k| / |G|).
                score = (overlap / float(denom_m)) / (len(markers) / float(max(gene_count, 1))) if markers else np.nan
                enrich_matrix[ci, fi] = score
                enrichment_rows.append({"dataset": dataset, "seed": seed, "k": k, "cluster": cluster, "factor": factor, "overlap": overlap, "fa_gene_count": len(fa_genes), "marker_count": len(markers), "enrichment": score})
        enrich_df = pd.DataFrame(enrichment_rows)
        # Expand selected FA genes into tripartite network edges.  This is kept
        # separate from the compact enrichment table so heatmaps do not count
        # a factor once per gene.
        if gene_names:
            network_rows = []
            for _, enrich_row in enrich_df.iterrows():
                factor = int(enrich_row["factor"])
                cluster = enrich_row["cluster"]
                for gene_value in sorted(factor_gene_sets.get(factor, set()), key=str):
                    if isinstance(gene_value, (int, np.integer)) and int(gene_value) < len(gene_names):
                        gene = str(gene_names[int(gene_value)])
                    else:
                        gene = str(gene_value)
                    network_rows.append(
                        {
                            "dataset": str(dataset), "seed": int(seed), "kind": "network", "k": int(k),
                            "cluster": cluster, "factor": factor, "fa": factor, "gene": gene,
                            "weight": float(enrich_row["enrichment"]) if np.isfinite(enrich_row["enrichment"]) else 0.0,
                        }
                    )
            if network_rows:
                network_frames.append(pd.DataFrame(network_rows))
        for q in sorted({int(v) for v in top_qs}):
            tcs_rows = compute_tcs(fcs_matrix, enrich_matrix, [q], cluster_keys=clusters, factor_keys=factors)
            tcs_rows["dataset"] = str(dataset); tcs_rows["seed"] = int(seed); tcs_rows["k"] = int(k); tcs_rows["kind"] = "tcs"
            tcs_rows = tcs_rows.rename(columns={"tcs": "tcs_real"})
            tcs_rows["tcs"] = tcs_rows["tcs_real"]
            all_rows.append(tcs_rows)
            permuted = permute_tcs(fcs_matrix, enrich_matrix, [q], n_permutations=int(permutations), seed=int(seed) + k * 1000 + q)
            null = permuted["null"][:, :, 0]
            p_values = permuted["p_values"][:, 0]
            for ci, cluster in enumerate(clusters):
                real = float(tcs_rows.loc[tcs_rows["cluster"] == cluster, "tcs_real"].iloc[0])
                row = {"dataset": str(dataset), "seed": int(seed), "k": int(k), "q": int(q), "cluster": cluster, "tcs_real": real, "p_value": float(p_values[ci]) if np.isfinite(p_values[ci]) else np.nan}
                perm_rows.append(row)
                for rep, value in enumerate(null[:, ci]):
                    null_rows.append({**row, "permutation": int(rep), "tcs_null": float(value)})
        # Keep enrichment rows separate from TCS rows for plotting and CSV use.
        if not enrich_df.empty:
            enrich_df["kind"] = "enrichment"
            enrichment_frames.append(enrich_df)
    tcs_table = pd.concat([frame if isinstance(frame, pd.DataFrame) else pd.DataFrame(frame) for frame in all_rows], ignore_index=True, sort=False) if all_rows else pd.DataFrame()
    enrichment_table = pd.concat(enrichment_frames, ignore_index=True) if enrichment_frames else pd.DataFrame()
    network_table = pd.concat(network_frames, ignore_index=True) if network_frames else pd.DataFrame()
    parts = [frame for frame in (tcs_table, enrichment_table, network_table) if isinstance(frame, pd.DataFrame) and not frame.empty]
    combined = pd.concat(parts, ignore_index=True, sort=False) if parts else pd.DataFrame()
    return {"table": combined, "enrichment_table": enrichment_table, "permutations_table": pd.DataFrame(null_rows), "summary": pd.DataFrame(perm_rows)}


__all__ = [
    "rank_genes_by_fa",
    "top_gene_indices_by_fa",
    "matched_random_gene_sets",
    "compute_gec",
    "compute_fcs",
    "marker_enrichment",
    "compute_tcs",
    "permute_tcs",
    "permutation_tcs",
    "compute_experiment2",
    "compute_experiment3",
    "compute_experiment4",
]
