"""Original Wilcoxon and truth matching helpers, used only after training."""
from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple
import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
from scipy.optimize import linear_sum_assignment

def _safe_gene_list(gene_list: Iterable[str]) -> List[str]:
    cleaned = []
    seen = set()
    for gene in gene_list:
        gene = str(gene).strip()
        if not gene or gene in seen:
            continue
        seen.add(gene)
        cleaned.append(gene)
    return cleaned



def match_clusters_to_truth(
    true_labels: Sequence[object],
    pred_labels: Sequence[int],
) -> Tuple[np.ndarray, Dict[int, object], np.ndarray]:
    true_labels = np.asarray(true_labels)
    pred_labels = np.asarray(pred_labels)
    true_values, true_inverse = np.unique(true_labels, return_inverse=True)
    pred_values, pred_inverse = np.unique(pred_labels, return_inverse=True)

    contingency = np.zeros((len(true_values), len(pred_values)), dtype=np.int64)
    for t_idx, p_idx in zip(true_inverse, pred_inverse):
        contingency[t_idx, p_idx] += 1

    row_ind, col_ind = linear_sum_assignment(-contingency)
    mapping = {
        int(pred_values[col]): true_values[row]
        for row, col in zip(row_ind, col_ind)
    }
    mapped = np.asarray([mapping[int(label)] for label in pred_labels], dtype=object)
    return mapped, mapping, contingency



def _baseline_marker_table(
    expr: np.ndarray,
    gene_names: Sequence[str],
    labels: Sequence[object],
    method: str,
    top_k: int,
    dataset: str,
    seed: int,
    variant: str,
    label_column: str,
) -> pd.DataFrame:
    gene_names = _safe_gene_list(gene_names)
    adata = AnnData(X=np.asarray(expr, dtype=np.float32))
    adata.var_names = pd.Index(gene_names)
    adata.obs[label_column] = pd.Categorical(np.asarray(labels).astype(str))
    sc.tl.rank_genes_groups(
        adata,
        groupby=label_column,
        method=method,
        n_genes=min(int(top_k), len(gene_names)),
        tie_correct=True,
    )
    frames = []
    groups = list(adata.obs[label_column].cat.categories)
    for group in groups:
        try:
            ranked = sc.get.rank_genes_groups_df(adata, group=group)
        except Exception:
            ranked = pd.DataFrame()
        if ranked.empty:
            continue
        ranked = ranked.head(int(top_k)).copy()
        ranked["dataset"] = dataset
        ranked["seed"] = seed
        ranked["variant"] = variant
        ranked["cluster"] = group
        ranked["rank"] = np.arange(1, len(ranked) + 1)
        ranked["method"] = method
        ranked = ranked.rename(columns={"names": "gene"})
        frames.append(ranked[["dataset", "seed", "variant", "cluster", "rank", "gene", "logfoldchanges", "pvals_adj", "method"]])
    if not frames:
        return pd.DataFrame(
            columns=["dataset", "seed", "variant", "cluster", "rank", "gene", "logfoldchanges", "pvals_adj", "method"]
        )
    return pd.concat(frames, ignore_index=True)



def rank_posthoc_markers(
    expr: np.ndarray,
    gene_names: Sequence[str],
    labels: Sequence[object],
    *,
    method: str = "wilcoxon",
    top_k: int = 50,
    dataset: str = "",
    seed: int = 0,
    variant: str = "full",
    label_column: str = "cluster",
) -> pd.DataFrame:
    return _baseline_marker_table(
        expr=expr,
        gene_names=gene_names,
        labels=labels,
        method=method,
        top_k=top_k,
        dataset=dataset,
        seed=seed,
        variant=variant,
        label_column=label_column,
    )



def build_reference_marker_table(
    expr: np.ndarray,
    gene_names: Sequence[str],
    labels: Sequence[object],
    *,
    dataset: str,
    seed: int,
    variant: str,
    top_k: int = 50,
    label_column: str = "cluster",
) -> pd.DataFrame:
    return rank_posthoc_markers(
        expr,
        gene_names,
        labels,
        method="wilcoxon",
        top_k=top_k,
        dataset=dataset,
        seed=seed,
        variant=variant,
        label_column=label_column,
    ).assign(mapped_label=lambda frame: frame["cluster"])


