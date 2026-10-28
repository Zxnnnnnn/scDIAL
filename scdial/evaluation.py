"""Post-training evaluation using the original direct GEC/FCS/TCS helpers."""
from __future__ import annotations

import numpy as np

from .clustering import evaluate_with_details
from .fa_metrics import compute_experiment2, compute_experiment3, compute_experiment4
from .markers import as_numpy, predict_clusters, score_markers
from .reference_markers import build_reference_marker_table, match_clusters_to_truth


def frame_records(frame):
    return frame.to_dict(orient="records")


def evaluate_run(training, artifacts, args):
    """All uses of true labels and reference markers stay in this boundary."""
    zc = as_numpy(artifacts["Zc"])
    labels = np.asarray(artifacts["labels"], dtype=np.int64)
    summary, evaluation_runs = evaluate_with_details(zc.T, labels, artifacts["n_clusters"])
    names = ("ACC", "NMI", "Purity", "ARI", "Fscore", "Precision", "Recall")
    for index, name in enumerate(names):
        if not np.array_equal(summary[index], [training["metrics"][name]["mean"], training["metrics"][name]["std"]]):
            raise RuntimeError("Saved snapshot does not reproduce the selected epoch's " + name)

    report = {
        "best_epoch": training["best_epoch"], "selection_metric": training["selection_metric"],
        "clustering_metrics": training["metrics"], "clustering_evaluation_runs": evaluation_runs,
        "clustering_metric_unit": "fraction",
        "time_seconds": training["time_seconds"], "best_by_metric": training["best_by_metric"],
        "loss_history": training["loss_history"],
        "marker_update_count": training["marker_update_count"],
        "marker_loss_active_at_best": artifacts.get("marker_loss_active_at_best", False),
        "marker_loss_has_updated_parameters_at_best": artifacts.get("marker_loss_has_updated_parameters_at_best", False),
        "marker_updates_before_best": artifacts.get("marker_updates_before_best", 0),
        "training_marker_state_at_best": artifacts.get("marker_state"),
        "training_marker_state_at_end": artifacts.get("marker_state_at_end"),
        "marker_refresh_history": training["marker_refresh_history"],
        "alignment": artifacts["alignment_info"],
        "selected_genes": [str(g) for g in artifacts["selected_genes"]],
        "selected_gene_count": len(artifacts["selected_genes"]),
        "cell_count": int(zc.shape[1]), "cluster_count": artifacts["n_clusters"],
        "mc": training["mc"], "mf": training["mf"],
    }
    if artifacts.get("selected_expression") is None or artifacts.get("Hf") is None:
        report["marker_evaluation"] = {
            "status": "unavailable", "reason": "Requires named gene-aligned expression and an active feature branch."}
        return report, None

    expression = np.asarray(artifacts["selected_expression"], dtype=np.float32)
    genes = report["selected_genes"]
    predicted = predict_clusters(zc, artifacts["n_clusters"], args.seed)
    ranking = score_markers(expression, genes, artifacts, predicted, args.marker_top_l)
    report["marker_partition"] = {
        "method": "KMeans", "seed": args.seed, "n_init": 20,
        "labels": predicted.tolist(), "label_space": "raw predicted clusters",
        "gene_index_space": "selected_genes",
    }
    report["final_marker_indices_by_cluster"] = {
        str(k): v.tolist() for k, v in ranking["marker_indices"].items()}
    report["marker_counts_by_cluster"] = {str(k): len(v) for k, v in ranking["marker_indices"].items()}
    report["marker_scoring"] = {
        "formula": "sum_f W[f,g]*max(FCS[c,f]-0.5,0)*abs(rho[f,g]) / sum_f W[f,g]*max(FCS[c,f]-0.5,0); zero denominator gives zero",
        "top_l": args.marker_top_l, "tie_break": "ascending selected-gene index",
        "positive_scores_only": True, "true_labels_used": False,
        "fa_gene_weights": ranking["weights"],
        "signed_gene_fa_correlations": ranking["signed_correlations"],
        "fcs_raw_cluster_order": ranking["clusters"], "fcs_raw_cluster_matrix": ranking["fcs"],
        "scores_by_raw_cluster": ranking["scores"],
    }

    if artifacts["data_type"] == "multi":
        source_indices = np.asarray(artifacts["source_gene_indices"], dtype=np.int64)
        if source_indices.shape != (len(genes),):
            raise ValueError("RNA source indices must match the aligned gene order.")
        ranking["table"]["source_gene_index"] = ranking["table"]["gene_index"].map(
            lambda index: int(source_indices[int(index)]))
        report["marker_input_configuration"] = {
            "marker_expression_view": "RNA",
            "marker_distance_standardization": "per_gene",
            "model_input_standardization": "per_cell",
            "marker_candidate_genes": len(genes),
        }
        report["marker_evaluation"] = {"status": "not_requested_for_multi_omics"}
        return report, ranking["table"]

    # Match arbitrary cluster IDs to truth names only for the original TCS evaluation.
    label_mapping = artifacts.get("label_mapping")
    if label_mapping:
        inverse = {int(v): str(k) for k, v in label_mapping.items()}
        reference_labels = np.asarray([inverse[int(v)] for v in labels], dtype=object)
    else:
        reference_labels = labels.astype(str)
    mapped, pred_to_truth, _ = match_clusters_to_truth(reference_labels, predicted)
    reference = build_reference_marker_table(
        expression, genes, reference_labels, dataset=artifacts["dataset"], seed=args.seed,
        variant="reference", top_k=50, label_column="truth_label")
    reference["pred_cluster"] = reference["cluster"].astype(str)
    # Keep the public summary settings available independently of extra diagnostics.
    top_k = sorted(k for k in set(args.fa_top_k) | {10} if k <= len(genes))
    top_q = sorted(q for q in set(args.fa_top_q) | {2} if q <= as_numpy(artifacts["Hf"]).shape[0])
    if not top_k or not top_q:
        raise ValueError("At least one FA top-k and top-q must fit this dataset.")
    exp2 = compute_experiment2(expression, genes, artifacts, top_k, args.random_sets,
                               args.seed, artifacts["dataset"])
    exp3 = compute_experiment3(artifacts, mapped, artifacts["dataset"], args.seed)
    exp4 = compute_experiment4(
        exp2, exp3, reference, len(genes), top_k, top_q, gene_names=genes,
        permutations=args.permutations, seed=args.seed, dataset=artifacts["dataset"])
    observed = exp2["observed"]
    report["marker_evaluation"] = {
        "status": "complete", "fa_top_k": top_k, "fa_top_q": top_q,
        "reference_marker_method": "true-label Wilcoxon, tie_correct=True, top 50",
        "true_labels_used_for_training": False,
        "prediction_to_reference_label": pred_to_truth,
        "gec": {
            "definition": "Mean absolute Spearman across each FA's top-k abs(Zf) genes",
            "by_fa": frame_records(observed),
            "macro_mean_by_top_k": {str(k): float(observed.loc[observed.k == k, "gec"].mean()) for k in top_k},
            "matched_random_sets": args.random_sets,
        },
        "fcs": {
            "definition": "AUROC(abs(Hf[f,:]*gate), predicted-cluster membership)",
            "by_cluster_fa": frame_records(exp3["table"]),
            "macro_mean_all_cluster_fa_pairs": float(exp3["table"]["fcs"].mean()),
            "mean_max_per_fa": float(exp3["table"].groupby("factor")["fcs"].max().mean()),
        },
        "tcs": {
            "definition": "Mean fold enrichment among the top-q FAs by FCS; original direct wrapper",
            "fold_enrichment": "(overlap/FA_gene_count)/(reference_marker_count/selected_gene_count)",
            "by_cluster": frame_records(exp4["summary"]),
            "fa_cluster_enrichment": frame_records(exp4["enrichment_table"]),
            "macro_mean_by_top_k_top_q": {
                str(k): {str(q): float(exp4["summary"].loc[
                    (exp4["summary"].k == k) & (exp4["summary"].q == q), "tcs_real"].mean()) for q in top_q}
                for k in top_k},
            "permutations": args.permutations,
        },
    }
    return report, ranking["table"]
