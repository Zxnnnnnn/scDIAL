from __future__ import division, print_function

import copy
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from .model import DBONet_with_E
from .clustering import StatisticClustering
from .data import features_to_adj_omics, features_to_adj_single_omics
from .marker_consistency import MarkerConsistency


def _format_metric(metric_value):
    return "{}({})".format(metric_value[0] * 100, metric_value[1] * 100)


def _metrics_to_result(metrics):
    return dict(zip(SCORE_NAMES, [_format_metric(item) for item in metrics]))


def _metric_score(metrics, metric_name):
    metric_idx = SCORE_NAMES.index(metric_name)
    return float(metrics[metric_idx][0])


def _sum_reconstruction_loss(criterion, recon_list, target_list, n_view, device):
    loss = torch.tensor(0.0, device=device)
    for view_idx in range(n_view):
        loss = loss + criterion(recon_list[view_idx], target_list[view_idx])
    return loss


def _dictionary_perturbation_loss(dictionary_list, svd_list, n_view, device):
    loss_pert = torch.tensor(0.0, device=device)
    for view_idx in range(n_view):
        dictionary = dictionary_list[view_idx]
        loss_pert = loss_pert + torch.norm(dictionary - svd_list[view_idx], p="fro") ** 2
    return loss_pert / n_view


def _detach_to_cpu(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, list):
        return [_detach_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_detach_to_cpu(item) for item in value)
    if isinstance(value, dict):
        return {key: _detach_to_cpu(item) for key, item in value.items()}
    return value


def _snapshot_outputs(outputs):
    return {key: _detach_to_cpu(outputs[key]) for key in SNAPSHOT_KEYS if key in outputs}


def train(
    features_c,
    features_f,
    epoch,
    block,
    labels,
    n_view,
    n_clusters,
    model,
    optimizer,
    scheduler,
    device,
    lambda_view_entropy,
    enable_feature_aux,
    enable_d_perturb,
    lambda_d_pert,
    lambda_f=0.1,
    lambda_cf=0.05,
    beta_f=0.05,
    eta_f1=0.05,
    capture_artifacts=False,
    marker_controller=None,
    lambda_marker=0.0,
    history=None,
    training_state=None,
):
    del block

    best_by_metric = {
        metric_name: {
            "score": -float("inf"),
            "metrics": None,
            "best_epoch": 0,
            "snapshot": None,
        }
        for metric_name in SCORE_NAMES
    }
    features_c = [torch.tensor(x, dtype=torch.float32, device=device) for x in features_c]
    features_f = [torch.tensor(x.T, dtype=torch.float32, device=device) for x in features_f]

    history = history if history is not None else []
    marker_updates = 0
    criterion = nn.MSELoss()
    run_feature_branch = bool(enable_feature_aux) or bool(model.enable_cf_injection)
    effective_lambda_f = lambda_f if bool(enable_feature_aux) else 0.0
    effective_lambda_cf = lambda_cf if bool(enable_feature_aux) else 0.0

    with tqdm(total=epoch, desc="Training") as pbar:
        start_time = time.time()
        for epoch_idx in range(epoch):
            model.train()
            model.enable_d_perturb = bool(enable_d_perturb)

            optimizer.zero_grad()
            outputs = model(features_c, features_f, run_feature_branch=run_feature_branch)

            zc = outputs["Zc"]
            cell_recon = [torch.mm(outputs["Dc"][v], zc) for v in range(n_view)]
            loss_c_rec = _sum_reconstruction_loss(criterion, cell_recon, features_c, n_view, device)
            loss_c_view = -torch.sum(outputs["alpha_c"] * torch.log(outputs["alpha_c"] + 1e-12))
            loss_c_d_pert = _dictionary_perturbation_loss(
                outputs["Dc"], outputs["Dc_svd"], n_view, device
            )

            if run_feature_branch:
                zf = outputs["Zf"]
                feature_recon = [torch.mm(outputs["Df"][v], zf) for v in range(n_view)]
                loss_f_rec = _sum_reconstruction_loss(criterion, feature_recon, features_f, n_view, device)
                loss_f_view = -torch.sum(outputs["alpha_f"] * torch.log(outputs["alpha_f"] + 1e-12))
                loss_f_d_pert = _dictionary_perturbation_loss(
                    outputs["Df"], outputs["Df_svd"], n_view, device
                )

                zc_hat_norm = F.normalize(outputs["Zc_hat"], p=2, dim=0)
                zc_norm = F.normalize(zc, p=2, dim=0)
                loss_cf_feature = criterion(zc_hat_norm, zc_norm.detach())
            else:
                loss_f_rec = torch.tensor(0.0, device=device)
                loss_f_view = torch.tensor(0.0, device=device)
                loss_f_d_pert = torch.tensor(0.0, device=device)
                loss_cf_feature = torch.tensor(0.0, device=device)

            loss = (
                loss_c_rec
                + lambda_view_entropy * loss_c_view
                + lambda_d_pert * loss_c_d_pert
                + effective_lambda_f
                * (loss_f_rec + beta_f * loss_f_view + eta_f1 * loss_f_d_pert)
                + effective_lambda_cf * loss_cf_feature
            )

            original_loss = loss
            marker_loss = zc.new_zeros(())
            marker_active = False
            marker_gradient_norm = 0.0
            if marker_controller is not None and lambda_marker > 0:
                marker_loss, marker_active = marker_controller.loss(outputs, epoch_idx + 1)
                if marker_active:
                    marker_gradient = torch.autograd.grad(marker_loss, zc, retain_graph=True)[0]
                    marker_gradient_norm = float(torch.linalg.vector_norm(marker_gradient).detach().cpu())
                    loss = original_loss + lambda_marker * marker_loss
            marker_updates_before = marker_updates
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            scheduler.step(loss)
            if marker_active and marker_gradient_norm > 0:
                marker_updates += 1
            original_value = float(original_loss.detach().cpu())
            marker_value = float(marker_loss.detach().cpu())
            weighted_marker = lambda_marker * marker_value if marker_active else 0.0
            history.append({
                "epoch": epoch_idx + 1, "original_loss": original_value,
                "marker_loss": marker_value, "weighted_marker_loss": weighted_marker,
                "total_loss": float(loss.detach().cpu()),
                "effective_lambda_marker": lambda_marker if marker_active else 0.0,
                "weighted_marker_to_original_ratio": abs(weighted_marker) / (abs(original_value) + 1e-8),
                "marker_gradient_norm_zc": marker_gradient_norm,
                "marker_loss_active": marker_active,
                "marker_updates_before_snapshot": marker_updates_before,
                "marker_cache_version": (marker_controller.cache or {}).get("version", 0) if marker_controller else 0,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "cell_reconstruction": float(loss_c_rec.detach().cpu()),
                "feature_reconstruction": float(loss_f_rec.detach().cpu()),
                "cross_feature_alignment": float(loss_cf_feature.detach().cpu()),
            })

            output_zz = zc.detach().cpu().numpy().T
            metrics = StatisticClustering(output_zz, labels, n_clusters)
            history[-1]["clustering"] = {
                name: {"mean": float(value[0]), "std": float(value[1])}
                for name, value in zip(SCORE_NAMES, metrics)
            }
            for metric_name in SCORE_NAMES:
                metric_score = _metric_score(metrics, metric_name)
                if metric_score > best_by_metric[metric_name]["score"]:
                    snapshot = _snapshot_outputs(outputs) if capture_artifacts else None
                    if snapshot is not None:
                        snapshot["best_epoch"] = epoch_idx + 1
                        snapshot["selection_metric"] = metric_name
                        snapshot["marker_loss_active_at_best"] = marker_active
                        snapshot["marker_loss_has_updated_parameters_at_best"] = marker_updates_before > 0
                        snapshot["marker_updates_before_best"] = marker_updates_before
                        snapshot["marker_state"] = copy.deepcopy(marker_controller.cache) if marker_controller else None
                    best_by_metric[metric_name] = {
                        "score": metric_score,
                        "metrics": metrics,
                        "best_epoch": epoch_idx + 1,
                        "snapshot": snapshot,
                    }

            pbar.set_postfix(
                {
                    "Rec": "{:.4f}".format(float(loss_c_rec.detach().cpu())),
                    "DPertC": "{:.4f}".format(float(loss_c_d_pert.detach().cpu())),
                    "FRec": "{:.4f}".format(float(loss_f_rec.detach().cpu())),
                    "DPertF": "{:.4f}".format(float(loss_f_d_pert.detach().cpu())),
                    "CrossF": "{:.4f}".format(float(loss_cf_feature.detach().cpu())),
                    "Lmarker": "{:.6f}".format(marker_value),
                }
            )
            pbar.update(1)

        total_time = time.time() - start_time

    for metric_name, metric_info in best_by_metric.items():
        if metric_info["metrics"] is None:
            raise RuntimeError(f"No metrics were recorded for {metric_name}; epoch must be positive.")
        metric_info["result"] = _metrics_to_result(metric_info["metrics"])

    if training_state is not None:
        training_state["marker_refresh_history"] = marker_controller.refresh_history if marker_controller else []
        training_state["marker_state_at_end"] = copy.deepcopy(marker_controller.cache) if marker_controller else None
        training_state["marker_update_count"] = marker_updates
    if capture_artifacts:
        return best_by_metric, total_time
    return best_by_metric, total_time


def main(dataset_name, args, collect_artifacts=False):
    seed = args.seed
    lr = args.lr
    if args.device != "cpu" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available. Use --device cpu.")
    device = torch.device("cpu" if args.device == "cpu" else "cuda:" + args.device)

    if device.type == "cuda":
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    data_type = str(getattr(args, "data_type", "single")).lower()
    selected_exprs = None
    lambda_marker = float(getattr(args, "lambda_marker", 0.0))
    need_expression = collect_artifacts or lambda_marker > 0
    if data_type == "single":
        single_outputs = features_to_adj_single_omics(
            dataset_name,
            args.data_root,
            label_key=args.label_key,
            gene_set=args.gene_set,
            n_top_genes=args.n_top_genes,
            log1p=args.log1p,
            anchor_multiplier=getattr(args, "anchor_multiplier", None),
            mc_multiplier=getattr(args, "mc_multiplier", None),
            mf_multiplier=getattr(args, "mf_multiplier", None),
            return_expression=need_expression,
        )
        if need_expression:
            (
                features_c,
                features_common,
                zc_init,
                dc_init,
                ec_init,
                zf_init,
                df_init,
                ef_init,
                mc,
                mf,
                n_view,
                num_sample,
                labels,
                n_clusters,
                common_genes,
                alignment_info,
                selected_exprs,
            ) = single_outputs
        else:
            (
                features_c,
                features_common,
                zc_init,
                dc_init,
                ec_init,
                zf_init,
                df_init,
                ef_init,
                mc,
                mf,
                n_view,
                num_sample,
                labels,
                n_clusters,
                common_genes,
                alignment_info,
            ) = single_outputs
    elif data_type == "multi":
        multi_outputs = features_to_adj_omics(
            dataset_name,
            args.data_root,
            feature_align_mode=getattr(args, "feature_align_mode", "auto"),
            latent_feature_dim=getattr(args, "latent_feature_dim", 2000),
            anchor_multiplier=getattr(args, "anchor_multiplier", None),
            mc_multiplier=getattr(args, "mc_multiplier", None),
            mf_multiplier=getattr(args, "mf_multiplier", None),
            return_expression=need_expression,
        )
        (
            features_c,
            features_common,
            zc_init,
            dc_init,
            ec_init,
            zf_init,
            df_init,
            ef_init,
            mc,
            mf,
            n_view,
            num_sample,
            labels,
            n_clusters,
            common_genes,
            alignment_info,
        ) = multi_outputs[:16]
        if need_expression:
            selected_exprs = multi_outputs[16]
    else:
        raise ValueError("data_type must be one of {'single', 'multi'}, got {}.".format(data_type))

    standardized_exprs = None
    if selected_exprs is not None:
        standardized_exprs = (features_common[0].T if data_type == "single" else
                              StandardScaler().fit_transform(selected_exprs).astype(np.float32))
    marker_controller = None
    if lambda_marker > 0:
        if selected_exprs is None or not alignment_info.get("is_gene_level", False):
            raise ValueError("Marker loss requires named, gene-aligned expression; multi-omics latent or order-only alignment is unsupported. Use --lambda_marker 0 for clustering only.")
        if not (bool(args.enable_feature_aux) or bool(args.enable_cf_injection)):
            raise ValueError("Marker loss requires the feature branch.")
        marker_controller = MarkerConsistency(
            selected_exprs, standardized_exprs, common_genes, n_clusters, seed,
            top_l=getattr(args, "marker_top_l", 50),
            pair_count=getattr(args, "marker_pairs", 4096),
            warmup=getattr(args, "marker_warmup", 1),
            refresh_interval=getattr(args, "marker_refresh", 5),
        )
    history, training_state = [], {}

    model = DBONet_with_E(
        mc,
        mf,
        n_view,
        args.block,
        args.thre1,
        args.thre2,
        zc_init,
        dc_init,
        ec_init,
        zf_init,
        df_init,
        ef_init,
        device,
        alpha_temperature=args.alpha_temperature,
        enable_d_perturb=args.enable_d_perturb,
        d_perturb_scale=args.d_perturb_scale,
        enable_d_f_perturb=args.enable_d_f_perturb,
        d_f_perturb_scale=args.d_f_perturb_scale,
        theta_f1=args.thre_f1,
        theta_f2=args.thre_f2,
        enable_cf_injection=args.enable_cf_injection,
        beta_cf_init=args.beta_cf_init,
        beta_cf_max=args.beta_cf_max,
        cf_gate_init=args.cf_gate_init,
        feature_to_cell_bridge=args.feature_to_cell_bridge,
    )

    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, betas=(0.90, 0.92), eps=0.01, weight_decay=0.15
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.3, patience=15, verbose=True, min_lr=1e-8
    )

    best_by_metric, total_time = train(
        features_c,
        features_common,
        args.epoch,
        args.block,
        labels,
        n_view,
        n_clusters,
        model,
        optimizer,
        scheduler,
        device,
        args.lambda_view_entropy,
        bool(args.enable_feature_aux),
        bool(args.enable_d_perturb),
        args.lambda_d_pert,
        args.lambda_f,
        args.lambda_cf,
        args.beta_f,
        args.eta_f1,
        capture_artifacts=collect_artifacts,
        marker_controller=marker_controller,
        lambda_marker=lambda_marker,
        history=history,
        training_state=training_state,
    )

    selection_raw = str(getattr(args, "selection_metric", "ACC"))
    selection_metric = next((name for name in SCORE_NAMES if name.lower() == selection_raw.lower()), selection_raw)
    if selection_metric not in SCORE_NAMES:
        raise ValueError(f"selection_metric must be one of {SCORE_NAMES}, got {selection_metric}.")

    result_dict = dict(best_by_metric[selection_metric]["result"])
    result_dict["n_clusters"] = int(n_clusters)
    result_dict["mc"] = int(mc)
    result_dict["mf"] = int(mf)
    result_dict["time_seconds"] = round(total_time, 6)
    result_dict["selection_metric"] = selection_metric
    result_dict["best_epoch"] = int(best_by_metric[selection_metric]["best_epoch"])
    result_dict["best_by_metric"] = {
        metric_name: {
            "best_epoch": int(metric_info["best_epoch"]),
            "result": dict(metric_info["result"]),
        }
        for metric_name, metric_info in best_by_metric.items()
    }
    result_dict["metrics"] = {
        name: {"mean": float(value[0]), "std": float(value[1])}
        for name, value in zip(SCORE_NAMES, best_by_metric[selection_metric]["metrics"])
    }
    result_dict["loss_history"] = history
    result_dict.update(training_state)
    if collect_artifacts:
        artifacts = best_by_metric[selection_metric].get("snapshot") or {}
        artifacts["dataset"] = dataset_name
        artifacts["data_type"] = data_type
        artifacts["selection_metric"] = selection_metric
        artifacts["best_epoch"] = int(best_by_metric[selection_metric]["best_epoch"])
        artifacts["selected_genes"] = common_genes
        artifacts["labels"] = labels
        artifacts["n_clusters"] = int(n_clusters)
        artifacts["label_mapping"] = alignment_info.get("label_mapping")
        artifacts["selected_expression"] = selected_exprs
        artifacts["alignment_info"] = alignment_info
        artifacts["standardized_expression"] = standardized_exprs
        artifacts["source_gene_indices"] = alignment_info.get("rna_source_feature_indices")
        artifacts["marker_state_at_end"] = training_state["marker_state_at_end"]
        return result_dict, artifacts
    return result_dict
    
SCORE_NAMES = ["ACC", "NMI", "Purity", "ARI", "Fscore", "Precision", "Recall"]
SNAPSHOT_KEYS = ["Zc", "Zf", "Df", "Hf", "Zc_hat", "gate_cf", "alpha_c", "alpha_f"]