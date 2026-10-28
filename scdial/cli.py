import argparse
import math
from pathlib import Path

from .data import discover_multi_omics_datasets, discover_single_omics_datasets
from .pipeline import run_dataset


def parse_selected_datasets(all_datasets, datasets_arg):
    if datasets_arg is None or not datasets_arg.strip():
        return [all_datasets[0]]
    if datasets_arg.strip().lower() == "all":
        return all_datasets
    valid = {name.lower(): name for name in all_datasets}
    selected, invalid = [], []
    for token in datasets_arg.split(","):
        name = token.strip()
        if not name:
            continue
        if name.lower() in valid:
            selected.append(valid[name.lower()])
        else:
            invalid.append(name)
    if invalid:
        raise ValueError("Invalid dataset(s): {}. Available: {} or 'all'.".format(
            ", ".join(invalid), ", ".join(all_datasets)))
    return list(dict.fromkeys(selected))


def build_parser():
    parser = argparse.ArgumentParser(description="Run scDIAL on a single-cell dataset.")
    parser.add_argument("--data_root", type=str)
    parser.add_argument("--data_type", choices=["single", "multi"])
    parser.add_argument("--datasets", help="Dataset names separated by comma, or 'all'.")
    parser.add_argument("--device", default="0", help="CUDA device id or 'cpu'.")
    parser.add_argument("--selection_metric", choices=["ACC", "NMI", "Purity", "ARI", "Fscore", "Precision", "Recall"])
    parser.add_argument("--seed", type=int)
    parser.add_argument("--block", type=int)
    parser.add_argument("--thre1", type=float, default=0.01)
    parser.add_argument("--thre2", type=float, default=0.001)
    parser.add_argument("--lambda_view_entropy", type=float)
    parser.add_argument("--enable_feature_aux", type=int, choices=[0, 1])
    parser.add_argument("--enable_d_perturb", type=int, choices=[0, 1])
    parser.add_argument("--enable_anchor_perturb", type=int, choices=[0, 1], default=None)
    parser.add_argument("--d_perturb_scale", type=float)
    parser.add_argument("--lambda_d_pert", type=float)
    parser.add_argument("--alpha_temperature", type=float, default=1.0)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--epoch", type=int)
    parser.add_argument("--thre_f1", type=float, default=0.01)
    parser.add_argument("--thre_f2", type=float, default=0.001)
    parser.add_argument("--beta_f", type=float)
    parser.add_argument("--eta_f1", type=float)
    parser.add_argument("--enable_d_f_perturb", type=int, choices=[0, 1])
    parser.add_argument("--d_f_perturb_scale", type=float)
    parser.add_argument("--lambda_f", type=float)
    parser.add_argument("--lambda_cf", type=float)
    parser.add_argument("--enable_cf_injection", type=int, choices=[0, 1])
    parser.add_argument("--beta_cf_init", type=float)
    parser.add_argument("--cf_gate_init", type=float, default=None)
    parser.add_argument("--beta_cf_max", type=float, default=0.6)
    parser.add_argument("--feature_to_cell_bridge", choices=["direct", "x"])
    parser.add_argument("--anchor_multiplier", type=int)
    parser.add_argument("--mc_multiplier", type=int, default=None)
    parser.add_argument("--mf_multiplier", type=int, default=None)
    parser.add_argument("--label_key", default="cell_type1")
    parser.add_argument("--gene_set", choices=["seurat", "scmap", "top_variable", "all"])
    parser.add_argument("--n_top_genes", type=int)
    parser.add_argument("--log1p", choices=["auto", "always", "never"])
    parser.add_argument("--feature_align_mode", choices=["auto", "names", "order", "latent"])
    parser.add_argument("--latent_feature_dim", type=int, default=2000)
    parser.add_argument("--output_dir", type=Path)
    parser.add_argument("--lambda_marker", type=float)
    parser.add_argument("--marker_top_l", type=int)
    parser.add_argument("--marker_pairs", type=int)
    parser.add_argument("--marker_warmup", type=int)
    parser.add_argument("--marker_refresh", type=int)
    parser.add_argument("--fa_top_k", default="3,5,10,20")
    parser.add_argument("--fa_top_q", default="1,2,4")
    parser.add_argument("--random_sets", type=int, default=100)
    parser.add_argument("--permutations", type=int, default=1000)
    parser.add_argument("--threads", type=int)
    parser.add_argument("--run_control", action="store_true")
    parser.add_argument("--no_control", dest="run_control", action="store_false")
    parser.set_defaults(**DEFAULT_CONFIG)
    return parser


def parse_args(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.epoch <= 0 or args.threads <= 0:
        parser.error("--epoch and --threads must be positive")
    if not math.isfinite(args.lambda_marker) or args.lambda_marker < 0:
        parser.error("--lambda_marker must be finite and nonnegative")
    if min(args.marker_top_l, args.marker_pairs, args.marker_refresh, args.random_sets, args.permutations) <= 0 or args.marker_warmup < 0:
        parser.error("Marker/evaluation counts must be positive and warmup nonnegative")
    for key in ("fa_top_k", "fa_top_q"):
        try:
            values = sorted({int(value.strip()) for value in getattr(args, key).split(",")})
            if not values or min(values) <= 0:
                raise ValueError
            setattr(args, key, values)
        except ValueError:
            parser.error("--{} must contain comma-separated positive integers".format(key))
    if args.enable_anchor_perturb is not None:
        args.enable_d_perturb = args.enable_anchor_perturb
        args.enable_d_f_perturb = args.enable_anchor_perturb
    args.data_root = str(Path(args.data_root).resolve())
    args.output_dir = Path(args.output_dir).resolve()
    return args


def main(argv=None):
    import torch
    args = parse_args(argv)
    torch.set_num_threads(args.threads)
    discovery = discover_single_omics_datasets if args.data_type == "single" else discover_multi_omics_datasets
    available = discovery(args.data_root)
    for dataset in parse_selected_datasets(available, args.datasets):
        run_dataset(dataset, args)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = {
    "data_root": str(PROJECT_ROOT), "data_type": "single", "datasets": None,
    "selection_metric": "ACC", "seed": 42, "block": 2,
    "lambda_view_entropy": 0.05, "enable_feature_aux": 1, "enable_d_perturb": 1,
    "d_perturb_scale": 0.05, "lambda_d_pert": 0.05, "lr": 0.01, "epoch": 100,
    "beta_f": 0.05, "eta_f1": 0.05, "enable_d_f_perturb": 1,
    "d_f_perturb_scale": 0.05, "lambda_f": 0.03, "lambda_cf": 0.02,
    "enable_cf_injection": 1, "beta_cf_init": 0.02, "feature_to_cell_bridge": "direct",
    "anchor_multiplier": 1, "gene_set": "seurat", "n_top_genes": 2000, "log1p": "auto",
    "feature_align_mode": "auto", "output_dir": PROJECT_ROOT / "result",
    "lambda_marker": 0.01, "marker_top_l": 50, "marker_pairs": 4096,
    "marker_warmup": 1, "marker_refresh": 5, "threads": 4, "run_control": False,
}

if __name__ == "__main__":
    main()
