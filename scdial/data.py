from pathlib import Path

import h5py
import numpy as np
import scipy.sparse as sp
from scipy.linalg import svd
from scipy.sparse.linalg import svds
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def _decode_string_array(arr):
    values = []
    for item in np.asarray(arr).reshape(-1):
        if isinstance(item, bytes):
            values.append(item.decode("utf-8", errors="ignore"))
        else:
            values.append(str(item))
    return values


def _single_omics_files(data_root):
    data_root = Path(data_root)
    if not data_root.exists():
        raise FileNotFoundError(f"Data root not found: {data_root}")
    paths = set(data_root.glob("*/data.h5"))
    paths.update(data_root.glob("*/*/data.h5"))
    return sorted(path for path in paths if path.is_file())


def discover_single_omics_datasets(data_root):
    datasets = list(dict.fromkeys(path.parent.name for path in _single_omics_files(data_root)))

    if not datasets:
        raise RuntimeError(f"No usable single-omics data.h5 files found under: {data_root}")
    return datasets


def discover_multi_omics_datasets(data_root):
    data_root = Path(data_root)
    if not data_root.exists():
        raise FileNotFoundError(f"Data root not found: {data_root}")

    datasets = []
    for item in sorted(data_root.iterdir()):
        if not item.is_dir():
            continue
        has_rna = bool(list(item.glob("*_RNA.h5")))
        has_atac = bool(list(item.glob("*_ATAC.h5")))
        has_label = bool(list(item.glob("*cell_label*.h5")))
        if has_rna and has_atac and has_label:
            datasets.append(item.name)

    if not datasets:
        raise RuntimeError(f"No usable multi-omics datasets found under: {data_root}")
    return datasets


def resolve_single_omics_file(dataset, data_root):
    data_root = Path(data_root)
    dataset_path = Path(dataset)
    if dataset_path.exists() and dataset_path.is_file():
        return dataset_path
    if dataset_path.exists() and dataset_path.is_dir():
        h5_path = dataset_path / "data.h5"
        if h5_path.exists():
            return h5_path

    candidates = {}
    for h5_path in _single_omics_files(data_root):
        candidates[h5_path.parent.name.lower()] = h5_path

    match = candidates.get(str(dataset).lower())
    if match is None:
        available = ", ".join(sorted(path.parent.name for path in candidates.values()))
        raise FileNotFoundError(f"Dataset '{dataset}' not found. Available: {available}")
    return match


def read_h5_overview(h5_path, label_key="cell_type1"):
    h5_path = Path(h5_path)
    with h5py.File(h5_path, "r") as f:
        shape = tuple(np.array(f["exprs/shape"]).astype(int).tolist())
        n_obs = len(f["obs_names"]) if "obs_names" in f else shape[0]
        n_var = len(f["var_names"]) if "var_names" in f else shape[1]
        has_label = f"obs/{label_key}" in f
        n_clusters = None
        if has_label:
            labels = _decode_string_array(f[f"obs/{label_key}"][()])
            n_clusters = len(np.unique(labels))
        gene_sets = []
        for key in ("seurat_genes", "scmap_genes"):
            if f"uns/{key}" in f:
                gene_sets.append(f"{key}:{len(f[f'uns/{key}'])}")

    return {
        "dataset": h5_path.parent.name,
        "path": str(h5_path),
        "shape": shape,
        "n_obs": n_obs,
        "n_var": n_var,
        "has_label": has_label,
        "n_clusters": n_clusters,
        "gene_sets": gene_sets,
    }


def _read_sparse_group(group):
    keys = set(group.keys())
    if not {"data", "indices", "indptr", "shape"}.issubset(keys):
        raise ValueError("Sparse group must contain data, indices, indptr, and shape.")

    data = np.array(group["data"])
    indices = np.array(group["indices"])
    indptr = np.array(group["indptr"])
    shape = tuple(np.array(group["shape"]).astype(int).tolist())

    encoding = group.attrs.get("encoding-type", "csr_matrix")
    if isinstance(encoding, bytes):
        encoding = encoding.decode("utf-8", errors="ignore")

    if encoding == "csc_matrix":
        return sp.csc_matrix((data, indices, indptr), shape=shape).tocsr()
    return sp.csr_matrix((data, indices, indptr), shape=shape)


def read_exprs_matrix(h5_path):
    with h5py.File(str(h5_path), "r") as f:
        if "exprs" not in f:
            raise KeyError(f"No 'exprs' sparse group found in {h5_path}")
        exprs = _read_sparse_group(f["exprs"])
    return exprs.astype(np.float32).tocsr()


def read_var_names(h5_path, n_features):
    with h5py.File(str(h5_path), "r") as f:
        if "var_names" not in f:
            return [f"gene_{i}" for i in range(n_features)]
        names = _decode_string_array(f["var_names"][()])

    if len(names) != n_features:
        return [f"gene_{i}" for i in range(n_features)]
    return names


def read_labels(h5_path, label_key="cell_type1"):
    label_path = f"obs/{label_key}"
    with h5py.File(str(h5_path), "r") as f:
        if label_path not in f:
            available = sorted(f["obs"].keys()) if "obs" in f else []
            raise KeyError(
                f"Label key '{label_key}' not found in {h5_path}. "
                f"Available obs keys: {available}"
            )
        raw_labels = _decode_string_array(f[label_path][()])

    mapping = {}
    labels = []
    for value in raw_labels:
        if value not in mapping:
            mapping[value] = len(mapping)
        labels.append(mapping[value])
    return np.asarray(labels, dtype=np.int64), mapping


def _read_uns_gene_list(h5_path, key):
    path = f"uns/{key}_genes"
    with h5py.File(str(h5_path), "r") as f:
        if path not in f:
            return None
        return _decode_string_array(f[path][()])


def _first_index_by_name(names):
    mapping = {}
    for idx, name in enumerate(names):
        if name not in mapping:
            mapping[name] = idx
    return mapping


def _sparse_column_variance(x_mat):
    x_mat = x_mat.tocsr().astype(np.float32)
    mean = np.asarray(x_mat.mean(axis=0)).ravel()
    squared = x_mat.copy()
    squared.data **= 2
    mean_sq = np.asarray(squared.mean(axis=0)).ravel()
    return np.maximum(mean_sq - mean**2, 0.0)


def _top_variable_gene_indices(x_mat, n_top_genes):
    n_top = min(int(n_top_genes), x_mat.shape[1])
    if n_top <= 0:
        raise ValueError(f"n_top_genes must be positive, got {n_top_genes}")
    variances = _sparse_column_variance(x_mat)
    if n_top == x_mat.shape[1]:
        order = np.argsort(-variances)
    else:
        top = np.argpartition(-variances, n_top - 1)[:n_top]
        order = top[np.argsort(-variances[top])]
    return order.astype(np.int64), "top_variable"


def _select_gene_indices(h5_path, x_mat, var_names, gene_set, n_top_genes):
    gene_set = str(gene_set).lower()
    if gene_set == "all":
        return np.arange(x_mat.shape[1], dtype=np.int64), var_names, "all"

    fallback_orders = {
        "seurat": ["seurat", "scmap", "top_variable"],
        "scmap": ["scmap", "seurat", "top_variable"],
        "top_variable": ["top_variable"],
    }
    if gene_set not in fallback_orders:
        raise ValueError("gene_set must be one of {'seurat', 'scmap', 'top_variable', 'all'}.")

    name_to_idx = _first_index_by_name(var_names)
    for source in fallback_orders[gene_set]:
        if source == "top_variable":
            indices, used_source = _top_variable_gene_indices(x_mat, n_top_genes)
            names = [var_names[int(i)] for i in indices]
            return indices, names, used_source

        requested_genes = _read_uns_gene_list(h5_path, source)
        if not requested_genes:
            continue

        indices = []
        seen = set()
        for gene in requested_genes:
            idx = name_to_idx.get(gene)
            if idx is None or idx in seen:
                continue
            seen.add(idx)
            indices.append(idx)

        if not indices:
            continue

        if n_top_genes is not None and len(indices) > int(n_top_genes):
            indices = indices[: int(n_top_genes)]

        selected_names = [var_names[int(i)] for i in indices]
        return np.asarray(indices, dtype=np.int64), selected_names, source

    indices, used_source = _top_variable_gene_indices(x_mat, n_top_genes)
    selected_names = [var_names[int(i)] for i in indices]
    return indices, selected_names, used_source


def _should_log1p(x_mat, mode):
    mode = str(mode).lower()
    if mode == "always":
        return True
    if mode == "never":
        return False
    if mode != "auto":
        raise ValueError("log1p must be one of {'auto', 'always', 'never'}.")

    data = x_mat.data
    if data.size == 0:
        return False

    step = max(1, data.size // 200000)
    sample = np.asarray(data[::step], dtype=np.float64)
    q99 = float(np.quantile(sample, 0.99))
    max_val = float(np.max(sample))
    integer_like = float(np.mean(np.abs(sample - np.round(sample)) < 1e-6)) > 0.95
    return q99 > 20.0 or max_val > 100.0 or (integer_like and q99 > 5.0)


def _standardize_cells_by_gene(x_cells_by_genes):
    scaler = StandardScaler(with_mean=True, with_std=True)
    x_norm = scaler.fit_transform(x_cells_by_genes).astype(np.float32)
    return np.nan_to_num(x_norm, nan=0.0, posinf=0.0, neginf=0.0)


def _validate_svds_k(matrix_shape, k, context):
    min_dim = min(matrix_shape)
    if k <= 0 or k >= min_dim:
        raise ValueError(
            f"{context}: svds requires 0 < k < min(shape); got k={k}, shape={matrix_shape}."
        )


def _run_svds(matrix, k, context):
    _validate_svds_k(matrix.shape, k, context)
    try:
        return svds(matrix, k=k, random_state=12)
    except TypeError:
        return svds(matrix, k=k)


def _svd_uv(matrix):
    u_mat, _, vt_mat = svd(matrix, full_matrices=False)
    return np.dot(u_mat, vt_mat).astype(np.float32)


def _one_hot_from_labels(labels, n_clusters):
    z_mat = np.zeros((n_clusters, labels.shape[0]), dtype=np.float32)
    for idx, label in enumerate(labels):
        z_mat[int(label), idx] = 1.0
    return z_mat


def _compute_anchor_sizes(
    n_clusters,
    n_cells,
    n_features,
    anchor_multiplier=None,
    mc_multiplier=None,
    mf_multiplier=None,
):
    explicit_split_multiplier = mc_multiplier is not None or mf_multiplier is not None
    explicit_anchor_multiplier = anchor_multiplier is not None and not explicit_split_multiplier

    if explicit_split_multiplier:
        if mc_multiplier is None or mf_multiplier is None:
            raise ValueError(
                "mc_multiplier and mf_multiplier must be provided together "
                "when using split anchor multipliers."
            )
        mc = int(n_clusters * int(mc_multiplier))
        mf = int(n_clusters * int(mf_multiplier))
    elif explicit_anchor_multiplier:
        multiplier = int(anchor_multiplier)
        if multiplier <= 0:
            raise ValueError(f"anchor_multiplier must be positive, got {anchor_multiplier}.")
        mc = int(n_clusters * multiplier)
        mf = int(mc)
    else:
        mc = int(n_clusters)
        mf = int(mc)

    max_k = min(n_cells, n_features) - 1
    if mc <= 0 or mf <= 0:
        raise ValueError(f"Anchor sizes must be positive; got mc={mc}, mf={mf}.")
    if mc > max_k or mf > max_k:
        raise ValueError(
            f"Anchor sizes exceed svds limit for data shape cells={n_cells}, features={n_features}. "
            f"Got mc={mc}, mf={mf}, max allowed={max_k}."
        )
    return mc, mf


def features_to_adj_single_omics(
    dataset,
    data_root,
    label_key="cell_type1",
    gene_set="seurat",
    n_top_genes=2000,
    log1p="auto",
    anchor_multiplier=None,
    mc_multiplier=None,
    mf_multiplier=None,
    return_expression=False,
):
    h5_path = resolve_single_omics_file(dataset, data_root)
    labels, label_mapping = read_labels(h5_path, label_key=label_key)
    n_cells = int(labels.shape[0])

    exprs = read_exprs_matrix(h5_path)
    if exprs.shape[0] != n_cells:
        if exprs.shape[1] == n_cells:
            exprs = exprs.T.tocsr()
        else:
            raise ValueError(
                f"Expression shape {exprs.shape} does not match label count {n_cells}."
            )

    var_names = read_var_names(h5_path, exprs.shape[1])
    apply_log = _should_log1p(exprs, log1p)
    if apply_log:
        exprs = exprs.copy()
        exprs.data = np.log1p(exprs.data).astype(np.float32)

    gene_indices, selected_genes, gene_source = _select_gene_indices(
        h5_path, exprs, var_names, gene_set, n_top_genes
    )
    exprs = exprs[:, gene_indices].astype(np.float32).tocsr()

    exprs_dense = exprs.toarray().astype(np.float32)
    x_dense = exprs_dense.copy()
    x_dense = _standardize_cells_by_gene(x_dense)
    x_view = x_dense.T.astype(np.float32)

    n_view = 1
    n_clusters = len(set(np.asarray(labels)))
    mc, mf = _compute_anchor_sizes(
        n_clusters,
        n_cells=x_view.shape[1],
        n_features=x_view.shape[0],
        anchor_multiplier=anchor_multiplier,
        mc_multiplier=mc_multiplier,
        mf_multiplier=mf_multiplier,
    )

    u_cell, _, _ = _run_svds(x_view.T, mc, "Cell anchor initialization")
    np.random.seed(12)
    kmeans_c = KMeans(n_clusters=mc, max_iter=100, n_init=10, random_state=12)
    cell_anchor_labels = kmeans_c.fit_predict(u_cell)
    zc_init = _one_hot_from_labels(cell_anchor_labels, mc)

    dc_init = []
    ec_init = []
    c_mat = np.dot(x_view, zc_init.T)
    d_view = _svd_uv(c_mat)
    e_view = x_view - np.dot(d_view, zc_init)
    dc_init.append(d_view.astype(np.float32))
    ec_init.append(e_view.astype(np.float32))

    u_feature, _, _ = _run_svds(x_view, mf, "Feature anchor initialization")
    kmeans_f = KMeans(n_clusters=mf, max_iter=100, n_init=10, random_state=12)
    feature_anchor_labels = kmeans_f.fit_predict(u_feature)
    zf_init = _one_hot_from_labels(feature_anchor_labels, mf)

    df_init = []
    ef_init = []
    x_feature = x_view.T.astype(np.float32)
    c_mat = np.dot(x_feature, zf_init.T)
    d_feature = _svd_uv(c_mat)
    e_feature = x_feature - np.dot(d_feature, zf_init)
    df_init.append(d_feature.astype(np.float32))
    ef_init.append(e_feature.astype(np.float32))

    info = {
        "dataset_path": str(h5_path),
        "used_mode": "single_rna",
        "is_gene_level": True,
        "label_key": label_key,
        "label_mapping": label_mapping,
        "gene_set_requested": gene_set,
        "gene_set_used": gene_source,
        "log1p_requested": log1p,
        "log1p_applied": apply_log,
        "message": "Single RNA view is used for both cell and feature branches.",
    }

    result = (
        [x_view],
        [x_view],
        zc_init,
        dc_init,
        ec_init,
        zf_init,
        df_init,
        ef_init,
        mc,
        mf,
        n_view,
        n_cells,
        labels,
        n_clusters,
        selected_genes,
        info,
    )

    if return_expression:
        return result + (exprs_dense,)
    return result


def _iter_h5_datasets(h5_obj, prefix=""):
    for key in h5_obj.keys():
        item = h5_obj[key]
        path = f"{prefix}/{key}" if prefix else key
        if isinstance(item, h5py.Dataset):
            yield path, item
        elif isinstance(item, h5py.Group):
            yield from _iter_h5_datasets(item, path)


def _iter_matrix_candidate_datasets(h5_obj, prefix=""):
    for path, ds in _iter_h5_datasets(h5_obj, prefix):
        if np.issubdtype(ds.dtype, np.number) or ds.dtype.kind in {"S", "U", "O"}:
            yield path, ds


def _try_read_csr_group_dense(group):
    keys = set(group.keys())
    if not {"data", "indices", "indptr", "shape"}.issubset(keys):
        return None

    data = np.array(group["data"])
    indices = np.array(group["indices"])
    indptr = np.array(group["indptr"])
    shape = tuple(np.array(group["shape"]).astype(int).tolist())
    return sp.csr_matrix((data, indices, indptr), shape=shape).toarray()


def _read_multi_h5_matrix(file_path):
    candidates = []

    with h5py.File(str(file_path), "r") as f:
        for key in f.keys():
            item = f[key]
            if isinstance(item, h5py.Group):
                csr = _try_read_csr_group_dense(item)
                if csr is not None:
                    arr = np.asarray(csr)
                    if arr.ndim == 2:
                        candidates.append((arr.size, f"{key}(csr)", arr))

        for path, ds in _iter_matrix_candidate_datasets(f):
            try:
                arr = np.array(ds)
            except Exception:
                continue
            if arr.ndim == 2 and np.issubdtype(arr.dtype, np.number):
                candidates.append((arr.size, path, arr))
            elif arr.ndim > 2 and np.issubdtype(arr.dtype, np.number):
                flat = arr.reshape(arr.shape[0], -1)
                candidates.append((flat.size, path, flat))

    if not candidates:
        raise ValueError(f"No usable matrix dataset found in: {file_path}")

    _, picked_key, picked = max(candidates, key=lambda x: x[0])
    return np.asarray(picked)


def _read_multi_h5_feature_names(file_path, n_features):
    name_tokens = ("gene", "genes", "feature", "features", "var", "name", "names")
    candidates = []

    with h5py.File(str(file_path), "r") as f:
        for path, ds in _iter_h5_datasets(f):
            try:
                arr = np.array(ds)
            except Exception:
                continue
            arr = np.squeeze(arr)
            if arr.ndim != 1 or arr.shape[0] != n_features:
                continue
            if arr.dtype.kind not in {"S", "U", "O"}:
                continue
            score = 1 if any(token in path.lower() for token in name_tokens) else 0
            candidates.append((score, path, _decode_string_array(arr)))

    if not candidates:
        return None

    candidates.sort(key=lambda item: item[0], reverse=True)
    _, picked_key, names = candidates[0]
    return names, f"h5:{picked_key}"


def _read_sidecar_feature_names(dataset_dir, view_name, n_features):
    view_name = view_name.upper()
    patterns = [
        f"*_{view_name}_features.tsv",
        f"*_{view_name}_features.txt",
        f"*_{view_name}_features.csv",
        f"*_{view_name}_genes.tsv",
        f"*_{view_name}_genes.txt",
        f"*_{view_name}_genes.csv",
        f"{view_name}_features.tsv",
        f"{view_name}_features.txt",
        f"{view_name}_features.csv",
        f"{view_name}_genes.tsv",
        f"{view_name}_genes.txt",
        f"{view_name}_genes.csv",
    ]

    candidates = []
    for pattern in patterns:
        candidates.extend(sorted(Path(dataset_dir).glob(pattern)))

    seen = set()
    for path in candidates:
        if path in seen:
            continue
        seen.add(path)

        rows = []
        with open(path, "r", encoding="utf-8-sig", errors="ignore") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                if "\t" in line:
                    token = line.split("\t")[0].strip()
                elif "," in line:
                    token = line.split(",")[0].strip()
                else:
                    token = line.split()[0].strip()
                if token:
                    rows.append(token)

        if len(rows) == n_features + 1 and rows[0].lower() in {"gene", "genes", "feature", "features", "name", "names"}:
            rows = rows[1:]

        if len(rows) != n_features:
            continue

        return rows, f"sidecar:{path.name}"

    return None, None


def _read_common_gene_names(dataset_dir, n_features):
    patterns = [
        "*_common_genes.tsv",
        "*_common_genes.txt",
        "*_common_genes.csv",
        "common_genes.tsv",
        "common_genes.txt",
        "common_genes.csv",
    ]

    candidates = []
    for pattern in patterns:
        candidates.extend(sorted(Path(dataset_dir).glob(pattern)))

    for path in candidates:
        rows = []
        with open(path, "r", encoding="utf-8-sig", errors="ignore") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                if "\t" in line:
                    token = line.split("\t")[0].strip()
                elif "," in line:
                    token = line.split(",")[0].strip()
                else:
                    token = line.split()[0].strip()
                if token:
                    rows.append(token)

        if len(rows) == n_features + 1 and rows[0].lower() in {"gene", "genes", "feature", "features", "name", "names"}:
            rows = rows[1:]

        if len(rows) != n_features:
            continue

        return rows, f"common_genes:{path.name}"

    return None, None


def _read_multi_feature_names(dataset_dir, h5_path, view_name, n_features):
    names_and_source = _read_multi_h5_feature_names(h5_path, n_features)
    if names_and_source is not None:
        return names_and_source
    names, source = _read_sidecar_feature_names(dataset_dir, view_name, n_features)
    if names is not None:
        return names, source
    return _read_common_gene_names(dataset_dir, n_features)


def _read_multi_h5_labels(file_path):
    candidates = []

    with h5py.File(str(file_path), "r") as f:
        for path, ds in _iter_matrix_candidate_datasets(f):
            try:
                arr = np.array(ds)
            except Exception:
                continue
            arr = np.squeeze(arr)
            if arr.ndim == 1:
                candidates.append((arr.size, path, arr))
            elif arr.ndim == 2 and 1 in arr.shape:
                candidates.append((arr.size, path, arr.reshape(-1)))

    if not candidates:
        raise ValueError(f"No usable 1D labels found in: {file_path}")

    _, picked_key, labels = max(candidates, key=lambda x: x[0])
    labels = np.asarray(labels)

    if labels.dtype.kind in {"S", "U", "O"}:
        labels = labels.astype(str)
        mapping = {value: idx for idx, value in enumerate(np.unique(labels))}
        labels = np.array([mapping[value] for value in labels], dtype=np.int64)
    else:
        labels = labels.astype(np.int64)

    uniq_vals = np.unique(labels)
    if uniq_vals.min() != 0:
        mapping = {value: idx for idx, value in enumerate(uniq_vals)}
        labels = np.array([mapping[value] for value in labels], dtype=np.int64)

    return labels


def _resolve_omics_files(dataset_dir):
    dataset_dir = Path(dataset_dir)
    rna_files = sorted(dataset_dir.glob("*_RNA.h5"))
    atac_files = sorted(dataset_dir.glob("*_ATAC.h5"))
    label_files = sorted(dataset_dir.glob("*cell_label*.h5"))

    if not rna_files:
        raise FileNotFoundError(f"No RNA .h5 found in: {dataset_dir}")
    if not atac_files:
        raise FileNotFoundError(f"No ATAC .h5 found in: {dataset_dir}")
    if not label_files:
        raise FileNotFoundError(f"No label .h5 found in: {dataset_dir}")

    return rna_files[0], atac_files[0], label_files[0]


def _ensure_samples_by_features(x, n_samples, view_name):
    x = np.asarray(x)
    if x.ndim != 2:
        x = x.reshape(x.shape[0], -1)

    if x.shape[0] == n_samples:
        return x
    if x.shape[1] == n_samples:
        return x.T

    raise ValueError(
        f"{view_name} shape {x.shape} does not match label count {n_samples}; "
        "cannot infer sample dimension."
    )


def _standardize_rows(x_mat):
    mean = np.mean(x_mat, axis=1, keepdims=True)
    std = np.std(x_mat, axis=1, keepdims=True)
    std[std < 1e-6] = 1.0
    return ((x_mat - mean) / std).astype(np.float32)


def _build_latent_feature_views(x_list, latent_feature_dim):
    min_allowed = min(min(x.shape) - 1 for x in x_list)
    g_latent = int(min(latent_feature_dim, min_allowed))
    if g_latent <= 0:
        raise ValueError(
            "Latent feature alignment requires every view to have min(shape) > 1; "
            f"got shapes {[x.shape for x in x_list]}."
        )

    x_common = []
    for view_idx, x_mat in enumerate(x_list):
        _, singular_values, vt_mat = _run_svds(
            x_mat,
            g_latent,
            f"Latent feature alignment view {view_idx}",
        )
        latent = singular_values[:, None] * vt_mat
        latent = _standardize_rows(latent)
        x_common.append(latent.astype(np.float32))

    common_features = [f"latent_feature_{i}" for i in range(g_latent)]
    info = {
        "requested_mode": "latent",
        "used_mode": "latent",
        "g": g_latent,
        "is_gene_level": False,
        "feature_name_sources": [],
        "message": "Latent fallback is trainable but not a gene-level biological intersection.",
    }
    return x_common, common_features, info


def _build_common_gene_views(
    x_list,
    feature_names_list=None,
    feature_name_sources=None,
    align_mode="auto",
    latent_feature_dim=2000,
):
    align_mode = str(align_mode).lower()
    valid_modes = {"auto", "names", "order", "latent"}
    if align_mode not in valid_modes:
        raise ValueError(f"feature_align_mode must be one of {sorted(valid_modes)}, got {align_mode}.")

    has_all_names = bool(feature_names_list) and all(names is not None for names in feature_names_list)

    if align_mode in {"auto", "names"} and has_all_names:
        name_maps = [_first_index_by_name(names) for names in feature_names_list]
        common = set(name_maps[0].keys())
        for mapping in name_maps[1:]:
            common &= set(mapping.keys())

        ordered_common = [name for name in feature_names_list[0] if name in common]
        ordered_common = list(dict.fromkeys(ordered_common))
        if not ordered_common:
            raise ValueError("No common gene names found across views.")

        x_common = []
        for x_mat, mapping in zip(x_list, name_maps):
            idx = [mapping[name] for name in ordered_common]
            x_common.append(x_mat[idx, :].astype(np.float32))

        info = {
            "requested_mode": align_mode,
            "used_mode": "names",
            "g": len(ordered_common),
            "is_gene_level": True,
            "feature_name_sources": feature_name_sources or [],
            "message": "Feature views aligned by real feature names.",
        }
        return x_common, ordered_common, info

    if align_mode == "names":
        raise ValueError(
            "feature_align_mode='names' requires feature names for every view. "
            f"Sources found: {feature_name_sources or []}"
        )

    if align_mode == "latent":
        x_common, common_features, info = _build_latent_feature_views(x_list, latent_feature_dim)
        info["requested_mode"] = align_mode
        return x_common, common_features, info

    feature_dims = [x.shape[0] for x in x_list]
    if align_mode in {"auto", "order"} and len(set(feature_dims)) == 1:
        g_dim = feature_dims[0]
        common_genes = [f"gene_{i}" for i in range(g_dim)]
        x_common = [x.astype(np.float32) for x in x_list]
        info = {
            "requested_mode": align_mode,
            "used_mode": "order",
            "g": g_dim,
            "is_gene_level": True,
            "feature_name_sources": feature_name_sources or [],
            "message": "Feature names absent; equal feature counts are treated as pre-aligned common genes.",
        }
        return x_common, common_genes, info

    if align_mode == "order":
        raise ValueError(
            "feature_align_mode='order' requires equal feature counts across views; "
            f"got {feature_dims}."
        )

    x_common, common_features, info = _build_latent_feature_views(x_list, latent_feature_dim)
    info["requested_mode"] = align_mode
    info["message"] = (
        "Feature names absent and feature counts differ; using latent fallback. "
        "This is trainable but not a gene-level biological intersection."
    )
    return x_common, common_features, info


def _multi_marker_expression(dataset_dir, raw_rna, feature_names, common_genes, alignment_info):
    """Recover RNA columns without changing either model view or its RNG state."""
    if alignment_info.get("used_mode") != "names" or not feature_names[0]:
        return None, {"marker_input_status": "unavailable_without_named_gene_alignment"}

    mapping = _first_index_by_name(feature_names[0])
    source_indices = np.asarray([mapping[name] for name in common_genes], dtype=np.int64)
    metadata = {
        "marker_input_status": "ready", "marker_expression_view": "RNA",
        "rna_source_feature_indices": source_indices.tolist(),
    }
    return raw_rna[:, source_indices].astype(np.float32), metadata


def features_to_adj_omics(
    dataset,
    data_root,
    feature_align_mode="auto",
    latent_feature_dim=2000,
    anchor_multiplier=None,
    mc_multiplier=None,
    mf_multiplier=None,
    return_expression=False,
):
    dataset_dir = Path(data_root) / dataset
    if not dataset_dir.exists():
        dataset_path = Path(dataset)
        if dataset_path.exists() and dataset_path.is_dir():
            dataset_dir = dataset_path
        else:
            raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    rna_path, atac_path, label_path = _resolve_omics_files(dataset_dir)

    labels = _read_multi_h5_labels(label_path)
    num_sample = labels.shape[0]

    raw_rna = _ensure_samples_by_features(_read_multi_h5_matrix(rna_path), num_sample, "RNA")
    raw_atac = _ensure_samples_by_features(_read_multi_h5_matrix(atac_path), num_sample, "ATAC")
    raw_views = [raw_rna, raw_atac]
    view_paths = [rna_path, atac_path]
    view_names = ["RNA", "ATAC"]

    n_view = len(raw_views)
    n_cluster = len(set(np.asarray(labels)))
    explicit_split_multiplier = mc_multiplier is not None or mf_multiplier is not None
    explicit_anchor_multiplier = anchor_multiplier is not None and not explicit_split_multiplier
    if explicit_split_multiplier:
        if mc_multiplier is None or mf_multiplier is None:
            raise ValueError(
                "mc_multiplier and mf_multiplier must be provided together "
                "when using split anchor multipliers."
            )
        mc_multiplier = int(mc_multiplier)
        mf_multiplier = int(mf_multiplier)
        if mc_multiplier <= 0 or mf_multiplier <= 0:
            raise ValueError(
                "mc_multiplier and mf_multiplier must be positive integers; "
                f"got mc_multiplier={mc_multiplier}, mf_multiplier={mf_multiplier}."
            )
        mc = int(n_cluster * mc_multiplier)
    elif explicit_anchor_multiplier:
        anchor_multiplier = int(anchor_multiplier)
        if anchor_multiplier <= 0:
            raise ValueError(f"anchor_multiplier must be a positive integer, got {anchor_multiplier}.")
        mc = int(n_cluster * anchor_multiplier)
    else:
        mc = int(n_cluster)

    x_cell = []
    feature_names = []
    feature_name_sources = []
    for view_idx, raw_x in enumerate(raw_views):
        scaler = StandardScaler(with_mean=True, with_std=True)
        x_norm = scaler.fit_transform(raw_x.T).astype(np.float32)
        x_cell.append(x_norm)
        names, source = _read_multi_feature_names(
            dataset_dir,
            view_paths[view_idx],
            view_names[view_idx],
            x_norm.shape[0],
        )
        feature_names.append(names)
        feature_name_sources.append(source)

    x_common, common_genes, alignment_info = _build_common_gene_views(
        x_cell,
        feature_names,
        feature_name_sources,
        align_mode=feature_align_mode,
        latent_feature_dim=latent_feature_dim,
    )

    xx = np.vstack(x_cell)
    mf_upper = max(2, min(num_sample - 1, xx.shape[0] - 1))
    if explicit_split_multiplier:
        mf = int(n_cluster * mf_multiplier)
    elif explicit_anchor_multiplier:
        mf = int(mc)
    else:
        mf = int(max(2, min(mc, mf_upper)))

    u_cell, _, _ = _run_svds(xx.T, mc, "Cell anchor initialization")
    np.random.seed(12)
    kmeans_c = KMeans(n_clusters=mc, max_iter=100, n_init=10, random_state=12)
    cell_labels = kmeans_c.fit_predict(u_cell)
    zc_init = _one_hot_from_labels(cell_labels, mc)

    dc_init = []
    ec_init = []
    for view_idx, x_view in enumerate(x_cell):
        c_mat = np.dot(x_view, zc_init.T)
        d_view = _svd_uv(c_mat)
        e_view = x_view - np.dot(d_view, zc_init)
        dc_init.append(d_view.astype(np.float32))
        ec_init.append(e_view.astype(np.float32))

    x_feature_profile = np.mean(np.stack(x_common, axis=0), axis=0).astype(np.float32)
    u_feature, _, _ = _run_svds(x_feature_profile, mf, "Feature anchor initialization")
    kmeans_f = KMeans(n_clusters=mf, max_iter=100, n_init=10, random_state=12)
    feature_labels = kmeans_f.fit_predict(u_feature)
    zf_init = _one_hot_from_labels(feature_labels, mf)

    df_init = []
    ef_init = []
    for view_idx, x_bar in enumerate(x_common):
        x_feature = x_bar.T.astype(np.float32)
        c_mat = np.dot(x_feature, zf_init.T)
        d_view = _svd_uv(c_mat)
        e_view = x_feature - np.dot(d_view, zf_init)
        df_init.append(d_view.astype(np.float32))
        ef_init.append(e_view.astype(np.float32))
    outputs = (
        x_cell,
        x_common,
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
        n_cluster,
        common_genes,
        alignment_info,
    )
    if return_expression:
        expression, marker_info = _multi_marker_expression(
            dataset_dir, raw_rna, feature_names, common_genes, alignment_info)
        alignment_info.update(marker_info)
        return outputs + (expression,)
    return outputs
