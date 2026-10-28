"""
This program is to evaluate clustering performance.
"""

from scipy.stats import mode
from sklearn.cluster import KMeans
import numpy as np
from sklearn import metrics
from scipy.optimize import linear_sum_assignment as linear_assignment
from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics.cluster._supervised import check_clusterings
import warnings

warnings.filterwarnings("ignore")


def KMeansClustering(features, gnd, clusterNum, randNum):
    kmeans = KMeans(n_clusters=clusterNum, n_init=1, max_iter=500, random_state=randNum)
    if np.isnan(features).any():
        nan = np.isnan(features)
        features[nan] = 0.000001
    estimator = kmeans.fit(features)
    clusters = estimator.labels_
    labels = np.zeros_like(clusters)
    for i in range(clusterNum):
        mask = clusters == i
        labels[mask] = mode(gnd[mask])[0]

    return labels


def cluster_acc(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    d = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((d, d), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    ind = linear_assignment(w.max() - w)
    ind = ind.T
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size


def clustering_purity(labels_true, labels_pred):
    y_true = labels_true.copy()
    y_pred = labels_pred.copy()
    if y_true.shape[1] != 1:
        y_true = y_true.T
    if y_pred.shape[1] != 1:
        y_pred = y_pred.T

    n_samples = len(y_true)

    u_y_true = np.unique(y_true)
    n_true_classes = len(u_y_true)
    y_true_temp = np.zeros((n_samples, 1))
    if n_true_classes != max(y_true):
        for i in range(n_true_classes):
            y_true_temp[np.where(y_true == u_y_true[i])] = i + 1
        y_true = y_true_temp

    u_y_pred = np.unique(y_pred)
    n_pred_classes = len(u_y_pred)
    y_pred_temp = np.zeros((n_samples, 1))
    if n_pred_classes != max(y_pred):
        for i in range(n_pred_classes):
            y_pred_temp[np.where(y_pred == u_y_pred[i])] = i + 1
        y_pred = y_pred_temp

    u_y_pred = np.unique(y_pred)
    n_pred_classes = len(u_y_pred)

    n_correct = 0
    for i in range(n_pred_classes):
        incluster = y_true[np.where(y_pred == u_y_pred[i])]

        inclunub = np.histogram(incluster, bins=range(1, int(max(incluster)) + 1))[0]
        if len(inclunub) != 0:
            n_correct = n_correct + max(inclunub)

    purity = n_correct / len(y_pred)

    return purity


def b3_precision_recall_fscore(labels_true, labels_pred):
    labels_true, labels_pred = check_clusterings(labels_true, labels_pred)

    if labels_true.shape == (0,):
        raise ValueError("input labels must not be empty.")

    n_samples = len(labels_true)
    true_clusters = {}
    pred_clusters = {}

    for i in range(n_samples):
        true_cluster_id = labels_true[i]
        pred_cluster_id = labels_pred[i]

        if true_cluster_id not in true_clusters:
            true_clusters[true_cluster_id] = set()
        if pred_cluster_id not in pred_clusters:
            pred_clusters[pred_cluster_id] = set()

        true_clusters[true_cluster_id].add(i)
        pred_clusters[pred_cluster_id].add(i)

    for cluster_id, cluster in true_clusters.items():
        true_clusters[cluster_id] = frozenset(cluster)
    for cluster_id, cluster in pred_clusters.items():
        pred_clusters[cluster_id] = frozenset(cluster)

    precision = 0.0
    recall = 0.0

    intersections = {}

    for i in range(n_samples):
        pred_cluster_i = pred_clusters[labels_pred[i]]
        true_cluster_i = true_clusters[labels_true[i]]

        if (pred_cluster_i, true_cluster_i) in intersections:
            intersection = intersections[(pred_cluster_i, true_cluster_i)]
        else:
            intersection = pred_cluster_i.intersection(true_cluster_i)
            intersections[(pred_cluster_i, true_cluster_i)] = intersection

        precision += len(intersection) / len(pred_cluster_i)
        recall += len(intersection) / len(true_cluster_i)

    precision /= n_samples
    recall /= n_samples

    f_score = 2 * precision * recall / (precision + recall)

    return f_score, precision, recall


def clusteringMetrics(trueLabel, predictiveLabel):
    acc = metrics.accuracy_score(trueLabel, predictiveLabel)
    nmi = normalized_mutual_info_score(trueLabel, predictiveLabel)
    purity = clustering_purity(trueLabel.reshape((-1, 1)), predictiveLabel.reshape(-1, 1))
    ari = metrics.adjusted_rand_score(trueLabel, predictiveLabel)
    fscore, precision, recall = b3_precision_recall_fscore(trueLabel, predictiveLabel)

    return acc, nmi, purity, ari, fscore, precision, recall


def StatisticClustering(features, gnd, clusterNum):
    repNum = 10
    acc_list = np.zeros((repNum, 1))
    nmi_list = np.zeros((repNum, 1))
    purity_list = np.zeros((repNum, 1))
    ari_list = np.zeros((repNum, 1))
    fscore_list = np.zeros((repNum, 1))
    precision_list = np.zeros((repNum, 1))
    recall_list = np.zeros((repNum, 1))

    for i in range(repNum):
        predictiveLabel = KMeansClustering(features, gnd, clusterNum, i)
        acc, nmi, purity, ari, fscore, precision, recall = clusteringMetrics(gnd, predictiveLabel)

        acc_list[i] = acc
        nmi_list[i] = nmi
        purity_list[i] = purity
        ari_list[i] = ari
        fscore_list[i] = fscore
        precision_list[i] = precision
        recall_list[i] = recall

    acc_mean_std = np.around([np.mean(acc_list), np.std(acc_list)], decimals=4)
    nmi_mean_std = np.around([np.mean(nmi_list), np.std(nmi_list)], decimals=4)
    purity_mean_std = np.around([np.mean(purity_list), np.std(purity_list)], decimals=4)
    ari_mean_std = np.around([np.mean(ari_list), np.std(ari_list)], decimals=4)
    fscore_mean_std = np.around([np.mean(fscore_list), np.std(fscore_list)], decimals=4)
    precision_mean_std = np.around([np.mean(precision_list), np.std(precision_list)], decimals=4)
    recall_mean_std = np.around([np.mean(recall_list), np.std(recall_list)], decimals=4)

    return (
        acc_mean_std,
        nmi_mean_std,
        purity_mean_std,
        ari_mean_std,
        fscore_mean_std,
        precision_mean_std,
        recall_mean_std,
    )


def evaluate_with_details(features, gnd, cluster_num):
    """Exact legacy evaluation, with raw clusters and majority labels retained."""
    runs, values = [], []
    for seed in range(10):
        estimator = KMeans(n_clusters=cluster_num, n_init=1, max_iter=500, random_state=seed)
        raw = estimator.fit_predict(np.asarray(features).copy())
        mapped = np.zeros_like(raw)
        mapping = {}
        for cluster in range(cluster_num):
            mask = raw == cluster
            if np.any(mask):
                label = int(np.asarray(mode(gnd[mask])[0]).reshape(-1)[0])
                mapped[mask] = label
                mapping[str(cluster)] = label
        scores = clusteringMetrics(gnd, mapped)
        values.append(scores)
        runs.append({"seed": seed, "raw_labels": raw.tolist(),
                     "majority_mapped_labels": mapped.tolist(),
                     "majority_mapping": mapping,
                     "metrics": dict(zip(("ACC", "NMI", "Purity", "ARI", "Fscore", "Precision", "Recall"),
                                         (float(v) for v in scores)))})
    values = np.asarray(values)
    summary = np.around([values.mean(axis=0), values.std(axis=0)], decimals=4).T
    return summary, runs
