"""Metrics are computed from predicted cluster IDs and reference categories AFTER clustering.

No cat_* and no sug_* columns are read here; ground truth is never used for fitting.
Noise (-1) and any other negative label are treated as singleton abstentions.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class Evaluation:
    summary: dict
    category_table: pd.DataFrame
    query_table: pd.DataFrame


def evaluate_clustering(ids: np.ndarray, truth: np.ndarray, clusters: np.ndarray,
                        *, method: str = "", query_details: bool = False) -> Evaluation:
    """Evaluate each video's cluster as an unordered recommendation set.

    A query recommends all *other* members of the same non-negative cluster.
    Precision=0, Recall=0, F1=0 for a query without candidates or relevant
    items. This deliberately penalizes singleton/abstained queries.
    """
    ids = np.asarray(ids).astype(str)
    truth = np.asarray(truth).astype(str)
    pred = np.asarray(clusters)
    n = len(truth)
    if n < 2 or len(pred) != n or len(ids) != n:
        raise ValueError("ids, category, and predictions must have equal lengths >=2")
    if len(set(ids)) != n:
        raise ValueError("Repeated item IDs; cannot evaluate unambiguously")
    if any(not x for x in truth):
        raise ValueError("Empty target category; explicit ground truth is required")
    if pred.ndim != 1:
        raise ValueError("Predictions must be 1-D")
    # Avoid treating numeric negative noise labels as one large, meaningful cluster.
    pred = pd.Series(pred).astype(str).to_numpy()
    negative = pd.to_numeric(pd.Series(pred), errors="coerce").to_numpy() < 0
    missing = pd.isna(pd.Series(clusters)).to_numpy()
    invalid = negative | missing
    grouping = pred.astype(object)
    for ix in np.flatnonzero(invalid):
        grouping[ix] = ("__abstained__", int(ix))
    # Factorization explicitly handles tuples for abstained items.
    cluster_ids, cluster_keys = pd.factorize(pd.Series(grouping), sort=False)
    label_ids, label_keys = pd.factorize(pd.Series(truth), sort=False)
    ncl = len(cluster_keys)
    ncat = len(label_keys)
    count_cl = np.bincount(cluster_ids, minlength=ncl)
    count_cat = np.bincount(label_ids, minlength=ncat)
    # O(n) contingency lookup, without constructing an n-by-n matrix.
    co_keys = cluster_ids.astype(np.int64) * int(ncat) + label_ids.astype(np.int64)
    contingency = np.bincount(co_keys, minlength=ncl * ncat).reshape(ncl, ncat)
    relevant_in_cluster = contingency[cluster_ids, label_ids] - 1
    candidates = count_cl[cluster_ids] - 1
    total_relevant = count_cat[label_ids] - 1
    precision = np.divide(relevant_in_cluster, candidates, out=np.zeros(n, dtype=float), where=candidates > 0)
    recall = np.divide(relevant_in_cluster, total_relevant, out=np.zeros(n, dtype=float), where=total_relevant > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros(n, dtype=float), where=(precision + recall) > 0)
    baseline = np.mean(total_relevant / (n - 1))
    # Ordered video pairs are counted exactly, without self-pairs.
    tp_pairs = int(np.sum(contingency.astype(np.int64) * (contingency.astype(np.int64) - 1)))
    candidate_pairs = int(np.sum(count_cl.astype(np.int64) * (count_cl.astype(np.int64) - 1)))
    relevant_pairs = int(np.sum(count_cat.astype(np.int64) * (count_cat.astype(np.int64) - 1)))
    pair_p = tp_pairs / candidate_pairs if candidate_pairs else 0.0
    pair_r = tp_pairs / relevant_pairs if relevant_pairs else 0.0
    pair_f1 = 2 * pair_p * pair_r / (pair_p + pair_r) if pair_p + pair_r else 0.0
    q = pd.DataFrame({"video_id": ids, "category": truth, "cluster": cluster_ids,
                      "n_candidates": candidates, "n_relevant_candidates": relevant_in_cluster,
                      "n_relevant_in_corpus": total_relevant,
                      "cluster_precision": precision, "cluster_recall": recall, "cluster_f1": f1})
    cat = q.groupby('category', sort=True, as_index=False).agg(
        n_queries=('video_id', 'size'),
        precision=('cluster_precision', 'mean'),
        recall=('cluster_recall', 'mean'),
        f1=('cluster_f1', 'mean'),
        mean_candidates=('n_candidates', 'mean'),
        fraction_with_candidates=('n_candidates', lambda s: float((s > 0).mean())),
        fraction_with_relevant=('n_relevant_candidates', lambda s: float((s > 0).mean())))
    sizes = count_cl.astype(float)
    summary = {
        'n': n, 'n_clusters_found': ncl, 'n_reference_categories': ncat,
        'n_noise': int(invalid.sum()), 'noise_rate': float(invalid.mean()),
        'precision': float(precision.mean()), 'recall': float(recall.mean()),
        'f1': float(f1.mean()),
        'macro_category_precision': float(cat.precision.mean()),
        'macro_category_recall': float(cat.recall.mean()),
        'macro_category_f1': float(cat.f1.mean()),
        'pair_precision': pair_p, 'pair_recall': pair_r, 'pair_f1': pair_f1,
        'mean_candidates': float(candidates.mean()), 'median_candidates': float(np.median(candidates)),
        'mean_cluster_size': float(sizes.mean()), 'median_cluster_size': float(np.median(sizes)),
        'singleton_query_fraction': float((candidates == 0).mean()),
        'candidate_coverage': float((candidates > 0).mean()),
        'hit_coverage': float((relevant_in_cluster > 0).mean()),
        'random_precision': float(baseline),
        'precision_lift': float(precision.mean() / baseline) if baseline else 0.0,
    }
    return Evaluation(summary, cat, q if query_details else q.iloc[:0].copy())


def evaluate_topk(ids: np.ndarray, truth: np.ndarray, clusters: np.ndarray,
                  vectors: np.ndarray, topks: tuple[int, ...] = (10, 20),
                  *, query_details: bool = False, block: int = 64) -> tuple[dict, pd.DataFrame]:
    """Nearest-neighbor ranking *within* the predicted cluster using fixed input features.

    Features are never trained or adjusted using reference categories.
    For each query, missing slots (clusters smaller than k) are nonrelevant.
    P@k = # relevant / k; AP@k uses min(k, number relevant corpus-wide).
    Singleton/noise queries receive 0 for all ranked metrics.
    """
    ids = np.asarray(ids).astype(str)
    truth = np.asarray(truth).astype(str)
    clusters = np.asarray(clusters)
    if vectors.shape[0] != len(ids) or vectors.ndim != 2:
        raise ValueError("Embeddings must be (N, D), aligned by IDs")
    if not np.all(np.isfinite(vectors)):
        raise ValueError("Embeddings contain nonfinite values")
    topks = tuple(sorted(set(int(k) for k in topks)))
    if not topks or min(topks) <= 0:
        raise ValueError("topk must contain strictly positive integers")
    n = len(ids)
    # Vectorized category IDs for each query; true categories only read at evaluation.
    categories, classids = np.unique(truth, return_inverse=True)
    class_counts = np.bincount(classids)
    total_rel = class_counts[classids] - 1
    features = np.asarray(vectors, dtype=np.float32)
    norm = np.linalg.norm(features, axis=1, keepdims=True)
    features = np.divide(features, norm, out=np.zeros_like(features), where=norm > 0)
    pred_str = pd.Series(clusters).astype(str).to_numpy()
    neg = pd.to_numeric(pd.Series(pred_str), errors='coerce').to_numpy() < 0
    null = pd.isna(pd.Series(clusters)).to_numpy()
    groups: dict[str, list[int]] = {}
    for i, cl in enumerate(pred_str):
        if neg[i] or null[i]:
            continue
        groups.setdefault(cl, []).append(i)
    ks = np.asarray(topks, dtype=np.int64)
    maxk = topks[-1]
    prec = np.zeros((n, len(ks)), dtype=np.float32)
    rec = np.zeros_like(prec)
    ap = np.zeros_like(prec)
    ndcg = np.zeros_like(prec)
    ideal_discounts = 1.0 / np.log2(np.arange(2, maxk + 2, dtype=np.float64))
    ideal_cum = np.cumsum(ideal_discounts)
    for member_list in groups.values():
        if len(member_list) <= 1:
            continue
        ix = np.asarray(member_list, dtype=np.int64)
        G = features[ix]
        ids_group = classids[ix]
        for start in range(0, len(ix), block):
            stop = min(start + block, len(ix))
            sims = G[start:stop] @ G.T
            for local, index in enumerate(range(start, stop)):
                sims[local, index] = -np.inf
            neighbors = min(maxk, len(ix) - 1)
            if neighbors == len(ix) - 1:
                order = np.argsort(-sims, axis=1, kind='stable')[:, :neighbors]
            else:
                shortlist = np.argpartition(-sims, neighbors - 1, axis=1)[:, :neighbors]
                row_order = np.argsort(-np.take_along_axis(sims, shortlist, axis=1), axis=1, kind='stable')
                order = np.take_along_axis(shortlist, row_order, axis=1)
            for local, orig in enumerate(ix[start:stop]):
                y = (ids_group[order[local]] == classids[orig]).astype(np.float64)
                cum = np.cumsum(y)
                p_at_rank = cum / np.arange(1, len(y) + 1)
                ap_sum = np.cumsum(p_at_rank * y)
                dcg_sum = np.cumsum(y * ideal_discounts[:len(y)])
                for j, k in enumerate(ks):
                    available = min(k, len(y))
                    val = cum[available-1] if available else 0.0
                    prec[orig,j] = val / k
                    rec[orig,j] = val / total_rel[orig] if total_rel[orig] else 0.0
                    denom = min(k, total_rel[orig])
                    ap[orig,j] = ap_sum[available-1] / denom if denom and available else 0.0
                    n_ideal = min(k, total_rel[orig])
                    ndcg[orig,j] = dcg_sum[available-1] / ideal_cum[n_ideal-1] if n_ideal and available else 0.0
    metrics = {}
    details = {'video_id': ids} if query_details else None
    for j, k in enumerate(ks):
        for metric, vals in (('precision', prec), ('recall', rec), ('map', ap), ('ndcg', ndcg)):
            name = f'{metric}_at_{k}'
            metrics[name] = float(vals[:,j].mean())
            if details is not None: details[name] = vals[:,j]
    return metrics, (pd.DataFrame(details) if details is not None else pd.DataFrame())
