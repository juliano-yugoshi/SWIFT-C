"""Discrete weak-label clustering; training never accesses reference labels.

Only explicitly designated thematic suggestions are allowed as weak labels.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
import csv
import io
import json
import re
import unicodedata
import zipfile

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import normalize as row_normalize

SUGGESTION_COLUMNS = ("sug_gemini", "sug_claude", "sug_llama")


def canonical_label(value: str) -> str:
    """Surface normalization only: no semantic embedding, synonym or taxonomy."""
    s = unicodedata.normalize("NFKC", str(value or "")).casefold()
    s = re.sub(r"[\W_]+", " ", s, flags=re.UNICODE)
    return " ".join(s.split())


def open_real_data(path: str | Path, label_columns=SUGGESTION_COLUMNS,
                   id_column="video_id", target_column="category") -> pd.DataFrame:
    """Read only explicitly selected suggestion columns, IDs and evaluation target.

    Categorical hints cat_* and the target column are never used as weak labels.
    """
    path = Path(path)
    columns = [id_column, *label_columns, target_column]
    if (not label_columns or any(not c or c in (id_column, target_column)
                                 or c.lower().startswith("cat_") for c in label_columns)):
        raise ValueError("Weak-label columns must exclude categorical hints, IDs and evaluation targets")
    csv.field_size_limit(min(2**31 - 1, 2**30))
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as z:
            names = [n for n in z.namelist() if n.lower().endswith(".csv") and not n.startswith("__MACOSX/")]
            if len(names) != 1:
                raise ValueError(f"Expected exactly one CSV in ZIP, found {names}")
            with z.open(names[0]) as fh:
                with io.TextIOWrapper(fh, encoding="utf-8-sig", errors="replace") as stream:
                    frame = pd.read_csv(stream, usecols=lambda c: c in columns,
                                        keep_default_na=False, dtype=str, low_memory=False)
    else:
        frame = pd.read_csv(path, usecols=lambda c: c in columns,
                            keep_default_na=False, dtype=str, low_memory=False)
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if frame[id_column].duplicated().any():
        raise ValueError("Duplicate instance IDs; cannot safely match embedding cache")
    return frame[columns].copy()


def make_weak_array(frame: pd.DataFrame, label_columns=SUGGESTION_COLUMNS) -> np.ndarray:
    return frame[list(label_columns)].fillna("").astype(str).apply(lambda col: col.str.strip()).to_numpy()


def vote_labels(weak: np.ndarray, normalize: bool = False) -> tuple[np.ndarray, dict]:
    """Open-vocabulary discrete IDs using modal label per instance.

    When there is a tie, break it by corpus frequency (without target labels),
    then lexicographically, so no LLM gets privileged priority.
    """
    norm = canonical_label if normalize else lambda s: str(s).strip()
    processed = [[norm(x) for x in row if str(x).strip()] for row in weak]
    frequency = Counter(x for row in processed for x in row if x)
    decisions, ties, empty = [], 0, 0
    for row in processed:
        row = [x for x in row if x]
        if not row:
            decisions.append("__NO_SUGGESTION__")
            empty += 1
            continue
        freq = Counter(row)
        winning_count = max(freq.values())
        top = [s for s, count in freq.items() if count == winning_count]
        if len(top) > 1:
            ties += 1
        decisions.append(sorted(top, key=lambda s: (-frequency[s], s))[0])
    arr = np.asarray(decisions, dtype=str)
    return arr, {"n_unique_labels": len(set(arr)), "tie_fraction": ties / len(arr),
                 "missing_fraction": empty / len(arr), "n_singleton_voted_labels":
                 sum(c == 1 for c in Counter(arr).values())}


def _read_npz_x(path: Path, wanted_ids: np.ndarray) -> np.ndarray:
    with np.load(path, allow_pickle=False) as z:
        if "X" not in z or "ids" not in z:
            raise ValueError(f"{path.name}: expected arrays 'X' and 'ids'")
        ids = np.asarray(z["ids"]).astype(str)
        if len(ids) != len(wanted_ids) or len(set(ids.tolist())) != len(ids):
            raise ValueError(f"{path.name}: incompatible number of IDs")
        if np.array_equal(ids, wanted_ids):
            X = np.asarray(z["X"], dtype=np.float32)
        else:
            loc = {k: i for i, k in enumerate(ids)}
            if set(loc) != set(wanted_ids.tolist()):
                raise ValueError(f"{path.name}: missing/extra IDs")
            X = np.asarray(z["X"], dtype=np.float32)[[loc[k] for k in wanted_ids]]
    if X.ndim != 2 or X.shape[0] != len(wanted_ids):
        raise ValueError(f"{path.name}: unexpected input embedding shape")
    if not np.isfinite(X).all():
        raise ValueError(f"{path.name}: non-finite input embeddings")
    return row_normalize(X, norm="l2").astype(np.float32)


def load_cached_inputs(path: str | Path, wanted_ids: np.ndarray) -> tuple[np.ndarray, Path]:
    """Reuse input embeddings from the SWIFT-C NPZ cache. Never read label embeddings."""
    path = Path(path)
    if path.is_dir():
        candidates = sorted(path.glob("embeddings_*.npz"))
        if not candidates:
            candidates = sorted(path.glob("*.npz"))
        if not candidates:
            raise FileNotFoundError(f"No .npz embeddings found in {path}")
        errors = []
        for candidate in candidates:
            try:
                return _read_npz_x(candidate, wanted_ids), candidate
            except (ValueError, KeyError, zipfile.BadZipFile) as exc:
                errors.append(f"{candidate.name}: {exc}")
        raise ValueError("No embedding cache aligned to CSV IDs: " + "; ".join(errors[:3]))
    return _read_npz_x(path, wanted_ids), path


@dataclass
class DiscreteMustLinkResult:
    clusters: np.ndarray
    baseline_clusters: np.ndarray
    n_groups: int
    groups_size_ge_2: int
    covered_fraction: float
    average_support: float
    iterations: int
    changed_fraction: float


def _make_incidence(weak: np.ndarray, mode="normalized") -> tuple[sparse.csr_matrix, np.ndarray, int]:
    """Instance × repeated literal-label membership matrix, sparse."""
    normalizer = canonical_label if mode == "normalized" else lambda v: str(v).strip()
    groups = defaultdict(list)
    for i, vals in enumerate(weak):
        for label in set(normalizer(v) for v in vals if str(v).strip()):
            if label:
                groups[label].append(i)
    ids, cols, sizes = [], [], []
    for members in groups.values():
        if len(members) < 2:
            continue
        col = len(sizes)
        ids.extend(members)
        cols.extend([col] * len(members))
        sizes.append(len(members))
    mat = sparse.coo_matrix((np.ones(len(ids), dtype=np.float32), (ids, cols)),
                            shape=(weak.shape[0], len(sizes)), dtype=np.float32).tocsr()
    return mat, np.asarray(sizes, dtype=np.float32), len(groups)


def discrete_mustlink_kmeans(X: np.ndarray, weak: np.ndarray, k: int, seed: int,
                             strength: float = 0.2, max_iterations: int = 6,
                             mode: str = "normalized") -> DiscreteMustLinkResult:
    """K-Means plus soft same-string must-link disagreement penalty.

    Minimize approximately: Σ_i ||x_i-μ_{y_i}||² +
    λ Σ_g [fraction of members assigned to different clusters].

    Scores use leave-one-out group assignment fractions. Common labels are
    normalized by group size (otherwise hubs dominate). No semantic
    similarity of labels is used. All updates are unsupervised; K is fixed.
    """
    if X.shape[0] != weak.shape[0]:
        raise ValueError("Input and weak-label rows are misaligned")
    if not 2 <= k < len(X):
        raise ValueError("k must be in [2, n_instances)")
    km = KMeans(n_clusters=k, n_init=10, random_state=seed, max_iter=250)
    initial = km.fit_predict(X).astype(np.int32)
    labels = initial.copy()
    incidence, group_sizes, all_groups = _make_incidence(weak, mode=mode)
    if len(group_sizes) == 0:
        return DiscreteMustLinkResult(labels, initial, all_groups, 0, 0., 0., 0, 0.)
    counts_per_instance = np.asarray(incidence.sum(axis=1)).ravel().astype(np.float32)
    reciprocal = np.reciprocal(group_sizes - 1)
    self_correction = np.asarray(incidence @ reciprocal).ravel()
    denom = np.maximum(counts_per_instance, 1)
    changed_count = 0
    n_iter = 0
    for iteration in range(max_iterations):
        n_iter = iteration + 1
        one_hot = sparse.csr_matrix((np.ones(len(labels), dtype=np.float32),
                                     (np.arange(len(labels)), labels)), shape=(len(labels), k))
        memberships = (incidence.T @ one_hot).toarray().astype(np.float32)
        memberships *= reciprocal[:, None]
        votes = np.asarray(incidence @ memberships, dtype=np.float32)
        # Leave-one-out: each row's own contribution must not support itself.
        votes[np.arange(len(labels)), labels] -= self_correction
        votes /= denom[:, None]
        # Note: if no repeated suggestion, votes remain zero.
        centers = np.stack([X[labels == q].mean(axis=0) if np.any(labels == q) else km.cluster_centers_[q]
                            for q in range(k)], axis=0)
        d2 = np.maximum(0, ((X * X).sum(axis=1)[:, None]
                     + (centers * centers).sum(axis=1)[None, :] - 2 * X @ centers.T))
        proposal = np.argmax(-d2 + strength * votes, axis=1).astype(np.int32)
        # Avoid empty clusters in a K-constrained comparison.
        counts = np.bincount(proposal, minlength=k)
        for empty in np.flatnonzero(counts == 0):
            excess = counts[proposal] > 1
            donors = np.flatnonzero(excess)
            if not len(donors):
                raise RuntimeError("No donor to restore empty cluster")
            penalties = d2[donors, proposal[donors]]
            donor = donors[int(np.argmax(penalties))]
            counts[proposal[donor]] -= 1
            proposal[donor] = empty
            counts[empty] += 1
        changes = int(np.sum(proposal != labels))
        labels = proposal
        changed_count += changes
        if changes == 0:
            break
    memberships = (incidence.T @ sparse.csr_matrix((np.ones(len(labels), dtype=np.float32),
                              (np.arange(len(labels)), labels)), shape=(len(labels), k))).toarray()
    # Pair agreement fraction per source group, weighted by group size.
    matches = (memberships * (memberships - 1)).sum(axis=1)
    total = group_sizes * (group_sizes - 1)
    avg_agree = float(np.average(matches / total, weights=group_sizes))
    return DiscreteMustLinkResult(labels, initial, all_groups, len(group_sizes),
                                  float(np.mean(counts_per_instance > 0)), avg_agree,
                                  n_iter, float(np.mean(labels != initial)))


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Called only after all cluster predictions have been generated."""
    return {"ari": float(adjusted_rand_score(y_true, y_pred)),
            "nmi": float(normalized_mutual_info_score(y_true, y_pred)),
            "ami": float(adjusted_mutual_info_score(y_true, y_pred))}


def text_diagnostics(weak: np.ndarray) -> dict:
    values = [str(v).strip() for v in weak.ravel() if str(v).strip()]
    normalized = [canonical_label(v) for v in values]
    counts = Counter(values)
    canonical_counts = Counter(normalized)
    same_exact = [len(set(row)) == 1 for row in weak if all(str(s).strip() for s in row)]
    same_normalized = [len(set(canonical_label(v) for v in row)) == 1 for row in weak
                       if all(str(s).strip() for s in row)]
    return {
        "n_rows": int(len(weak)),
        "n_suggestions_present": int(len(values)),
        "n_unique_exact_labels": len(counts),
        "n_unique_normalized_labels": len(canonical_counts),
        "fraction_exact_singletons": float(sum(v == 1 for v in counts.values()) / max(1, len(counts))),
        "fraction_normalized_singletons": float(sum(v == 1 for v in canonical_counts.values()) / max(1, len(canonical_counts))),
        "unanimity_exact_fraction": float(np.mean(same_exact)) if same_exact else 0.,
        "unanimity_normalized_fraction": float(np.mean(same_normalized)) if same_normalized else 0.,
        "all_present_fraction": float(len(same_exact) / len(weak)),
    }
