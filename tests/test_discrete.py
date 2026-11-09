import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from swiftc.discrete.core import (
    canonical_label, vote_labels, open_real_data, make_weak_array,
    discrete_mustlink_kmeans, load_cached_inputs,
)
from swiftc.discrete.cli import build_parser, run


def sample(tmp_path):
    n = 48
    rng = np.random.default_rng(4)
    y = np.repeat(["class0", "class1", "class2"], 16)
    center = np.eye(3)[np.repeat(np.arange(3), 16)]
    X = (center + rng.normal(0, .05, (n, 3))).astype(np.float32)
    ids = np.array([f"video{i}" for i in range(n)])
    df = pd.DataFrame({
        "video_id": ids, "category": y,
        "cat_gemini": ["POISON__LOOK_AT_GOLD"] * n,
        "cat_claude": ["POISON__LOOK_AT_GOLD"] * n,
        "cat_llama": ["POISON__LOOK_AT_GOLD"] * n,
        "sug_gemini": [f"Topic {int(i // 16)}" for i in range(n)],
        "sug_claude": [f" topic {int(i // 16)} " for i in range(n)],
        "sug_llama": [f"topic-{int(i // 16)}" for i in range(n)],
    })
    csv = tmp_path / "data.csv"
    df.to_csv(csv, index=False)
    emb = tmp_path / "embeddings_aligned.npz"
    np.savez_compressed(emb, X=X, ids=ids)
    return csv, emb, df


def test_text_normalization_not_semantics():
    assert canonical_label('  Food-Preparation ') == 'food preparation'
    assert canonical_label('Food Preparation') == 'food preparation'
    assert canonical_label('Culinary Arts') != canonical_label('food preparation')


def test_data_loader_does_not_read_cat(tmp_path):
    csv, _, _ = sample(tmp_path)
    frame = open_real_data(csv)
    assert "cat_gemini" not in frame
    assert not any(col.startswith("cat_") for col in frame)
    assert list(frame.columns) == ["video_id", "sug_gemini", "sug_claude", "sug_llama", "category"]
    with pytest.raises(ValueError):
        open_real_data(csv, ["cat_gemini"])


def test_vote_normalization_changes_discrete_agreement():
    weak = np.asarray([["Food Preparation", "food-preparation", "Food/Preparation"],
                       ["Culinary Arts", "cooking", "recipe"]])
    exact, s1 = vote_labels(weak, normalize=False)
    normalized, s2 = vote_labels(weak, normalize=True)
    assert s1["tie_fraction"] == 1
    assert s2["tie_fraction"] == 0.5
    assert normalized[0] == "food preparation"


def test_mustlink_deterministic(tmp_path):
    csv, emb, df = sample(tmp_path)
    X, _ = load_cached_inputs(emb, df.video_id.to_numpy())
    weak = make_weak_array(open_real_data(csv))
    a = discrete_mustlink_kmeans(X, weak, 3, seed=42)
    b = discrete_mustlink_kmeans(X, weak, 3, seed=42)
    assert np.array_equal(a.clusters, b.clusters)
    assert len(np.unique(a.clusters)) == 3
    assert a.covered_fraction > .9


def test_embedding_ids_checked(tmp_path):
    csv, emb, df = sample(tmp_path)
    with pytest.raises(ValueError):
        load_cached_inputs(emb, np.array(["missing"] * len(df)))


def test_ground_truth_and_cat_do_not_affect_predictions(tmp_path):
    csv, emb, df = sample(tmp_path)
    parser = build_parser()
    args = parser.parse_args(["--input", str(csv), "--embeddings", str(emb),
                              "--k", "3", "--seeds", "42", "--output", str(tmp_path / "one")])
    run(args)
    p1 = pd.read_csv(tmp_path / "one" / "predictions.csv")
    df["category"] = df["category"].to_numpy()[::-1]
    df["cat_gemini"] = "REALLY_DIFFERENT_TARGET"
    df.to_csv(csv, index=False)
    args.output = str(tmp_path / "two")
    run(args)
    p2 = pd.read_csv(tmp_path / "two" / "predictions.csv")
    pd.testing.assert_frame_equal(p1, p2)
