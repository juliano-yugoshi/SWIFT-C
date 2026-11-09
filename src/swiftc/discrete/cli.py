"""Minimal CLI to demonstrate discrete weak-label fragmentation on real-world CSVs."""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .core import (SUGGESTION_COLUMNS, open_real_data, make_weak_array, vote_labels,
                   discrete_mustlink_kmeans, evaluate, text_diagnostics,
                   load_cached_inputs)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="swiftc discrete", description="Discrete weak-label baselines on real-world CSVs")
    parser.add_argument("--input", required=True, help="CSV or ZIP containing a single real-world CSV")
    parser.add_argument("--output", default="results/discrete_realworld")
    parser.add_argument("--embeddings", default=None,
                        help="Existing text embedding .npz or its embedding_cache directory; no re-encoding")
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--seeds", default="42,123,2025", help="Seeds for discrete must-link K-Means")
    parser.add_argument("--strength", type=float, default=0.2,
                        help="Prespecified discrete must-link penalty coefficient")
    parser.add_argument("--max-iterations", type=int, default=6)
    parser.add_argument("--id-column", default="video_id")
    parser.add_argument("--target-column", default="category", help="Evaluation ONLY")
    parser.add_argument("--label-columns", default=",".join(SUGGESTION_COLUMNS),
                        help="Comma-separated thematic weak-label columns. Defaults to the MSR-VTT suggestion fields.")
    return parser


def run(args: argparse.Namespace) -> tuple[pd.DataFrame, dict]:
    console = Console()
    cols = tuple(x.strip() for x in args.label_columns.split(",") if x.strip())
    if (not cols or any(c.lower().startswith("cat_") or c in (args.target_column, args.id_column) for c in cols)):
        raise ValueError("Weak-label columns must exclude categorical hints and evaluation targets")
    seeds = tuple(int(x) for x in args.seeds.split(",") if x.strip())
    if not seeds:
        raise ValueError("At least one seed is required")
    if args.strength < 0 or args.max_iterations < 1:
        raise ValueError("strength must be >=0; max-iterations >=1")
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    console.print(Panel.fit(
        "[bold]Discrete open-vocabulary weak-label baselines[/bold]\n"
        "Real-world dataset | No predefined categorical suggestions | No LLM inference",
        title="SWIFT-C DISCRETE BASELINES"))
    t = time.perf_counter()
    frame = open_real_data(args.input, cols, args.id_column, args.target_column)
    ids = frame[args.id_column].astype(str).to_numpy()
    y_true = frame[args.target_column].astype(str).to_numpy()  # Evaluation only
    weak = make_weak_array(frame, cols)
    diagnostics = text_diagnostics(weak)
    diagnostics.update({"sources": cols, "k_target": args.k, "seeds": seeds,
                        "reference_column_used_only_for_evaluation": args.target_column,
                        "ignored_columns": "cat_* (never loaded)",
                        "n_gold_classes": int(len(np.unique(y_true)))})
    console.print(f"[cyan]DATA[/cyan] {len(ids):,} instances, {len(cols)} sources, "
                  f"{diagnostics['n_unique_exact_labels']:,} exact label strings")
    rows = []
    predictions = pd.DataFrame({args.id_column: ids})
    for method, normal in (("exact_vote", False), ("normalized_vote", True)):
        elapsed = time.perf_counter()
        raw, stats = vote_labels(weak, normalize=normal)
        encoded, _ = pd.factorize(raw, sort=True)
        metrics = evaluate(y_true, encoded)
        results = {"method": method, "seed": "", "k_requested": "",
                   "n_clusters_found": stats["n_unique_labels"], "n": len(ids),
                   **metrics, "duration_s": round(time.perf_counter() - elapsed, 4),
                   "must_link_groups": "", "covered_fraction": "", "modified_fraction": "",
                   "tie_fraction": stats["tie_fraction"]}
        rows.append(results)
        predictions[method] = encoded
        diagnostics[method] = stats
        console.print(f"[green]DONE[/green] {method} | clusters={stats['n_unique_labels']:,} "
                      f"| ties={stats['tie_fraction']:.1%} | ARI={metrics['ari']:.4f} "
                      f"| NMI={metrics['nmi']:.4f}")
    if args.embeddings:
        embeddings, cache_path = load_cached_inputs(args.embeddings, ids)
        diagnostics["input_embedding_cache_used"] = str(cache_path)
        console.print(f"[cyan]CACHE[/cyan] Reusing input embeddings {embeddings.shape} from {cache_path}")
        for seed in seeds:
            started = time.perf_counter()
            result = discrete_mustlink_kmeans(
                embeddings, weak, k=args.k, seed=seed, strength=args.strength,
                max_iterations=args.max_iterations, mode="normalized")
            metrics = evaluate(y_true, result.clusters)
            rows.append({"method": "discrete_mustlink_kmeans", "seed": seed,
                         "k_requested": args.k, "n_clusters_found": len(np.unique(result.clusters)),
                         "n": len(ids), **metrics,
                         "duration_s": round(time.perf_counter() - started, 4),
                         "must_link_groups": result.groups_size_ge_2,
                         "covered_fraction": result.covered_fraction,
                         "modified_fraction": result.changed_fraction, "tie_fraction": ""})
            predictions[f"discrete_mustlink_s{seed}"] = result.clusters
            diagnostics[f"mustlink_seed_{seed}"] = {
                "n_label_keys": result.n_groups,
                "n_repeated_label_groups": result.groups_size_ge_2,
                "instances_covered_by_repeated_labels": result.covered_fraction,
                "pairwise_agreement_inside_label_groups": result.average_support,
                "iterations": result.iterations,
                "fraction_changed_from_input_kmeans": result.changed_fraction}
            console.print(f"[green]DONE[/green] discrete_mustlink_kmeans seed={seed} "
                          f"| ARI={metrics['ari']:.4f} | NMI={metrics['nmi']:.4f} "
                          f"| changed={result.changed_fraction:.2%}")
    else:
        console.print("[yellow]SKIP[/yellow] discrete_mustlink_kmeans: "
                      "provide --embeddings to reuse the existing text embedding cache")
    report = pd.DataFrame(rows)
    report.to_csv(out / "metrics.csv", index=False)
    predictions.to_csv(out / "predictions.csv", index=False)
    diagnostics["duration_s"] = round(time.perf_counter() - t, 2)
    with open(out / "diagnostics.json", "w", encoding="utf-8") as fh:
        json.dump(diagnostics, fh, ensure_ascii=False, indent=2, default=lambda v: v.item() if hasattr(v, 'item') else str(v))
    summary = (report.groupby("method", as_index=False)
               .agg(ari_mean=("ari", "mean"), ari_std=("ari", "std"),
                    nmi_mean=("nmi", "mean"), ami_mean=("ami", "mean"),
                    n_clusters=("n_clusters_found", "mean")))
    summary.to_csv(out / "summary.csv", index=False)
    table = Table(title="Real-world discrete baselines (descriptive comparison)")
    for col in ("Method", "ARI", "NMI", "AMI", "Clusters"):
        table.add_column(col, justify="right" if col != "Method" else "left")
    for _, row in summary.iterrows():
        table.add_row(row.method, f"{row.ari_mean:.4f}", f"{row.nmi_mean:.4f}",
                      f"{row.ami_mean:.4f}", f"{row.n_clusters:,.0f}")
    console.print(table)
    lines = [
        "# Discrete natural-language weak-label baselines",
        "", "The training procedures consume exclusively the existing `sug_*` strings.",
        "Neither `cat_*` nor the reference class are used in fitting. Reference labels",
        "are used only for ARI, NMI and AMI evaluation after predictions exist.",
        "", "| Method | ARI | NMI | AMI | Produced clusters |",
        "|---|---:|---:|---:|---:|",
    ]
    for _, row in summary.iterrows():
        lines.append(f"| {row.method} | {row.ari_mean:.4f} | {row.nmi_mean:.4f} | "
                     f"{row.ami_mean:.4f} | {row.n_clusters:,.0f} |")
    lines.extend(["", "## Text fragmentation",
                  f"- Unique exact strings: {diagnostics['n_unique_exact_labels']:,}",
                  f"- Unique normalized strings: {diagnostics['n_unique_normalized_labels']:,}",
                  f"- Fraction of exact strings occurring once: {diagnostics['fraction_exact_singletons']:.1%}",
                  f"- Exact unanimous agreement across LLMs: {diagnostics['unanimity_exact_fraction']:.1%} "
                  "(conditional on all sources being present)",
                  "", "## Interpretation and limitations", "",
                  "- `exact_vote` and `normalized_vote` create a variable number of clusters; "
                  "they are descriptive open-vocabulary label-ID baselines, not K=20 clustering algorithms.",
                  "- `discrete_mustlink_kmeans` uses a fixed K and a soft same-string must-link penalty "
                  "over the original item embeddings. Labels are not embedded or matched by meaning.",
                  "- Text normalization handles case, whitespace and punctuation only; it does not infer synonyms.",
                  "- NMI can be inflated by high-cardinality partitions. Prefer ARI and AMI when interpreting direct label IDs.",
                  "- These deliberately simple controls do not establish that all discrete weak-supervision "
                  "methods are poor. They only quantify these specific baselines.",
                  "- The must-link penalty coefficient is prespecified, not tuned to the real-world ground truth."])
    (out / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    console.print(f"[bold green]SAVED[/bold green] {out.resolve()} (metrics, summary, diagnostics, predictions, report)")
    return report, diagnostics


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()
