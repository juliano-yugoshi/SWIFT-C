"""CLI for evaluating saved clustering predictions on items with reference categories."""
from __future__ import annotations
import argparse
import hashlib
import json
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from .core import evaluate_clustering, evaluate_topk


def read_reference(path: Path, id_col: str, target_col: str) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}")
    if path.suffix.lower() == '.zip':
        with zipfile.ZipFile(path) as archive:
            names = [n for n in archive.namelist() if n.lower().endswith('.csv') and not n.startswith('__MACOSX/')]
            if len(names) != 1:
                raise ValueError(f"Expected exactly one CSV in ZIP, found {len(names)}: {names[:5]}")
            with archive.open(names[0]) as fh:
                df = pd.read_csv(fh, usecols=[id_col, target_col], dtype=str,
                                 keep_default_na=False, encoding='utf-8-sig')
    else:
        df = pd.read_csv(path, usecols=[id_col, target_col], dtype=str,
                         keep_default_na=False, encoding='utf-8-sig')
    if df[id_col].duplicated().any():
        raise ValueError(f"Duplicate IDs in reference: {id_col}")
    if (~df[target_col].astype(bool)).any():
        raise ValueError(f"Missing/empty reference categories: {target_col}")
    return df


def require_aligned(reference: pd.DataFrame, ids: np.ndarray, id_col: str, target_col: str) -> np.ndarray:
    """Strict identifier alignment. No positional fallback."""
    ids = np.asarray(ids).astype(str)
    if len(set(ids)) != len(ids):
        raise ValueError("Prediction IDs must be unique")
    target = reference.set_index(id_col)[target_col]
    if not set(ids).issubset(target.index):
        extra = list(set(ids) - set(target.index))[:5]
        raise ValueError(f"Prediction contains IDs missing from reference, e.g. {extra}")
    return target.loc[ids].to_numpy(dtype=str)


def _read_pred(path: Path, id_col: str) -> tuple[np.ndarray, np.ndarray]:
    # The clustering writer may save IDs from pandas as dtype=object in NPZ.
    # Such NumPy files require allow_pickle=True. Only load trusted, locally created experiment caches.
    with np.load(path, allow_pickle=True) as file:
        if 'pred' not in file or 'ids' not in file:
            raise ValueError(f"Prediction NPZ must have both 'pred' and 'ids': {path}")
        return file['ids'].astype(str), file['pred'].copy()


def _load_embedding_cache(path: Path) -> tuple[np.ndarray, np.ndarray, Path]:
    paths = [path] if path.is_file() else sorted(path.glob('embeddings_*.npz'))
    if not paths:
        raise FileNotFoundError(f"No embedding cache found in {path}")
    if len(paths) > 1:
        raise ValueError(f"Multiple embedding caches in {path}; provide the exact .npz file")
    # The embedding cache may also store pandas IDs as an object array.
    # Use this only with locally generated, trusted NPZ files.
    with np.load(paths[0], allow_pickle=True) as file:
        if 'ids' not in file or 'X' not in file:
            raise ValueError("Embedding cache must contain ids and X")
        return file['ids'].astype(str), file['X'].astype(np.float32), paths[0]


def _id_signature(ids: np.ndarray) -> str:
    return hashlib.sha256('\x00'.join(sorted(ids.tolist())).encode()).hexdigest()[:16]


def _normalize_list(csv: str | None, typ=str) -> set | None:
    return {typ(x.strip()) for x in csv.split(',') if x.strip()} if csv else None


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        prog='swiftc retrieval',
        description='Post-hoc item-as-query retrieval evaluation for saved clustering predictions.')
    p.add_argument('--results', nargs='*', default=[], type=Path,
                   help='One or more saved clustering result directories (or runs.csv paths).')
    p.add_argument('--input', required=True, type=Path, help='Reference CSV or ZIP with item IDs and evaluation categories.')
    p.add_argument('--output', default=Path('results/video_recommend_eval'), type=Path)
    p.add_argument('--discrete', type=Path, help='Optional discrete baseline directory with predictions.csv.')
    p.add_argument('--embeddings', type=Path, default=None,
                   help='Optional exact embeddings_*.npz or cache directory. Enables input-embedding nearest-neighbor ranking within clusters.')
    p.add_argument('--topk', default='10,20', help='Ranking cutoffs, used only with --embeddings (default 10,20).')
    p.add_argument('--methods', default=None, help='Optional comma-separated methods to include.')
    p.add_argument('--source-sets', default=None, help='Optional comma-separated source_set filter, e.g. all.')
    p.add_argument('--k', default=None, help='Optional comma-separated K filter, e.g. 20.')
    p.add_argument('--seeds', default=None, help='Optional comma-separated seed filter.')
    p.add_argument('--id-column', default='video_id')
    p.add_argument('--target-column', default='category')
    p.add_argument('--query-details', action='store_true',
                   help='Write per-video metrics (large output on full experiments).')
    p.add_argument('--strict', action='store_true', help='Fail instead of skipping broken/missing prediction files.')
    p.add_argument('--no-plots', action='store_true', help='Skip PNG figure generation.')
    return p.parse_args(argv)


def load_experiments(args) -> tuple[list[dict], list[str]]:
    records = []
    warnings = []
    desired_methods = _normalize_list(args.methods)
    desired_sources = _normalize_list(args.source_sets)
    desired_ks = _normalize_list(args.k, int)
    desired_seeds = _normalize_list(args.seeds, int)
    for result_arg in args.results:
        runs_file = result_arg if result_arg.is_file() else result_arg / 'runs.csv'
        if not runs_file.exists():
            raise FileNotFoundError(f"runs.csv not found: {runs_file}")
        frame = pd.read_csv(runs_file)
        required = {'data_kind', 'method', 'seed', 'k', 'source_set', 'prediction_file'}
        absent = required - set(frame.columns)
        if absent:
            raise ValueError(f"Missing columns in {runs_file}: {sorted(absent)}")
        frame = frame.loc[frame['data_kind'].astype(str).str.lower() == 'real'].copy()
        frame = frame.loc[frame['prediction_file'].notna()]
        frame = frame.drop_duplicates(subset=['method', 'source_set', 'k', 'seed', 'prediction_file'], keep='last')
        for r in frame.to_dict('records'):
            method = str(r['method'])
            group = str(r['source_set'])
            k = int(r['k'])
            seed = int(r['seed'])
            if desired_methods and method not in desired_methods: continue
            if desired_sources and group not in desired_sources: continue
            if desired_ks and k not in desired_ks: continue
            if desired_seeds and seed not in desired_seeds: continue
            rel = Path(str(r['prediction_file']))
            if rel.is_absolute() or '..' in rel.parts:
                msg = f"Unsafe prediction path ignored: {rel}"
                if args.strict: raise ValueError(msg)
                warnings.append(msg); continue
            pred_path = runs_file.parent / rel
            if not pred_path.is_file():
                msg = f"Missing predictions for {method} seed={seed}: {pred_path}"
                if args.strict: raise FileNotFoundError(msg)
                warnings.append(msg); continue
            records.append({'method':method,'source_set':group,'k':k,'seed':seed,
                            'path':pred_path, 'run_origin':str(runs_file.parent), 'type':'clustering'})
    if args.discrete:
        pred_file = args.discrete if args.discrete.is_file() else args.discrete / 'predictions.csv'
        if not pred_file.exists():
            raise FileNotFoundError(f"Discrete predictions not found: {pred_file}")
        discrete = pd.read_csv(pred_file)
        if args.id_column not in discrete:
            raise ValueError(f"Discrete predictions missing {args.id_column}")
        for col in discrete.columns:
            if col == args.id_column: continue
            if col in ('exact_vote', 'normalized_vote'):
                method, seed, k = col, -1, 0
            elif col.startswith('discrete_mustlink_s'):
                method = 'discrete_mustlink_kmeans'
                try: seed = int(col.rsplit('_s', 1)[1])
                except Exception: raise ValueError(f"Invalid discrete prediction name: {col}")
                # If metrics.csv is present, recover the requested K, else 20 standard.
                mp = pred_file.parent / 'metrics.csv'
                k = 20
                if mp.exists():
                    dfm = pd.read_csv(mp)
                    match = dfm[(dfm['method'] == method) & (pd.to_numeric(dfm['seed'], errors='coerce') == seed)]
                    if len(match) and pd.notna(match.iloc[0].get('k_requested')):
                        k = int(match.iloc[0]['k_requested'])
            else:
                warnings.append(f"Unrecognized discrete prediction column ignored: {col}")
                continue
            if desired_methods and method not in desired_methods: continue
            if desired_ks and k not in desired_ks: continue
            if desired_seeds and seed not in desired_seeds: continue
            if desired_sources and 'all' not in desired_sources: continue
            records.append({'method':method, 'source_set':'all', 'k':k,'seed':seed,
                            'type':'discrete', 'column':col,'path':pred_file,
                            'run_origin':str(pred_file.parent)})
    return records, warnings


def _summary_table(per_run: pd.DataFrame) -> pd.DataFrame:
    metrics = ['precision','recall','f1','macro_category_precision','macro_category_recall','macro_category_f1',
               'pair_precision','pair_recall','pair_f1','candidate_coverage','hit_coverage',
               'mean_candidates','precision_lift','singleton_query_fraction']
    metrics += sorted(c for c in per_run.columns if ('_at_' in c))
    metrics = [m for m in metrics if m in per_run]
    agg = {'n_runs':('f1', 'size')}
    for m in metrics:
        agg[m+'_mean'] = (m, 'mean')
        agg[m+'_std'] = (m, 'std')
    summary = per_run.groupby(['method','source_set','k'], dropna=False).agg(**agg).reset_index()
    return summary.sort_values(['k','source_set','f1_mean'],ascending=[True,True,False])


def _paired(per_run: pd.DataFrame) -> pd.DataFrame:
    """Only compare identical video sets, K, source set and seeds."""
    metrics = ['precision','recall','f1','macro_category_f1','pair_f1','candidate_coverage']
    metrics += sorted(c for c in per_run.columns if '_at_' in c)
    metrics = [m for m in metrics if m in per_run]
    records = []
    keys = ['run_origin','source_set','k','seed','id_signature']
    main = per_run.loc[per_run['method'] != 'fusion']
    baselines = per_run.loc[per_run['method'] == 'fusion']
    if baselines.empty: return pd.DataFrame()
    for _, a in main.iterrows():
        relevant = baselines
        for key in keys:
            relevant = relevant.loc[relevant[key] == a[key]]
        if relevant.empty:
            # Discrete methods are from a separate run_origin; pair on same IDs/K/source/seed.
            if a['type'] == 'discrete':
                relevant = baselines
                for key in ('source_set','k','seed','id_signature'):
                    relevant = relevant.loc[relevant[key] == a[key]]
        if len(relevant) != 1: continue
        b = relevant.iloc[0]
        row = {'method':a['method'], 'source_set':a['source_set'],'k':a['k'], 'seed':a['seed']}
        for m in metrics: row['delta_'+m] = float(a[m] - b[m])
        records.append(row)
    return pd.DataFrame(records)



def _ranked_long_tables(per_run: pd.DataFrame, summary: pd.DataFrame, paired: pd.DataFrame,
                        topks: tuple[int, ...]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Provide explicit, readable @K tables in addition to wide per-run/summary CSVs."""
    run_rows = []
    summary_rows = []
    paired_rows = []
    for k in topks:
        metrics = ('precision', 'recall', 'map', 'ndcg')
        run_cols = ['method', 'source_set', 'k', 'seed']
        for _, r in per_run.iterrows():
            row = {c: r[c] for c in run_cols}
            row['top_k'] = k
            for metric in metrics:
                row[f'{metric}_at_k'] = r[f'{metric}_at_{k}']
            run_rows.append(row)
        for _, r in summary.iterrows():
            row = {c: r[c] for c in ('method', 'source_set', 'k', 'n_runs')}
            row['top_k'] = k
            for metric in metrics:
                row[f'{metric}_at_k_mean'] = r[f'{metric}_at_{k}_mean']
                row[f'{metric}_at_k_std'] = r[f'{metric}_at_{k}_std']
            summary_rows.append(row)
        if not paired.empty:
            for _, r in paired.iterrows():
                row = {c: r[c] for c in ('method','source_set','k','seed')}
                row['top_k'] = k
                for metric in metrics:
                    row[f'delta_{metric}_at_k'] = r[f'delta_{metric}_at_{k}']
                paired_rows.append(row)
    return pd.DataFrame(run_rows), pd.DataFrame(summary_rows), pd.DataFrame(paired_rows)


def _plots(summary: pd.DataFrame, out: Path) -> None:
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('PLOTS skipped (install matplotlib, or use --no-plots)')
        return
    subset = summary
    if subset.empty: return
    fig, ax = plt.subplots(figsize=(8,5))
    ax.scatter(subset['precision_mean'],subset['recall_mean'],s=70)
    for _, r in subset.iterrows():
        ax.annotate(r['method'], (r['precision_mean'],r['recall_mean']),xytext=(4,4),textcoords='offset points',fontsize=8)
    ax.set(xlabel='Cluster Precision (mean per video)',ylabel='Cluster Recall (mean per video)',title='Cluster-restricted recommendation quality')
    ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(out/'precision_recall.png',dpi=180)
    plt.close(fig)
    fig, ax=plt.subplots(figsize=(9, max(3,0.38*len(subset)+1)))
    ordered=subset.sort_values('f1_mean')
    ax.barh(ordered['method'],ordered['f1_mean'])
    ax.set(xlabel='Mean per-video Cluster F1',title='Cluster recommendation F1')
    fig.tight_layout()
    fig.savefig(out/'cluster_f1.png',dpi=180)
    plt.close(fig)


def _report(summary:pd.DataFrame, comparisons:pd.DataFrame, n_runs:int, warnings:list[str],
            ref:Path, embedding_cache:Path | None) -> str:
    lines = ['# Cluster-based item recommendation', '',
             'Each video is a query. Its recommendations are all *other* videos in the same predicted cluster.',
             'The only relevance proxy is equality of the reference category. The reference categories are used exclusively for evaluation.',
             'No clustering or encoder execution is performed.', '',
             f'- Dataset: `{ref}`', f'- Prediction runs evaluated: {n_runs}',
             f'- Ranking from cached, frozen **input embeddings only**: `{embedding_cache}`' if embedding_cache else
             '- No ranking features provided: evaluated the **unordered complete cluster**, not P@10/mAP@10.',
             '', '## Definitions', '',
             '- Cluster Precision: mean over video queries of relevant recommended videos / all recommended videos.',
             '- Cluster Recall: mean over video queries of relevant recommended videos / all other videos in the same true category.',
             '- Cluster F1: mean of the **per-query** harmonic mean (not the harmonic mean of aggregate P and R).',
             '- Singleton and noise videos are treated as queries with zero recommendations and P=R=F1=0.',
             '- Macro Category P/R/F1: equally-weighted mean of per-category query averages.',
             '- Pair P/R/F1: micro-aggregated ratios over all ordered query-candidate pairs.',
             '- Candidate Coverage: proportion of queries with at least one candidate. Hit Coverage: proportion with at least one relevant candidate.',
             '- Random Precision: average same-category prevalence of each query across the full dataset; lift = precision / random precision.',
             '- If ranking enabled, candidates are sorted using cosine similarity of **input embeddings**, identical across methods. Missing top-k items are nonrelevant.',
             '- mAP@k is the mean truncated AP@k, normalized by min(k, number of relevant corpus items); nDCG@k uses binary category relevance.',
             '', '## Cluster-level comparison across completed runs', '',
             '| Method | Runs | Cluster Precision | Cluster Recall | Cluster F1 | Candidate coverage |',
             '|---|---:|---:|---:|---:|---:|']
    selected = summary.sort_values('f1_mean',ascending=False)
    for _, row in selected.iterrows():
        lines.append(f"| {row['method']} | {int(row['n_runs'])} | {row['precision_mean']:.4f} | {row['recall_mean']:.4f} | {row['f1_mean']:.4f} | {row['candidate_coverage_mean']:.2%} |")
    if selected.empty: lines.append('| *(No completed runs)* | | | | | |')
    ranked_cols = sorted((c for c in summary if c.startswith('precision_at_') and c.endswith('_mean')),
                         key=lambda c: int(c.split('_at_')[1].split('_')[0]))
    if ranked_cols:
        lines += ['', '## Ranked Top-K recommendation results', '',
                  'For each query video, rank **other videos in its cluster** by cosine similarity '
                  'of the frozen original item embeddings. Relevance means same reference category. '
                  'All methods use the **same ranking embeddings**.', '',
                  '**Important:** these are genuine ranked @K metrics; the cluster-level Precision/Recall/F1 '
                  'in the preceding table are calculated on the **entire unordered cluster**.', '']
        for column in ranked_cols:
            top_k = int(column.split('_at_')[1].split('_')[0])
            lines += [f'### Top-{top_k}', '',
                      f'| Method | Runs | Precision@{top_k} | Recall@{top_k} | mAP@{top_k} | nDCG@{top_k} |',
                      '|---|---:|---:|---:|---:|---:|']
            display = summary[(summary['source_set']=='all') & (summary['k']==20)].copy()
            display = display.sort_values(f'map_at_{top_k}_mean', ascending=False)
            for _,r in display.iterrows():
                lines.append(f"| {r['method']} | {int(r['n_runs'])} | "
                             f"{r[f'precision_at_{top_k}_mean']:.4f} | "
                             f"{r[f'recall_at_{top_k}_mean']:.4f} | "
                             f"{r[f'map_at_{top_k}_mean']:.4f} | "
                             f"{r[f'ndcg_at_{top_k}_mean']:.4f} |")
            if display.empty: lines.append('| *(No K=20/all runs)* | | | | | |')
            lines += ['', f'All methods/source/K combinations: `ranking_summary.csv` '
                       '(filter by `top_k`), with per-seed values in `ranking_per_run.csv`.', '']
    else:
        lines += ['', '## Ranked Top-K recommendation results', '',
                  '**NOT CALCULATED.** No usable embedding cache was provided via `--embeddings`.',
                  'The cluster-level Precision/Recall/F1 values above are **not** Precision@10 '
                  'or Precision@20. Supply `--embeddings` and optionally `--topk 10,20` to compute ranked retrieval.', '']
    if not comparisons.empty:
        lines += ['', '## Matched comparisons against fusion', '',
                  'Pairing requires matching source, K, seed and item IDs. Differences are descriptive, not statistical significance claims.',
                  '', '| Method | n paired runs | Δ Cluster F1 | Δ Precision | Δ Recall |', '|---|---:|---:|---:|---:|']
        paired = comparisons.groupby('method',as_index=False).agg(n=('delta_f1','size'),
                         df1=('delta_f1','mean'),dp=('delta_precision','mean'),dr=('delta_recall','mean')).sort_values('df1',ascending=False)
        for _,r in paired.iterrows():
            lines.append(f"| {r['method']} | {int(r['n'])} | {r['df1']:+.4f} | {r['dp']:+.4f} | {r['dr']:+.4f} |")
    lines += ['', '## Limitations', '',
              '- Category agreement is a proxy for relevance, not behavioral relevance or human similarity judgments.',
              '- Cluster metrics do not depend on a ranker. Ranked metrics additionally depend on frozen input embeddings.',
              '- Methods with different numbers of clusters or a large number of singletons must be interpreted with coverage and recall.',
              '- Three seeds on one dataset are not independent datasets; use descriptive paired differences and do not assert significance.',
              '- No ground truth is used to select, refine, or reorder clusters.']
    if warnings:
        lines += ['', '## Skipped/missing predictions', '']
        lines += [f'- {w}' for w in warnings[:50]]
        if len(warnings)>50: lines.append(f'- ... and {len(warnings)-50} more (see warnings.txt)')
    return '\n'.join(lines) + '\n'


def run(args) -> tuple[pd.DataFrame,pd.DataFrame]:
    t0=time.perf_counter()
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    reference=read_reference(args.input,args.id_column,args.target_column)
    print(f'DATA {len(reference):,} items; {reference[args.target_column].nunique()} categories')
    if not args.results and not args.discrete:
        raise ValueError('Provide --results and/or --discrete')
    records,warnings=load_experiments(args)
    if not records:
        raise ValueError('No usable predictions. Inspect --results and --discrete paths and filters.')
    embeddings=None; emb_index=None; emb_file=None
    if args.embeddings:
        emb_ids,embeddings,emb_file = _load_embedding_cache(args.embeddings)
        if len(set(emb_ids))!=len(emb_ids):raise ValueError('Duplicate IDs in embedding cache')
        emb_index=pd.Series(np.arange(len(emb_ids)),index=emb_ids)
        print(f'CACHE {emb_file} (N={len(emb_ids)}, D={embeddings.shape[1]})')
    if embeddings is None:
        print('RANKING NOT COMPUTED: --embeddings was not provided. '
              'Cluster Precision/Recall/F1 use ALL cluster members; --topk does nothing without embeddings.')
    else:
        print(f'RANKING ENABLED: Top-K={args.topk}; cosine similarity of frozen input embeddings; '
              'candidates restricted to the same cluster.')
    topks=tuple(int(x) for x in args.topk.split(',') if x.strip())
    disccache={}
    rows=[]; per_categories=[]
    query_path=out/'per_query.csv'
    if args.query_details and query_path.exists():query_path.unlink()
    query_written=False
    for j, rec in enumerate(records,1):
        tic=time.perf_counter()
        try:
            if rec['type']=='clustering':
                ids,pred=_read_pred(rec['path'],args.id_column)
            else:
                csv=rec['path']
                if csv not in disccache: disccache[csv]=pd.read_csv(csv,dtype={args.id_column:str})
                d=disccache[csv]
                ids=d[args.id_column].astype(str).to_numpy()
                pred=d[rec['column']].to_numpy()
            y=require_aligned(reference,ids,args.id_column,args.target_column)
            scores=evaluate_clustering(ids,y,pred,method=rec['method'],query_details=args.query_details)
            ranking={};ranking_queries=pd.DataFrame()
            if embeddings is not None:
                if not set(ids).issubset(emb_index.index):
                    raise ValueError('Embedding cache missing prediction IDs')
                xi=embeddings[emb_index.loc[ids].to_numpy()]
                ranking,ranking_queries=evaluate_topk(ids,y,pred,xi,topks,query_details=args.query_details)
            row={key:rec[key] for key in ('method','source_set','k','seed','type','run_origin')}
            row.update({'id_signature':_id_signature(ids),'prediction_file':str(rec['path']),
                        **scores.summary,**ranking})
            rows.append(row)
            cats=scores.category_table.copy()
            for key in ('method','source_set','k','seed','type','run_origin'):
                cats[key]=rec[key]
            per_categories.append(cats)
            if args.query_details:
                qs=scores.query_table.copy()
                if not ranking_queries.empty:
                    qs=qs.merge(ranking_queries,on='video_id',how='left',validate='one_to_one')
                for key in ('method','source_set','k','seed','type','run_origin'):
                    qs[key]=rec[key]
                qs.to_csv(query_path,mode='a',header=not query_written,index=False)
                query_written=True
            ranked_log = (' ' + ' '.join(
                f"P@{k}={row[f'precision_at_{k}']:.4f} "
                f"mAP@{k}={row[f'map_at_{k}']:.4f} "
                f"nDCG@{k}={row[f'ndcg_at_{k}']:.4f}"
                for k in topks)) if embeddings is not None else ''
            print(f"[{j}/{len(records)}] {rec['method']} {rec['source_set']} K={rec['k']} seed={rec['seed']} "
                  f"ClusterP={row['precision']:.4f} ClusterR={row['recall']:.4f} "
                  f"ClusterF1={row['f1']:.4f}{ranked_log} "
                  f"({time.perf_counter()-tic:.1f}s)")
        except (KeyError, ValueError, OSError, IndexError) as exc:
            msg=f"Failed {rec['method']} from {rec['path']}: {type(exc).__name__}: {exc}"
            if args.strict:raise RuntimeError(msg) from exc
            warnings.append(msg)
            print('WARNING', msg)
    if not rows:
        raise ValueError('All predictions failed validation; check warnings and files')
    per_run=pd.DataFrame(rows)
    per_run.to_csv(out/'per_run.csv',index=False)
    summary=_summary_table(per_run)
    summary.to_csv(out/'summary.csv',index=False)
    paired=_paired(per_run)
    if not paired.empty:paired.to_csv(out/'paired_vs_fusion.csv',index=False)
    if embeddings is not None:
        ranked_runs, ranked_summary, ranked_paired = _ranked_long_tables(per_run, summary, paired, topks)
        ranked_runs.to_csv(out/'ranking_per_run.csv', index=False)
        ranked_summary.to_csv(out/'ranking_summary.csv', index=False)
        if not ranked_paired.empty:
            ranked_paired.to_csv(out/'ranking_paired_vs_fusion.csv', index=False)
    else:
        # Avoid confusing stale Top-K artifacts when reusing a previous output directory.
        for old in ('ranking_per_run.csv', 'ranking_summary.csv', 'ranking_paired_vs_fusion.csv'):
            (out/old).unlink(missing_ok=True)
    cat=pd.concat(per_categories,ignore_index=True)
    cat.to_csv(out/'per_category.csv',index=False)
    if warnings:(out/'warnings.txt').write_text('\n'.join(warnings)+'\n',encoding='utf-8')
    (out/'REPORT.md').write_text(_report(summary,paired,len(rows),warnings,args.input,emb_file),encoding='utf-8')
    metadata={'dataset':str(args.input),'n_reference_items':len(reference),
              'n_runs':len(rows),'n_skipped':len(warnings),
              'ground_truth_used_only_for_evaluation':True,
              'ranked_retrieval_enabled':bool(args.embeddings),
              'ranker':'cosine similarity of frozen input embeddings' if args.embeddings else None,
              'topk':topks if args.embeddings else None,
              'duration_s':time.perf_counter()-t0}
    (out/'metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    if not args.no_plots: _plots(summary,out)
    print(f"DONE {len(rows)} runs, {len(warnings)} warnings, {time.perf_counter()-t0:.1f}s. Outputs: {out.resolve()}")
    return per_run,summary


def main(argv=None):
    run(parse_args(argv))


if __name__ == '__main__':main()
