"""Post-hoc reporting; ground truth never enters clustering or EAC fitting."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import adjusted_rand_score

BASELINE='fusion'


def write_report(runs_path,out_dir,title='Clustering experiment'):
    out_dir=Path(out_dir);out_dir.mkdir(parents=True,exist_ok=True)
    df=pd.read_csv(runs_path)
    if df.empty:raise ValueError('No completed runs')
    metrics=[c for c in ('ari','nmi','ami','silhouette_fusion','noise_rate','n_clusters_found','duration_s') if c in df]
    keys=[c for c in ('data_kind','source_set','k_factor','k') if c in df]
    summary=df.groupby(keys+['method'],dropna=False)[metrics].agg(['mean','std','count']).reset_index()
    summary.columns=['_'.join([p for p in t if p]) if isinstance(t,tuple) else t for t in summary.columns]
    summary.to_csv(out_dir/'summary.csv',index=False)
    # Every scenario family has the same macro-level influence; seed repeats
    # and synthetic generations are nested repetitions, not extra datasets.
    macro=(df.groupby(['scenario','method'],dropna=False)[[c for c in ['ari','nmi','ami'] if c in df]].mean()
             .groupby('method').agg(['mean','std','count']).reset_index())
    macro.columns=['_'.join([p for p in t if p]) if isinstance(t,tuple) else t for t in macro.columns]
    macro.to_csv(out_dir/'macro_by_scenario.csv',index=False)
    try:
        macro.to_latex(out_dir/'macro_by_scenario.tex',index=False,float_format='%.4f',escape=True)
    except Exception:pass

    blocks=[c for c in ('scenario','generation_seed','source_set','k_factor','k','seed') if c in df]
    strata=[('overall',df)]
    strata_columns=['source_set']
    if str(df['data_kind'].iloc[0])=='real':strata_columns.append('k')
    elif df['k_factor'].nunique()>1:strata_columns.append('k_factor')
    if all(c in df for c in strata_columns):
        for identifier,part in df.groupby(strata_columns,dropna=False):
            key_values=identifier if isinstance(identifier,tuple) else (identifier,)
            stratum='; '.join(f'{c}={v}' for c,v in zip(strata_columns,key_values))
            strata.append((stratum,part))
    comparisons=[];deltas=[]
    for stratum,subdf in strata:
        wide=subdf.pivot_table(index=blocks,columns='method',values=metrics,aggfunc='mean')
        for method in sorted(set(subdf.method)-{BASELINE}):
            if ('ari',method) not in wide or ('ari',BASELINE) not in wide:continue
            subset=wide[[('ari',method),('ari',BASELINE)]].dropna()
            if subset.empty:continue
            d=(subset[('ari',method)]-subset[('ari',BASELINE)]).reset_index(name='delta_ari')
            d['method']=method;d['stratum']=stratum
            if ('nmi',method) in wide and ('nmi',BASELINE) in wide:
                d['delta_nmi']=(wide[('nmi',method)]-wide[('nmi',BASELINE)]).reindex(subset.index).to_numpy()
            else:d['delta_nmi']=np.nan
            deltas.append(d)
            # Each scenario family, not each seed/realization/edge, is an independent block.
            if 'scenario' in d.columns:
                family=d.groupby('scenario')[['delta_ari','delta_nmi']].mean()
            else:family=d[['delta_ari','delta_nmi']]
            diff=family.delta_ari.to_numpy()
            if len(diff)>1:
                rng=np.random.default_rng(9031)
                boot=np.mean(rng.choice(diff,size=(3000,len(diff)),replace=True),axis=1)
                lo,hi=np.quantile(boot,[.025,.975])
                try:pv=float(wilcoxon(diff).pvalue) if np.any(abs(diff)>1e-12) else 1.
                except ValueError:pv=np.nan
            else:lo=hi=pv=np.nan
            comparisons.append(dict(stratum=stratum,method=method,baseline=BASELINE,n_blocks=len(diff),
                                    mean_delta_ari=float(np.mean(diff)),
                                    mean_delta_nmi=float(family.delta_nmi.mean()),
                                    delta_ari_ci95_lo=float(lo),delta_ari_ci95_hi=float(hi),wilcoxon_p=pv))
    compares=pd.DataFrame(comparisons)
    if len(compares):
        compares['holm_p']=np.nan
        for stratum,indices in compares.groupby('stratum').groups.items():
            subset=compares.loc[indices,'wilcoxon_p'].to_numpy(float)
            valid=np.flatnonzero(np.isfinite(subset))
            if len(valid):
                order=valid[np.argsort(subset[valid])]
                adj=np.maximum.accumulate([(len(order)-j)*subset[i] for j,i in enumerate(order)])
                for i,value in zip(order,adj):compares.loc[indices[i],'holm_p']=min(float(value),1.)
    compares.to_csv(out_dir/'paired_comparison.csv',index=False)
    if len(compares):
        try:compares.to_latex(out_dir/'paired_comparison.tex',index=False,float_format='%.4f',escape=True)
        except Exception:pass
    if deltas:pd.concat(deltas,ignore_index=True).to_csv(out_dir/'deltas.csv',index=False)
    lines=[f'# {title}','',f'Observations: {len(df)}. ARI/NMI are for evaluation only.','',
           '## Per-configuration summary','',summary.round(4).to_markdown(index=False),'',
           '## Paired comparisons against 65/35 fusion','',
           compares.round(4).to_markdown(index=False) if len(compares) else 'Not enough paired runs.','']
    if 'tier' in df:
        hard=df.loc[df.tier.astype(str).eq('hard')]
        if len(hard):
            view=hard.groupby('method')[['ari','nmi']].mean().round(4).reset_index()
            lines+=['## Hard-scenario summary','',view.to_markdown(index=False),'']
    lines += ['## Evaluation protocol','',
              'The macro table averages over scenario families, not instance counts.',
              'Holm-adjusted p-values control the tested pairwise comparisons within each stratum.',
              'Cluster-family bootstrap CI resamples scenario families, not individual seeds or edges.',
              'Model parameters must be frozen before confirmation on new generator seeds.',
              'Real-world seed repetitions are not independent datasets; they are stability checks.',
              'Do not select K, source sets or smoothing hyperparameters by maximizing test ARI/NMI.',
              'A confidence interval containing zero is not evidence of reliable superiority.','']
    (out_dir/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    return summary,compares


def write_stability(runs_path,out_path):
    """Mean pairwise ARI between clustering seeds, *without* reference labels."""
    from itertools import combinations
    df=pd.read_csv(runs_path)
    cols=['data_kind','scenario','generation_seed','source_set','k','method']
    if 'prediction_file' not in df or not all(c in df for c in cols):return pd.DataFrame()
    rows=[]
    for keys,group in df.groupby(cols,dropna=False):
        predictions={}
        for _,r in group.iterrows():
            p=Path(str(r['prediction_file']))
            if not p.is_absolute():p=Path(runs_path).parent/p
            if p.is_file():
                with np.load(p) as f: predictions[int(r['seed'])]=f['pred'].copy()
        values=[adjusted_rand_score(a,b) for (_,a),(_,b) in combinations(sorted(predictions.items()),2)]
        if not values:continue
        rec=dict(zip(cols,keys))
        rec.update(seed_pairs=len(values),seed_stability_ari=float(np.mean(values)),
                   seed_stability_std=float(np.std(values)))
        rows.append(rec)
    out=pd.DataFrame(rows)
    Path(out_path).parent.mkdir(parents=True,exist_ok=True)
    out.to_csv(out_path,index=False)
    return out
