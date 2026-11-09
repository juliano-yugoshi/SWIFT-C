"""Publication-oriented plots. Only uses completed post-hoc evaluation tables."""
from pathlib import Path
import numpy as np
import pandas as pd

def plot_evaluations(runs_path,output,source='all',k_factor=1.,method='eac_adaptive'):
    try:
        import matplotlib.pyplot as plt
    except ImportError as ex:
        raise RuntimeError('Plotting requires: pip install -e ".[viz]"') from ex
    df=pd.read_csv(runs_path)
    df=df[df.source_set.eq(source)]
    if 'k_factor' in df:df=df[np.isclose(df.k_factor.astype(float),k_factor)]
    if not len(df): raise ValueError(f'No data for source={source} / K factor={k_factor}')
    out=Path(output);out.mkdir(exist_ok=True,parents=True)
    data=df.groupby('method')[['ari','nmi']].mean().sort_values('ari',ascending=False)
    fig,ax=plt.subplots(figsize=(8,4))
    ix=np.arange(len(data));bar_w=.38
    ax.bar(ix-bar_w/2,data.ari.values,bar_w,label='ARI')
    ax.bar(ix+bar_w/2,data.nmi.values,bar_w,label='NMI')
    ax.set_xticks(ix,data.index,rotation=25,ha='right')
    ax.set_ylabel('Score');ax.legend();ax.set_title(f'Clustering results | sources={source} | K factor={k_factor}')
    fig.tight_layout();fig.savefig(out/'method_comparison.png',dpi=200);fig.savefig(out/'method_comparison.svg');plt.close(fig)
    keys=[c for c in ['scenario','generation_seed','seed','k','k_factor','source_set'] if c in df]
    piv=df.pivot_table(index=keys,columns='method',values='ari',aggfunc='mean')
    if method not in piv or 'fusion' not in piv:return
    data=(piv[method]-piv['fusion']).groupby('scenario').mean().sort_values()
    fig,ax=plt.subplots(figsize=(10,max(4,len(data)*.25)))
    ax.barh(data.index,data.values)
    ax.axvline(0,linewidth=.9)
    ax.set_xlabel('Δ ARI relative to 65/35 fusion')
    ax.set_title(f'{method} vs fusion | {source}')
    fig.tight_layout();fig.savefig(out/'scenario_delta_ari.png',dpi=200);fig.savefig(out/'scenario_delta_ari.svg');plt.close(fig)
