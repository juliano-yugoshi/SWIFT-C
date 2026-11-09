#!/usr/bin/env python3
"""Prepare paper result tables from existing SWIFT-C outputs.

Reuses completed run CSVs; never fits a model or alters predictions.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

FAMILIES = {
    'input':('Description reference','Description K-means'),
    'spherical_input':('Description reference','Spherical K-means on descriptions'),
    'spectral_input':('Description reference','Spectral clustering on descriptions'),
    'eac_classic':('Description reference','EAC on descriptions'),
    'dec_input':('Description reference','DEC on descriptions'),
    'discrete_mustlink_kmeans':('Discrete constraints','Discrete must-link K-means'),
    'labels_only':('Representation','K-means on weak-label embeddings'),
    'fusion':('Representation','Semantic fusion K-means'),
    'spherical_fusion':('Representation','Spherical K-means on fusion'),
    'hdbscan_fusion':('Representation','HDBSCAN on fusion'),
    'spectral_fusion':('Graph','Spectral clustering on fusion'),
    'eac_multiview':('Graph','Multiview evidence accumulation'),
    'eac_uniform':('Graph','Uniform evidence refinement'),
    'eac_adaptive':('Graph','Adaptive evidence refinement'),
    'dec_fusion':('Latent','DEC on semantic fusion'),
}

def summarize(df, keys, cols, stem):
    agg=df.groupby(keys,dropna=False)[cols].agg(['mean','std']).reset_index()
    agg.columns=['_'.join(c).rstrip('_') if isinstance(c,tuple) else c for c in agg.columns]
    return agg

def report(workspace:Path):
    realpath=workspace/'realworld/runs.csv'
    synpath=workspace/'synthetic/runs.csv'
    discpath=workspace/'discrete/metrics.csv'
    retpath=workspace/'retrieval/ranking_per_run.csv'
    if not any(p.exists() for p in (realpath,synpath,discpath,retpath)):
        raise FileNotFoundError('No completed results found in workspace')
    out=workspace/'tables';out.mkdir(parents=True,exist_ok=True)
    exports=[]
    if synpath.exists():
        syn=pd.read_csv(synpath)
        if 'tier' in syn: syn=syn[syn.tier.isin(['control','hard'])]
        syn=syn[(syn.source_set=='all')&(syn.k_factor==1.0)]
        per_family=syn.groupby(['scenario','tier','method'],as_index=False)[['ari','nmi']].mean()
        summary=summarize(per_family,['method'],['ari','nmi'],'synthetic')
        hard=summarize(per_family[per_family.tier=='hard'],['method'],['ari','nmi'],'hard')
        hard=hard.rename(columns={k:'hard_'+k for k in hard.columns if k!='method'})
        summary=summary.merge(hard,on='method',how='left')
        summary.to_csv(out/'synthetic_by_method.csv',index=False)
        per_family.to_csv(out/'synthetic_by_scenario.csv',index=False)
        exports.append('synthetic_by_method.csv')
    clustering=[]
    if realpath.exists():
        real=pd.read_csv(realpath)
        real=real[(real.data_kind=='real')&(real.source_set=='all')&(real.k==20)]
        clustering.append(real[['method','seed','ari','nmi']])
    if discpath.exists():
        disc=pd.read_csv(discpath)
        disc=disc[disc.method=='discrete_mustlink_kmeans']
        clustering.append(disc[['method','seed','ari','nmi']])
    if clustering:
        combined=pd.concat(clustering,ignore_index=True)
        summary=summarize(combined,['method'],['ari','nmi'],'real')
        summary.to_csv(out/'clustering_by_method.csv',index=False)
        exports.append('clustering_by_method.csv')
    if retpath.exists():
        rankings=pd.read_csv(retpath)
        if 'source_set' in rankings: rankings=rankings[rankings.source_set=='all']
        if 'k' in rankings: rankings=rankings[rankings.k==20]
        keys=['method']
        if 'top_k' in rankings: keys.append('top_k')
        metrics=[c for c in ('precision_at_k','recall_at_k','map_at_k') if c in rankings]
        if not metrics:
            # Use the existing aggregate if the installed evaluator uses a long column schema.
            ranksummary=workspace/'retrieval/ranking_summary.csv'
            if ranksummary.exists():
                pd.read_csv(ranksummary).to_csv(out/'retrieval_by_method.csv',index=False)
                exports.append('retrieval_by_method.csv')
        else:
            summary=summarize(rankings,keys,metrics,'retrieval')
            summary.to_csv(out/'retrieval_by_method.csv',index=False)
            exports.append('retrieval_by_method.csv')
    (out/'README.md').write_text('# SWIFT-C article tables\n\nTables are computed from saved experiments.\n\n'+'\n'.join(exports)+'\n')
    print('TABLES',out.resolve())
    for e in exports: print(' ',e)

def main(argv=None):
    p=argparse.ArgumentParser()
    p.add_argument('--workspace',type=Path,required=True)
    args=p.parse_args(argv)
    report(args.workspace)

if __name__=='__main__':main()
