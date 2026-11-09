"""Experiment orchestration, caching, Rich status, resumable observations."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from contextlib import contextmanager
import hashlib,json,time,traceback,csv
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics import adjusted_rand_score,normalized_mutual_info_score,adjusted_mutual_info_score,silhouette_score
from threadpoolctl import threadpool_limits
from rich.console import Console
from rich.progress import Progress,SpinnerColumn,BarColumn,TextColumn,TimeElapsedColumn,TimeRemainingColumn,MofNCompleteColumn
from rich.table import Table
from .core import validate_arrays,select_sources,evidence_graph,method_embeddings,kmeans_predict,consensus,fused
from .baselines import spherical_kmeans,knn_graph,spectral_features,spectral_eac_features,density_predict,eac_average_linkage
from .report import write_report,write_stability

CORE_METHODS=('input','labels_only','fusion','eac_uniform','eac_adaptive')
QUICK_METHODS=CORE_METHODS+('spherical_input','spherical_fusion','spectral_input','eac_classic','eac_multiview')
PAPER_METHODS=QUICK_METHODS+('spectral_fusion','hdbscan_fusion','dec_input','dec_fusion')
METHODS=PAPER_METHODS+('eac_average_linkage',)
EAC_METHODS={'eac_classic','eac_multiview','eac_uniform','eac_adaptive','eac_average_linkage'}
SPECTRAL_METHODS={'spectral_input','spectral_fusion','eac_classic','eac_multiview'}
DEEP_METHODS={'dec_input','dec_fusion'}
INPUT_ONLY_METHODS={'input','spherical_input','spectral_input','eac_classic','eac_average_linkage','dec_input'}

class Tracker:
    def __init__(self,output,progress='rich',verbose=True):
        self.output=Path(output);self.output.mkdir(parents=True,exist_ok=True)
        self.mode=progress;self.verbose=verbose
        self.console=Console(force_terminal=False,highlight=False)
        self.log_path=self.output/'run.log'
        self.jsonl=self.output/'events.jsonl'
        self.n=0
    def event(self,stage,message,**extra):
        t=time.strftime('%Y-%m-%d %H:%M:%S')
        line=f'[{t}] [{stage}] {message}'
        with self.log_path.open('a') as f:f.write(line+'\n')
        with self.jsonl.open('a') as f:f.write(json.dumps({'at':t,'stage':stage,'message':message,**extra},default=str)+'\n')
        if self.verbose:self.console.print(f'[dim]{t}[/dim] [cyan]{stage}[/cyan] {message}')
    @contextmanager
    def progress(self,total,label):
        if self.mode=='rich':
            with Progress(SpinnerColumn(),TextColumn('[bold blue]{task.description}'),BarColumn(),MofNCompleteColumn(),TimeElapsedColumn(),TimeRemainingColumn(),console=self.console,transient=False) as p:
                tid=p.add_task(label,total=total)
                yield lambda inc=1:p.advance(tid,inc)
        else:
            start=time.monotonic();done=[0]
            def tick(inc=1):
                done[0]+=inc
                if done[0]==total or (self.verbose and (done[0]==1 or done[0]%max(total//15,1)==0)):
                    dt=max(time.monotonic()-start,.001)
                    eta=dt*(total-done[0])/max(done[0],1)
                    self.event('PROGRESS',f'{label}: {done[0]}/{total} elapsed={dt:.1f}s ETA={eta:.1f}s')
            yield tick
    def table(self,records):
        if not records:return
        t=Table(title='ARI / NMI summary (evaluation only)',show_lines=False)
        for n in ('Source','K','Method','ARI','NMI','Runs'):t.add_column(n)
        for r in records:
            def value(n):
                try:return f'{float(r[n]):.4f}'
                except (KeyError,TypeError,ValueError):return 'n/a'
            t.add_row(str(r.get('source_set','all')),str(r.get('k','-')),
                      str(r.get('method','-')),value('ari_mean'),value('nmi_mean'),
                      str(r.get('ari_count','-')))
        self.console.print(t)


def source_groups(names,selector):
    names=list(names)
    if selector=='all':return {'all':list(range(len(names)))}
    items=[s.strip() for s in selector.split(',') if s.strip()]
    groups={}
    for s in items:
        if s=='all':groups['all']=list(range(len(names)))
        elif s=='each':groups.update({v:[i] for i,v in enumerate(names)})
        elif s=='pairs':
            from itertools import combinations
            groups.update({'+'.join((names[i],names[j])):[i,j] for i,j in combinations(range(len(names)),2)})
        else:
            wanted=[p.strip() for p in s.split('+')]
            missing=set(wanted)-set(names)
            if missing:raise ValueError(f'Unknown weak-label sources {missing}; available {names}')
            groups['+'.join(wanted)]=[names.index(q) for q in wanted]
    return groups


def _graph_key(data_signature,group,k,config):
    return hashlib.sha256(json.dumps([data_signature,group,k,config],sort_keys=True).encode()).hexdigest()[:20]


def check_protocol(output,protocol,resume=True):
    """Prevent accidentally merging incompatible runs in the same output directory."""
    path=Path(output)/'run_protocol.json'
    if resume and path.exists():
        previous=json.loads(path.read_text())
        if previous!=protocol:
            raise ValueError('Output directory contains results from a different protocol. '
                             'Choose a new --output or pass --no-resume to overwrite previous results.')
    path.write_text(json.dumps(protocol,indent=2,sort_keys=True,default=str)+'\n')


def append_run(record,path):
    """Durable O(1) append per method-run; makes the full grid restartable."""
    path=Path(path)
    with path.open('a',newline='',encoding='utf-8') as fp:
        writer=csv.DictWriter(fp,fieldnames=list(record))
        if fp.tell()==0:writer.writeheader()
        writer.writerow(record)


def _atomic_runs(rows,path):
    tmp=path.with_suffix('.tmp.csv')
    pd.DataFrame(rows).to_csv(tmp,index=False)
    tmp.replace(path)


def run_one_dataset(X,L,M,y,ids,names,k,seed_set,source_selector,out,tracker,dataset_id,signature,
                    eac_seed=42,factors=(4,8),reps=2,label_weight=.35,eta=.25,threshold=.75,
                    sample_ids=None,k_factor=1.,tier='real',generation_seed=0,methods=CORE_METHODS,
                    resume=True,existing=None,callback=None,
                    knn_neighbors=15,backbone_weight=.03,density_pca_dim=32,
                    dec_latent_dim=32,dec_pretrain_epochs=8,dec_epochs=15,dec_batch_size=256,
                    dec_device='cpu',compute_silhouette=True):
    """Produce predictions without reference labels; evaluate only after inference."""
    X,L,M=validate_arrays(X,L,M)
    if y is not None:y=np.asarray(y)
    unknown=set(methods)-set(METHODS)
    if unknown:raise ValueError(f'Unknown methods: {sorted(unknown)}')
    methods=list(dict.fromkeys(methods))
    results=[]
    groups=source_groups(names,source_selector)
    stats=[]
    prediction_cache=out/'prediction_cache';prediction_cache.mkdir(parents=True,exist_ok=True)
    for group,indices in groups.items():
        Lg,Mg=select_sources(L,M,indices)
        if not Mg.any():
            tracker.event('DATA',f'{dataset_id} / {group}: all weak labels missing; input baselines remain valid')
        conf={'factors':list(factors),'reps':reps,'seed':eac_seed,'weight':label_weight,
              'eta':eta,'threshold':threshold,'knn':knn_neighbors,'backbone':backbone_weight}
        key=_graph_key(signature,group,int(k),conf)
        folder=out/'graphs';folder.mkdir(parents=True,exist_ok=True)
        feature_folder=out/'spectral_cache';feature_folder.mkdir(parents=True,exist_ok=True)
        g=None;classic=None
        if EAC_METHODS.intersection(methods):
            if EAC_METHODS.intersection(methods)-{'eac_classic'}:
                path=folder/f'{key}_multiview.npz'
                if path.exists():
                    g=sparse.load_npz(path)
                    tracker.event('EAC',f'Multiview cache hit: {dataset_id} {group} K={k}')
                else:
                    tracker.event('EAC',f'Building multiview graph {dataset_id} {group} K={k}: {3*len(factors)*reps} high-K partitions')
                    g,_=evidence_graph(X,Lg,Mg,k,eac_seed,factors,reps,label_weight,
                        progress=lambda view,done,total,elapsed,eta:tracker.event('EAC',
                        f'{dataset_id}/{group} {view}: partition {done}/{total} elapsed={elapsed:.1f}s ETA={eta:.1f}s'))
                    sparse.save_npz(path,g)
                stats.append({'dataset':dataset_id,'source_set':group,'k':k,'graph':'multiview',
                              'edges':g.nnz//2,'coverage':float(np.mean(np.diff(g.indptr)>0)),
                              'mean_degree':g.nnz/len(X),'fraction_strong':float(np.mean(g.data>=threshold)) if g.nnz else 0})
            if 'eac_classic' in methods or 'eac_average_linkage' in methods:
                classic_key=_graph_key(signature,'classic_input',int(k),{'factors':list(factors),'reps':reps,'seed':eac_seed})
                path=folder/f'{classic_key}_classic.npz'
                if path.exists(): classic=sparse.load_npz(path)
                else:
                    tracker.event('EAC',f'Building classical input-only EAC graph K={k}')
                    classic=consensus(X,k,eac_seed,factors,reps,
                        progress=lambda done,total,elapsed,eta:tracker.event('EAC',
                        f'{dataset_id}/input-only: partition {done}/{total} elapsed={elapsed:.1f}s ETA={eta:.1f}s'))
                    sparse.save_npz(path,classic)
                stats.append({'dataset':dataset_id,'source_set':group,'k':k,'graph':'classical_input',
                              'edges':classic.nnz//2,'coverage':float(np.mean(np.diff(classic.indptr)>0)),
                              'mean_degree':classic.nnz/len(X),'fraction_strong':float(np.mean(classic.data>=threshold)) if classic.nnz else 0})
        representations,logs=method_embeddings(X,Lg,Mg,g,label_weight,eta,threshold)
        # Deferred costly spectral decompositions until at least one seed requires them.
        f_cache={}
        density_cache=None
        def get_features(method):
            if method in f_cache:return f_cache[method]
            kcache=_graph_key(signature,'inputs' if method in ('spectral_input','eac_classic') else group,int(k),{'spectral_method':method,'knn':knn_neighbors,
                     'backbone':backbone_weight,'graph':key if method=='eac_multiview' else 'inputs'})
            path=feature_folder/f'{kcache}.npy'
            if path.exists(): feature=np.load(path)
            else:
                tracker.event('SPECTRAL',f'{dataset_id} group={group} K={k} method={method}: computing sparse eigenspace')
                if method=='spectral_input':feature=spectral_features(knn_graph(X,knn_neighbors),k)
                elif method=='spectral_fusion':feature=spectral_features(knn_graph(representations['fusion'],knn_neighbors),k)
                elif method=='eac_classic':feature=spectral_eac_features(classic,X,k,knn_neighbors,backbone_weight)
                elif method=='eac_multiview':feature=spectral_eac_features(g,representations['fusion'],k,knn_neighbors,backbone_weight)
                else:raise ValueError(method)
                np.save(path,feature)
            f_cache[method]=feature
            return feature
        for seed in seed_set:
            for method in methods:
                identity=(dataset_id,int(generation_seed),group,int(k),int(seed),method)
                if existing is not None and identity in existing:continue
                tic=time.monotonic()
                pred_path=None
                if method in INPUT_ONLY_METHODS:
                    pred_key=_graph_key(signature,'input_only',int(k),
                        {'method':method,'seed':int(seed),'knn':knn_neighbors,'factors':list(factors),
                         'reps':reps,'eac_seed':eac_seed,'backbone':backbone_weight,
                         'dec_latent_dim':dec_latent_dim,'dec_epochs':dec_epochs,
                         'dec_pretrain_epochs':dec_pretrain_epochs,'dec_batch_size':dec_batch_size,
                         'dec_device':dec_device})
                    pred_path=prediction_cache/f'{pred_key}.npz'
                try:
                    pred_cache_hit=bool(pred_path is not None and pred_path.exists())
                    if pred_cache_hit:
                        with np.load(pred_path) as cached:pred=cached['pred'].copy()
                    elif method in representations:pred=kmeans_predict(representations[method],k,seed)
                    elif method=='spherical_input':pred=spherical_kmeans(X,k,seed)
                    elif method=='spherical_fusion':pred=spherical_kmeans(representations['fusion'],k,seed)
                    elif method in SPECTRAL_METHODS:pred=kmeans_predict(get_features(method),k,seed)
                    elif method=='eac_average_linkage':pred=eac_average_linkage(classic,k)
                    elif method=='hdbscan_fusion':
                        if density_cache is None:
                            tracker.event('DENSITY',f'{dataset_id} K={k} group={group}: PCA({density_pca_dim}) + HDBSCAN')
                            density_cache=density_predict(representations['fusion'],k,seed,pca_dim=density_pca_dim)
                        pred=density_cache.copy()
                    elif method in DEEP_METHODS:
                        from .deep import dec_predict
                        V=X if method=='dec_input' else representations['fusion']
                        def on_epoch(phase,e,tot,loss):
                            if e==tot or e==1 or e%max(1,tot//5)==0:
                                tracker.event('DEC',f'{dataset_id}/{group} K={k} seed={seed} {phase} {e}/{tot} loss={loss:.6f}')
                        pred=dec_predict(V,k,seed=seed,latent_dim=dec_latent_dim,
                                   pretrain_epochs=dec_pretrain_epochs,dec_epochs=dec_epochs,
                                   batch_size=dec_batch_size,device=dec_device,callback=on_epoch)
                    else:raise ValueError(method)
                    if pred_path is not None and not pred_path.exists():
                        np.savez_compressed(pred_path,pred=np.asarray(pred,dtype=np.int32))
                    n_noise=int((pred==-1).sum())
                    n_clusters=int(len(set(pred)-{-1}))
                    if compute_silhouette and n_clusters>=2 and n_noise<len(X)-2:
                        keep=np.flatnonzero(pred!=-1)
                        # All methods are scored against the exact same fused geometry.
                        sil=np.nan
                        if len(keep)>n_clusters:
                            try:
                                sil=float(silhouette_score(representations['fusion'][keep],pred[keep],
                                     sample_size=min(350,len(keep)),metric='euclidean',random_state=42))
                            except ValueError:
                                pass
                    else:sil=np.nan
                    record={'data_kind':'synthetic' if tier!='real' else 'real',
                            'scenario':dataset_id,'tier':tier,'generation_seed':generation_seed,
                            'source_set':group,'k':int(k),'k_factor':float(k_factor),'seed':int(seed),
                            'method':method,'n':len(X),'source_count':len(indices),
                            'ari':float(adjusted_rand_score(y,pred)) if y is not None else np.nan,
                            'nmi':float(normalized_mutual_info_score(y,pred)) if y is not None else np.nan,
                            'ami':float(adjusted_mutual_info_score(y,pred)) if y is not None else np.nan,
                            'silhouette_fusion':sil,'n_clusters_found':n_clusters,
                            'n_noise':n_noise,'noise_rate':n_noise/len(X),
                            'duration_s':float(time.monotonic()-tic),'prediction_cache_hit':pred_cache_hit,
                            'graph_edges':int(g.nnz//2) if g is not None else 0,
                            'corrected_label_fraction':np.nan,'mean_eta':np.nan,'mean_support':np.nan}
                    record.update(logs.get(method,{}))
                    results.append(record)
                    tracker.event('RESULT',f'{dataset_id} / {group} K={k} seed={seed} {method}: ARI={record["ari"]:.4f} NMI={record["nmi"]:.4f} clusters={n_clusters} noise={n_noise/len(X):.1%} duration={record["duration_s"]:.1f}s')
                    if callback:callback(record,pred,np.asarray(ids),group,method)
                    if existing is not None:existing.add(identity)
                except Exception as ex:
                    tracker.event('FAILED',f'{dataset_id} / {group} / {method}: {type(ex).__name__}: {ex}')
                    raise
    return results,stats

def experiment_synthetic(root,out,scenarios,seeds,generation_seeds,profile='quick',sources='all',k_factors=(1,),
                         sample=0,methods=CORE_METHODS,label_weight=.35,eac_eta=.25,eac_threshold=.75,
                         knn_neighbors=15,backbone_weight=.03,density_pca_dim=32,
                         dec_latent_dim=32,dec_pretrain_epochs=8,dec_epochs=15,dec_batch_size=256,dec_device="cpu",
                         progress='rich',verbose=True,resume=True,dry_run=False):
    from .synthetic import generate
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    tracker=Tracker(out,progress,verbose)
    runs_path=out/'runs.csv'
    existing_rows=pd.read_csv(runs_path).to_dict('records') if resume and runs_path.exists() else []
    if not resume and runs_path.exists():runs_path.unlink()
    existing={(str(r['scenario']),int(r['generation_seed']),str(r['source_set']),int(r['k']),int(r['seed']),str(r['method'])) for r in existing_rows}
    eac_kwargs={'factors':(3,6) if profile=='quick' else (4,8),'reps':1 if profile=='quick' else 2}
    groups=source_groups(['gemini','claude','llama'],sources)
    total=len(scenarios)*len(generation_seeds)*len(groups)*len(k_factors)*len(seeds)*len(methods)
    check_protocol(out,{'kind':'synthetic','profile':profile,'scenarios':scenarios,'generations':list(generation_seeds),
                        'seeds':list(seeds),'sources':sources,'k_factors':list(k_factors),'sample':sample,
                        'methods':list(methods),'label_weight':label_weight,'eta':eac_eta,'threshold':eac_threshold,
                        'eac_factors':list(eac_kwargs['factors']),'eac_reps':eac_kwargs['reps'],
                        'knn_neighbors':knn_neighbors,'backbone_weight':backbone_weight,'density_pca_dim':density_pca_dim,
                        'dec_latent_dim':dec_latent_dim,'dec_pretrain_epochs':dec_pretrain_epochs,'dec_epochs':dec_epochs,
                        'dec_batch_size':dec_batch_size,'dec_device':dec_device},resume)
    tracker.event('PLAN',f'profile={profile} scenario_families={len(scenarios)} generation_seeds={generation_seeds} K_factors={k_factors} sources={list(groups)} seeds={seeds} planned method-runs={total}')
    if dry_run:return
    all_graphs=[]
    with threadpool_limits(1), tracker.progress(len(scenarios)*len(generation_seeds)*len(groups)*len(k_factors), 'Synthetic jobs') as tick:
        for gen_seed in generation_seeds:
            rootgen=Path(root)/f'generation_{gen_seed}'
            manifest=generate(rootgen,scenarios,seed=gen_seed,dim=96)
            for item in manifest:
                name=item['name'];npz=rootgen/'data'/name/'synthetic_embeddings.npz'
                with np.load(npz) as d:
                    X,L,M=d['X'].copy(),d['labels'].copy(),d['label_mask'].copy()
                    ids=d['ids'].copy()
                    # y is obtained separately below, outside graph/model construction.
                if sample and sample<len(X):
                    sample_idx=np.sort(np.random.default_rng(gen_seed+2).choice(len(X),size=sample,replace=False))
                    X,L,M,ids=X[sample_idx],L[sample_idx],M[sample_idx],ids[sample_idx]
                else:sample_idx=None
                with np.load(npz) as d:
                    yy=d['y'].copy()
                y=yy[sample_idx] if sample_idx is not None else yy
                for factor in k_factors:
                    k=max(2,min(len(X)-2,int(round(item['k']*factor))))
                    sig=f'{npz.resolve()}:{npz.stat().st_size}:{npz.stat().st_mtime_ns}:sample{sample}:ss{gen_seed}'
                    def on_result(r,pred,identifiers,grp,method):
                        pred_dir=out/'predictions';pred_dir.mkdir(exist_ok=True)
                        token=hashlib.sha256(json.dumps([r['scenario'],r['generation_seed'],grp,r['k'],r['seed'],method]).encode()).hexdigest()[:20]
                        pred_path=pred_dir/f'{token}.npz'
                        np.savez_compressed(pred_path,pred=np.asarray(pred,dtype=np.int32))
                        r['prediction_file']=str(pred_path.relative_to(out))
                        existing_rows.append(r)
                        append_run(r,runs_path)
                    added,gstats=run_one_dataset(X,L,M,y,ids,['gemini','claude','llama'],k,seeds,sources,out,tracker,name,sig,
                                              factors=eac_kwargs['factors'],reps=eac_kwargs['reps'],label_weight=label_weight,
                                              eta=eac_eta,threshold=eac_threshold,k_factor=factor,tier=item['tier'],
                                              generation_seed=gen_seed,methods=methods,existing=existing,callback=on_result,
                                              knn_neighbors=knn_neighbors,backbone_weight=backbone_weight,
                                              density_pca_dim=density_pca_dim,dec_latent_dim=dec_latent_dim,
                                              dec_pretrain_epochs=dec_pretrain_epochs,dec_epochs=dec_epochs,
                                              dec_batch_size=dec_batch_size,dec_device=dec_device)
                    all_graphs.extend(gstats)
                    for _ in groups:tick()
                tracker.event('DONE',f'{name} gen={gen_seed} ({len(X)} rows); completed={len(existing_rows)}/{total}')
    if existing_rows:
        if all_graphs:pd.DataFrame(all_graphs).to_csv(out/'graph_stats.csv',index=False)
        summary,comparisons=write_report(runs_path,out/'reports',title='Synthetic multi-LLM EAC clustering')
        write_stability(runs_path,out/'reports'/'seed_stability.csv')
        tracker.table(summary.sort_values('ari_mean',ascending=False).head(12).round(4).to_dict('records'))
    tracker.event('DONE',f'Synthetic benchmark complete: {runs_path}')


def experiment_realworld(input_path,mapping_path,out,encoder,precomputed=None,profile='quick',sample=0,sample_seed=42,k_values=(20,),
                         seeds=(42,),sources='all',methods=METHODS,batch_size=32,device=None,max_chars=4096,
                         cache_dir=None,label_weight=.35,eac_eta=.25,eac_threshold=.75,
                         knn_neighbors=15,backbone_weight=.03,density_pca_dim=32,
                         dec_latent_dim=32,dec_pretrain_epochs=8,dec_epochs=15,dec_batch_size=256,dec_device="cpu",
                         progress='rich',verbose=True,resume=True,dry_run=False,dataset_name=None):
    from .data import read_csv,extract_texts,embed_texts
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    tracker=Tracker(out,progress,verbose)
    mapping=json.loads(Path(mapping_path).read_text())
    dataset_name = dataset_name or Path(input_path).stem
    groups=source_groups(list(mapping['label_columns']),sources)
    check_protocol(out,{'kind':'real','profile':profile,'input':str(Path(input_path).resolve()),
                        'input_size':Path(input_path).stat().st_size,'mapping':mapping,'dataset_name':dataset_name,
                        'encoder':encoder,'precomputed':str(Path(precomputed).resolve()) if precomputed else None,'sample':sample,'sample_seed':sample_seed,'k_values':list(k_values),
                        'seeds':list(seeds),'sources':sources,'methods':list(methods),
                        'batch_size':batch_size,'device':device,'max_chars':max_chars,
                        'label_weight':label_weight,'eta':eac_eta,'threshold':eac_threshold,
                        'knn_neighbors':knn_neighbors,'backbone_weight':backbone_weight,'density_pca_dim':density_pca_dim,
                        'dec_latent_dim':dec_latent_dim,'dec_pretrain_epochs':dec_pretrain_epochs,'dec_epochs':dec_epochs,
                        'dec_batch_size':dec_batch_size,'dec_device':dec_device},resume)
    tracker.event('PLAN',f'Real-world {profile}: K={k_values} seeds={seeds} sources={list(groups)} methods={methods} encoder={encoder}')
    if dry_run:return
    df=read_csv(input_path,mapping)
    tracker.event('DATA',f'CSV loaded: {len(df):,} rows, {len(df.columns)} columns')
    ds=extract_texts(df,mapping,max_chars,sample=sample,sample_seed=sample_seed)
    tracker.event('DATA',f'Sample: {len(ds.ids):,} rows; sources={ds.source_names}, gold categories only evaluated after predictions')
    last_pct=[-1]
    def emb_progress(label,completed,total):
        pct=int(completed/max(total,1)*20)
        if pct>last_pct[0]:
            last_pct[0]=pct
            tracker.event('ENCODER',f'{label} {completed:,}/{total:,} ({completed/max(total,1):.1%})')
    tracker.event('ENCODER',f'Preparing {encoder} embeddings (existing cache will be used when available)')
    if precomputed:
        cache=Path(precomputed)
        with np.load(cache) as vectors:
            if 'ids' not in vectors or 'source_names' not in vectors:
                raise ValueError('Precomputed NPZ must include ids and source_names to guarantee alignment')
            if not np.array_equal(vectors['ids'].astype(str),ds.ids.astype(str)):
                raise ValueError('Precomputed embeddings do not match selected CSV rows/IDs and order')
            if list(vectors['source_names'].astype(str))!=ds.source_names:
                raise ValueError('Precomputed label sources do not match CSV column mapping')
            X,L,M=vectors['X'].copy(),vectors['labels'].copy(),vectors['label_mask'].copy()
        hit=True
    else:
        X,L,M,cache,hit=embed_texts(ds,encoder,cache_dir or out/'embedding_cache',batch_size,device,max_chars,emb_progress)
    tracker.event('ENCODER',f'{"Loaded" if hit else "Saved"} embedding cache {cache}; shape X={X.shape}, labels={L.shape}')
    rows_path=out/'runs.csv'
    existing_rows=pd.read_csv(rows_path).to_dict('records') if resume and rows_path.exists() else []
    if not resume and rows_path.exists():rows_path.unlink()
    existing={(str(r['scenario']),int(r['generation_seed']),str(r['source_set']),int(r['k']),int(r['seed']),str(r['method'])) for r in existing_rows}
    eac_kwargs={'factors':(3,6) if profile=='quick' else (4,8),'reps':1 if profile=='quick' else 2}
    graph_stats=[]
    samples=pd.DataFrame({'id':ds.ids,'source_row':ds.source_rows})
    samples.to_csv(out/'sample_manifest.csv',index=False)
    with threadpool_limits(1),tracker.progress(len(k_values)*len(groups),'Real-world jobs') as tick:
        for k in k_values:
            if k>=len(X):raise ValueError('K must be smaller than sample size')
            signature=f'{cache.name}:N{len(X)}:sample{sample_seed}'
            def on_result(r,pred,identifiers,grp,method):
                # Predictions contain no target values; their paths permit seed-stability audits.
                dst=out/'predictions';dst.mkdir(parents=True,exist_ok=True)
                pred_path=dst/f'{grp}_k{k}_{method}_s{r["seed"]}.npz'
                np.savez_compressed(pred_path,pred=np.asarray(pred,dtype=np.int32),ids=identifiers)
                r['prediction_file']=str(pred_path.relative_to(out))
                existing_rows.append(r);append_run(r,rows_path)
            added,gstats=run_one_dataset(X,L,M,ds.target,ds.ids,ds.source_names,k,seeds,sources,out,tracker,dataset_name,signature,
                                        factors=eac_kwargs['factors'],reps=eac_kwargs['reps'],label_weight=label_weight,
                                        eta=eac_eta,threshold=eac_threshold,tier='real',methods=methods,
                                        existing=existing,callback=on_result,
                                        knn_neighbors=knn_neighbors,backbone_weight=backbone_weight,
                                        density_pca_dim=density_pca_dim,dec_latent_dim=dec_latent_dim,
                                        dec_pretrain_epochs=dec_pretrain_epochs,dec_epochs=dec_epochs,
                                        dec_batch_size=dec_batch_size,dec_device=dec_device)
            graph_stats.extend(gstats)
            for _ in groups:tick()
    if existing_rows:
        if graph_stats:pd.DataFrame(graph_stats).to_csv(out/'graph_stats.csv',index=False)
        summary,comparisons=write_report(rows_path,out/'reports',title=f'{dataset_name}: semantic weak-label clustering')
        write_stability(rows_path,out/'reports'/'seed_stability.csv')
        tracker.table(summary.sort_values('ari_mean',ascending=False).head(12).round(4).to_dict('records'))
    tracker.event('DONE',f'Real-world benchmark complete: {rows_path}')
