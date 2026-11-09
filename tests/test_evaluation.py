import json
import zipfile
from pathlib import Path
from argparse import Namespace

import numpy as np
import pandas as pd
import pytest

from swiftc.retrieval.core import evaluate_clustering, evaluate_topk
from swiftc.retrieval.cli import main, read_reference, require_aligned


def test_perfect_recommendation():
    ids=np.array(['a','b','c','d'])
    truth=np.array(['A','A','B','B'])
    clusters=np.array([0,0,1,1])
    result=evaluate_clustering(ids,truth,clusters,query_details=True)
    for m in ('precision','recall','f1','pair_f1','macro_category_f1','candidate_coverage'):
        assert result.summary[m] == pytest.approx(1.0),m
    assert result.summary['mean_candidates']==1
    assert len(result.query_table)==4


def test_singletons_count_as_zero():
    ids=np.array(['a','b','c','d'])
    y=np.array(['A','A','B','B'])
    pred=np.array([0,0,1,2])
    result=evaluate_clustering(ids,y,pred)
    assert result.summary['precision']==pytest.approx(.5)
    assert result.summary['recall']==pytest.approx(.5)
    assert result.summary['f1']==pytest.approx(.5)
    assert result.summary['candidate_coverage']==pytest.approx(.5)
    assert result.summary['hit_coverage']==pytest.approx(.5)
    assert result.summary['pair_precision']==pytest.approx(1)
    assert result.summary['pair_recall']==pytest.approx(.5)


def test_noise_is_not_cluster():
    ids=np.array(['a','b','c','d'])
    truth=np.array(['A','A','B','B'])
    result=evaluate_clustering(ids,truth,np.array([-1,-1,0,0]))
    assert result.summary['n_noise']==2
    assert result.summary['n_clusters_found']==3
    assert result.summary['f1']==pytest.approx(.5)
    assert result.summary['pair_precision']==pytest.approx(1)


def test_per_query_f1_not_aggregate_harmonic_mean():
    ids=np.array([str(i) for i in range(6)])
    y=np.array(['A','A','A','B','B','B'])
    pred=np.array([0,0,1,0,1,1])
    r=evaluate_clustering(ids,y,pred)
    q=evaluate_clustering(ids,y,pred,query_details=True).query_table
    assert r.summary['f1']==pytest.approx(float(q.cluster_f1.mean()))
    assert r.summary['random_precision']==pytest.approx(2/5)


def test_ranked_within_cluster():
    ids=np.array(['v1','v2','v3','v4'])
    y=np.array(['A','A','B','B'])
    clusters=np.array([0,0,0,0])
    vectors=np.array([[1,0], [.99,.05], [0,1], [.05,.99]],dtype=np.float32)
    s, q = evaluate_topk(ids,y,clusters,vectors,(1,2),query_details=True,block=2)
    assert s['precision_at_1']==pytest.approx(1.)
    assert s['recall_at_1']==pytest.approx(1.)
    assert s['map_at_1']==pytest.approx(1.)
    assert s['ndcg_at_1']==pytest.approx(1.)
    assert s['precision_at_2']==pytest.approx(.5)
    assert s['map_at_2']==pytest.approx(1.)
    assert len(q)==4


def test_ranked_singleton_zero():
    ids=np.array(['a','b','c','d'])
    truth=np.array(['A','A','B','B'])
    vec=np.eye(4,dtype=np.float32)
    metrics,_=evaluate_topk(ids,truth,np.array([0,1,2,3]),vec,(1,10))
    assert all(val==0 for val in metrics.values())


def test_ground_truth_not_used_to_rank():
    ids=np.array(['a','b','c','d'])
    pred=np.array([0,0,0,0])
    x=np.array([[1,0],[.9,.1],[0,1],[.1,.9]],dtype=np.float32)
    m1,_=evaluate_topk(ids,np.array(['A','A','B','B']),pred,x,(1,))
    m2,_=evaluate_topk(ids,np.array(['A','B','A','B']),pred,x,(1,))
    # Same neighbors, only evaluation changes; different values are expected.
    assert m1['precision_at_1']!=m2['precision_at_1']


def test_ids_reordered_correctly():
    ref=pd.DataFrame({'video_id':['a','b','c'],'category':['x','y','z']})
    assert require_aligned(ref,np.array(['c','a','b']),'video_id','category').tolist()==['z','x','y']
    with pytest.raises(ValueError,match='missing from reference'):
        require_aligned(ref,np.array(['a','unknown']),'video_id','category')
    with pytest.raises(ValueError,match='unique'):
        require_aligned(ref,np.array(['a','a']),'video_id','category')


def _fixture(tmp):
    df = pd.DataFrame({'video_id':[f'v{i}' for i in range(6)],'category':['A']*3+['B']*3,
                       'cat_gemini':['IGNORED']*6,'sug_gemini':['NOT_READ']*6})
    df.to_csv(tmp/'real.csv',index=False)
    out=tmp/'wleac';(out/'predictions').mkdir(parents=True)
    idx=df.video_id.to_numpy()
    x=np.array([[1,0],[1,.1],[1,.2],[0,1],[.1,1],[.2,1]],dtype=np.float32)
    cache=tmp/'embeddings_foo.npz'
    np.savez_compressed(cache,ids=idx,X=x)
    rows=[]
    for method in ('fusion','eac_adaptive'):
        for seed in (42,123):
            labels=np.array([0,0,0,1,1,1]) if method=='fusion' else np.array([0,0,1,1,1,0])
            name=f'predictions/all_k2_{method}_s{seed}.npz'
            np.savez_compressed(out/name,ids=idx[::-1],pred=labels[::-1])
            rows.append({'data_kind':'real','method':method,'source_set':'all',
                         'k':2,'seed':seed,'prediction_file':name})
    pd.DataFrame(rows).to_csv(out/'runs.csv',index=False)
    discrete=tmp/'discrete'; discrete.mkdir()
    pd.DataFrame({'video_id':idx,'exact_vote':[0,0,0,1,1,1],
                  'discrete_mustlink_s42':[0,0,0,1,1,1]}).to_csv(discrete/'predictions.csv',index=False)
    pd.DataFrame([{'method':'discrete_mustlink_kmeans','seed':42,'k_requested':2}]).to_csv(discrete/'metrics.csv',index=False)
    return df, out, discrete, cache


def test_cli_end_to_end_with_cache_and_discrete(tmp_path):
    df,out,discrete,cache=_fixture(tmp_path)
    output=tmp_path/'evaluation'
    main(['--results',str(out),'--input',str(tmp_path/'real.csv'),
          '--output',str(output),'--discrete',str(discrete),'--embeddings',str(cache),
          '--topk','1,2','--no-plots','--query-details'])
    runs=pd.read_csv(output/'per_run.csv')
    summary=pd.read_csv(output/'summary.csv')
    comparisons=pd.read_csv(output/'paired_vs_fusion.csv')
    assert len(runs)==6
    assert len(summary)==4
    assert len(comparisons)==3  # eac 2 seeds, discrete mustlink 1 seed
    assert (runs['map_at_1']>=0).all()
    assert (output/'per_category.csv').exists()
    assert len(pd.read_csv(output/'per_query.csv'))==36
    assert runs.loc[runs.method=='fusion','f1'].eq(1).all()
    report = (output/'REPORT.md').read_text()
    assert 'Cluster Precision' in report
    assert '### Top-1' in report and '### Top-2' in report
    assert 'Precision@1' in report and 'mAP@2' in report
    ranked = pd.read_csv(output/'ranking_summary.csv')
    assert sorted(ranked.top_k.unique().tolist()) == [1, 2]
    assert {'precision_at_k_mean','recall_at_k_mean','map_at_k_mean','ndcg_at_k_mean'} <= set(ranked)
    assert len(pd.read_csv(output/'ranking_per_run.csv')) == 12
    assert (output/'ranking_paired_vs_fusion.csv').exists()
    meta=json.loads((output/'metadata.json').read_text())
    assert meta['ranked_retrieval_enabled'] is True


def test_missing_prediction_skipped_or_strict(tmp_path):
    df,out,discrete,cache=_fixture(tmp_path)
    (out/'predictions/all_k2_fusion_s123.npz').unlink()
    main(['--results',str(out),'--input',str(tmp_path/'real.csv'),
          '--output',str(tmp_path/'skips'),'--no-plots'])
    assert 'Missing predictions' in (tmp_path/'skips'/'warnings.txt').read_text()
    with pytest.raises(FileNotFoundError):
        main(['--results',str(out),'--input',str(tmp_path/'real.csv'),
              '--output',str(tmp_path/'strict'),'--strict','--no-plots'])


def test_without_embeddings_clearly_marks_missing_topk_and_removes_stale_files(tmp_path):
    _, out, _, cache = _fixture(tmp_path)
    target = tmp_path / 'evaluation'
    main(['--results', str(out), '--input', str(tmp_path/'real.csv'),
          '--output', str(target), '--embeddings', str(cache), '--topk', '1,2', '--no-plots'])
    assert (target/'ranking_summary.csv').exists()
    main(['--results', str(out), '--input', str(tmp_path/'real.csv'),
          '--output', str(target), '--no-plots'])
    assert not (target/'ranking_summary.csv').exists()
    assert 'NOT CALCULATED' in (target/'REPORT.md').read_text()
    assert all('_at_' not in c for c in pd.read_csv(target/'summary.csv').columns)
    assert json.loads((target/'metadata.json').read_text())['ranked_retrieval_enabled'] is False
