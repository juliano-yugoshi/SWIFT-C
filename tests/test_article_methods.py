import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics import adjusted_rand_score
from swiftc.clustering.baselines import spherical_kmeans,knn_graph,spectral_features,spectral_eac_features,density_predict,eac_average_linkage
from swiftc.clustering.deep import dec_predict
from swiftc.clustering.core import evidence_graph,normalized
from swiftc.clustering.runner import Tracker,run_one_dataset,METHODS
from swiftc.clustering.report import write_stability,write_report


def toy(n=160,k=4,d=32):
    rng=np.random.default_rng(112)
    y=np.arange(n)%k
    centers=np.eye(k,d,dtype=np.float32)
    X=normalized(centers[y]+rng.normal(0,.09,(n,d)))
    L=normalized((centers[y,None,:]+rng.normal(0,.11,(n,2,d))).reshape(-1,d)).reshape(n,2,d)
    return X,L,np.ones((n,2),bool),y


def test_independent_baselines():
    X,L,mask,y=toy()
    for seed in [7,42]:
        assert adjusted_rand_score(y,spherical_kmeans(X,4,seed))>.99
    A=knn_graph(X,12)
    assert sparse.issparse(A) and (A-A.T).nnz==0 and A.nnz<160*40
    z=spectral_features(A,4)
    assert z.shape==(160,4)
    G,_=evidence_graph(X,L,mask,4,factors=(3,5),reps=1)
    assert spectral_eac_features(G,X,4).shape==(160,4)
    # Exact average-linkage EAC remains a separate small-N reference.
    yp=eac_average_linkage(G,4)
    assert yp.shape==(160,)
    p=density_predict(X,4,pca_dim=16)
    assert p.shape==(160,) and np.isin(p,[-1,*range(200)]).all()


def test_dec_without_targets_and_with_kl():
    X,L,mask,y=toy(96,4,20)
    losses=[]
    p=dec_predict(X,4,seed=42,latent_dim=6,pretrain_epochs=2,dec_epochs=3,
                  batch_size=32,callback=lambda *args:losses.append(args))
    assert len(p)==96 and len(set(p))>1
    assert {'pretrain','dec_kl'}.issubset({item[0] for item in losses})


def test_full_suite_and_prediction_stability(tmp_path):
    X,L,M,y=toy(96,4,20)
    tracker=Tracker(tmp_path,progress='plain',verbose=False)
    results=[]
    def callback(r,p,ids,g,method):
        pth=tmp_path/f'{g}_{method}_{r["seed"]}.npz'
        np.savez_compressed(pth,pred=p)
        r['prediction_file']=str(pth)
        results.append(r.copy())
    produced,_=run_one_dataset(X,L,M,y,np.arange(len(y)),['a','b'],4,[41,42],
           'all',tmp_path,tracker,'toy','fresh',factors=(3,5),reps=1,
           methods=METHODS,dec_pretrain_epochs=1,dec_epochs=2,
           callback=callback)
    assert len(produced)==2*len(METHODS)
    assert set(r['method'] for r in results)==set(METHODS)
    df=pd.DataFrame(results)
    assert set(['ari','nmi','ami','silhouette_fusion','noise_rate','prediction_file']).issubset(df)
    csv_path=tmp_path/'runs.csv';df.to_csv(csv_path,index=False)
    stab=write_stability(csv_path,tmp_path/'seed_stability.csv')
    assert len(stab)==len(METHODS)
    assert all(stab.seed_stability_ari.between(-1,1))
    s,c=write_report(csv_path,tmp_path/'report')
    assert 'holm_p' in c.columns
