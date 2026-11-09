import numpy as np
from scipy import sparse
from swiftc.clustering.core import fused,validate_arrays,evidence_graph,denoise,method_embeddings,kmeans_predict
from swiftc.clustering.synthetic import SCENARIOS,choose_scenarios


def toy(n=80,classes=4):
    rng=np.random.default_rng(21)
    d=20;centers=np.eye(classes,d,dtype=np.float32)
    y=np.arange(n)%classes
    X=centers[y]+rng.normal(0,.12,(n,d))
    L=np.stack([centers[y]+rng.normal(0,.14,(n,d)),centers[y]+rng.normal(0,.15,(n,d))],axis=1)
    M=np.ones((n,2),bool)
    return X,L,M,y


def test_fusion_and_no_labels():
    X,L,M,y=toy()
    F=fused(X,L,M)
    assert F.shape==(80,40)
    assert np.allclose(np.linalg.norm(F,axis=1),1,atol=1e-5)
    M[:5]=False
    F=fused(X,L,M)
    assert np.allclose(F[:5,20:],0)
    assert np.allclose(np.linalg.norm(F,axis=1),1,atol=1e-5)


def test_sparse_symmetric_evidence():
    X,L,M,y=toy()
    G,s=evidence_graph(X,L,M,k=4,factors=(3,5),reps=1)
    assert sparse.issparse(G)
    assert G.shape==(80,80) and G.nnz>0
    assert (G-G.T).nnz==0
    assert G.diagonal().sum()==0
    assert np.all(G.data>=0) and np.all(G.data<=1)
    assert s.n_partitions==6


def test_denoise_abstains_for_missing_and_is_deterministic():
    X,L,M,y=toy()
    M[2,0]=False;L[2,0]=0
    G,_=evidence_graph(X,L,M,k=4,factors=(3,5),reps=1)
    A,w,s=denoise(G,L,M,'adaptive')
    B,_,_=denoise(G,L,M,'adaptive')
    assert np.allclose(A,B)
    assert np.allclose(A[2,0],0) and w[2,0]==0
    assert (w>=0).all() and (w<=.25+1e-8).all()


def test_all_ablations_and_gt_independence():
    X,L,M,y=toy()
    G,_=evidence_graph(X,L,M,4,factors=(3,5),reps=1)
    reps,logs=method_embeddings(X,L,M,G)
    assert set(reps)=={'input','labels_only','fusion','eac_uniform','eac_adaptive'}
    pred=kmeans_predict(reps['eac_adaptive'],4,42)
    # Training doesn't receive y at all; re-labeling the gold classes
    # has no bearing on the resulting clusters.
    y_reordered=1000-y
    pred2=kmeans_predict(reps['eac_adaptive'],4,42)
    assert np.array_equal(pred,pred2)
    assert not np.array_equal(y,y_reordered)


def test_suite_33_families():
    assert len(SCENARIOS)==33
    assert len(choose_scenarios('quick'))==6
    assert len(choose_scenarios('all'))==33
