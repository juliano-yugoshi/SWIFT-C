"""Ground-truth-free clustering. All model selection decisions use X and L only."""
from __future__ import annotations
from dataclasses import dataclass
import time
import numpy as np
from scipy import sparse
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize


def normalized(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return normalize(x).astype(np.float32)


def validate_arrays(X, L, mask):
    X = np.asarray(X, dtype=np.float32)
    L = np.asarray(L, dtype=np.float32)
    mask = np.asarray(mask, dtype=bool)
    if X.ndim != 2 or L.ndim != 3 or mask.shape != L.shape[:2] or X.shape[0] != L.shape[0]:
        raise ValueError(f"Incompatible input shapes: X={X.shape} L={L.shape} mask={mask.shape}")
    if X.shape[1] != L.shape[2]:
        raise ValueError("Instance and label embeddings must share dimensionality")
    if not np.isfinite(X).all() or not np.isfinite(L).all():
        raise ValueError("Non-finite values in embeddings")
    return normalized(X), normalized(L.reshape(-1, L.shape[-1])).reshape(L.shape), mask


def select_sources(L, mask, source_indices):
    indices = [int(i) for i in source_indices]
    if not indices or any(i < 0 or i >= L.shape[1] for i in indices):
        raise ValueError("At least one valid weak-label source is required")
    return L[:,indices,:],mask[:,indices]


def pool_labels(L, mask):
    n, m, d=L.shape
    weighted = np.einsum('nmd,nm->nd', L, mask.astype(np.float32))
    count = mask.sum(axis=1)
    return normalized(weighted/np.maximum(count[:,None],1)),count>0


def fused(X, L, mask, label_weight=0.35):
    """65/35 = squared Euclidean contribution of two independently unit-normalized views."""
    if not 0 <= label_weight <= 1: raise ValueError('label_weight must be in [0,1]')
    X,L,mask=validate_arrays(X,L,mask)
    P,present=pool_labels(L,mask)
    alpha=np.where(present,label_weight,0).astype(np.float32)
    F=np.concatenate([np.sqrt(1-alpha[:,None])*X,np.sqrt(alpha[:,None])*P],axis=1)
    return normalized(F)


def kmeans_predict(V, k, seed, n_init=10, max_iter=175):
    if not 2 <= k <= len(V): raise ValueError(f"K={k} invalid for N={len(V)}")
    return KMeans(n_clusters=int(k), n_init=n_init, max_iter=max_iter, random_state=int(seed)).fit_predict(V)


def consensus(Z,k,seed,factors=(4,8),reps=2,max_iter=70,progress=None):
    """Sparse co-association obtained by accumulating high-K partitions."""
    n=len(Z)
    rng=np.random.default_rng(seed)
    accumulator=sparse.csr_matrix((n,n),dtype=np.float32)
    run_count=0
    start=time.monotonic()
    total=len(factors)*reps
    for factor in factors:
        micro_k=min(n-1,max(k+2,int(round(factor*k))))
        for _ in range(reps):
            p=kmeans_predict(Z,micro_k,int(rng.integers(1,2**31)),n_init=1,max_iter=max_iter)
            indicators=sparse.csr_matrix((np.ones(n,dtype=np.float32),(np.arange(n),p)),shape=(n,micro_k))
            accumulator+=indicators@indicators.T
            run_count+=1
            if progress:
                elapsed=max(time.monotonic()-start,1e-6)
                progress(run_count,total,elapsed,elapsed*(total-run_count)/run_count)
    accumulator=accumulator*(1.0/run_count)
    accumulator.setdiag(0)
    accumulator.eliminate_zeros()
    return accumulator.tocsr()


@dataclass
class EvidenceStats:
    n_edges: int
    mean_degree: float
    coverage: float
    strong_edges: int
    n_partitions: int


def evidence_graph(X,L,mask,k,seed=42,factors=(4,8),reps=2,label_weight=.35,progress=None):
    """Independent input, fused and pooled-label views; never uses target labels."""
    X,L,mask=validate_arrays(X,L,mask)
    F=fused(X,L,mask,label_weight)
    pooled,present=pool_labels(L,mask)
    def report(view):
        return (lambda done,total,elapsed,eta: progress(view,done,total,elapsed,eta)) if progress else None
    Cf=consensus(F,k,seed,factors,reps,progress=report('fusion'))
    Cx=consensus(X,k,seed+153,factors,reps,progress=report('input'))
    if present.sum()>=2 and np.linalg.norm(pooled[present].std(axis=0))>1e-7:
        # Missing-label rows do not provide semantic evidence; the label-view
        # consensus is computed only on rows with at least one observed label.
        ix=np.flatnonzero(present)
        if len(ix)>k+2:
            Cp=consensus(pooled[ix],min(k,len(ix)-3),seed+307,factors,reps,progress=report('labels'))
            coo=Cp.tocoo()
            Cl=sparse.csr_matrix((coo.data,(ix[coo.row],ix[coo.col])),shape=Cf.shape)
        else:
            Cl=sparse.csr_matrix(Cf.shape,dtype=np.float32)
    else:
        Cl=sparse.csr_matrix(Cf.shape,dtype=np.float32)
    support=((Cf>=.25).astype(np.int8)+(Cx>=.25).astype(np.int8)+(Cl>=.25).astype(np.int8)).tocsr()
    C=(.5*Cf+.25*Cx+.25*Cl).tocsr()
    retain=((support>=2).astype(np.float32).maximum((C>=.55).astype(np.float32))).tocsr()
    G=C.multiply(retain).tocsr()
    G.setdiag(0);G.eliminate_zeros()
    G=G.maximum(G.T).tocsr()
    # Store only positive evidence. Absence of an edge does not mean cannot-link.
    strong=int((G.data>=.75).sum()//2)
    return G, EvidenceStats(n_edges=G.nnz//2,mean_degree=G.nnz/len(X),coverage=float(np.mean(np.diff(G.indptr)>0)),strong_edges=strong,n_partitions=3*len(factors)*reps)


def denoise(G,L,mask, mode='adaptive',eta=.25, threshold=.75):
    """Smooth only within strong EAC neighborhoods, for the same label source.

    adaptive eta_i,m = eta * clip((0.95 - cos(label_i,m, neighborhood_mean))/0.30,0,1).
    All missing labels remain missing. The inputs X are never modified.
    """
    if mode not in ('uniform','adaptive'):
        raise ValueError('mode must be uniform or adaptive')
    if not 0<=eta<=1: raise ValueError('eta must be in [0,1]')
    L=np.asarray(L,dtype=np.float32);mask=np.asarray(mask,dtype=bool)
    H=G.multiply((G>=threshold).astype(np.float32)).tocsr()
    H.setdiag(0);H.eliminate_zeros()
    updated=L.copy();weights=np.zeros(mask.shape,dtype=np.float32)
    support=np.zeros(mask.shape,dtype=np.int32)
    for m in range(L.shape[1]):
        S=H.multiply(mask[:,m][None,:]).tocsr()
        den=np.asarray(S.sum(1)).ravel()
        support[:,m]=np.diff(S.indptr)
        neighbors=(S@L[:,m,:])/np.maximum(den[:,None],1e-10)
        neighbors=normalized(neighbors)
        agreement=np.einsum('nd,nd->n',L[:,m,:],neighbors)
        use=(den>1e-9)&mask[:,m]
        w=np.full(len(L),eta,dtype=np.float32) if mode=='uniform' else eta*np.clip((.95-agreement)/.30,0,1)
        weights[use,m]=w[use]
        if np.any(use):
            updated[use,m]=normalized((1-w[use,None])*L[use,m]+w[use,None]*neighbors[use])
    return updated,weights,support


def method_embeddings(X,L,mask,G=None,label_weight=.35,eta=.25,threshold=.75):
    """Return all ablation representations; no targets are accessible here."""
    X,L,mask=validate_arrays(X,L,mask)
    original=fused(X,L,mask,label_weight)
    P,_=pool_labels(L,mask)
    rep={'input':X,'labels_only':P,'fusion':original}
    logs={}
    if G is not None:
        for key,mode in [('eac_uniform','uniform'),('eac_adaptive','adaptive')]:
            Lu,w,s=denoise(G,L,mask,mode=mode,eta=eta,threshold=threshold)
            rep[key]=fused(X,Lu,mask,label_weight)
            logs[key]={'corrected_label_fraction':float(np.mean(w[mask]>1e-7)) if mask.any() else 0.,
                       'mean_eta':float(w[mask].mean()) if mask.any() else 0.,
                       'mean_support':float(s[mask].mean()) if mask.any() else 0.}
    return rep,logs
