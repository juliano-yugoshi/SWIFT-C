"""Independent clustering baselines: no gold labels, no test-metric selection.

Spectral methods use a sparse cosine-neighborhood graph.  EAC graph
baselines have a weak geometry backbone to avoid unstable isolated components;
the backbone coefficient is explicit and fixed throughout the benchmark.
"""
from __future__ import annotations
import numpy as np
from scipy import sparse
import warnings
from sklearn.manifold import spectral_embedding
from sklearn.cluster import KMeans, HDBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
from .core import normalized, consensus


def spherical_kmeans(V, k: int, seed: int, n_init: int=10, max_iter: int=80):
    """Cosine-distance Lloyd iterations with unit-norm centroids and best restart."""
    V=normalized(V)
    n,d=V.shape
    rng=np.random.default_rng(seed)
    best=np.inf;best_labels=None
    for _ in range(n_init):
        cent=V[rng.choice(n,size=k,replace=False)].copy()
        old=None
        for step in range(max_iter):
            s=V@cent.T
            labels=s.argmax(axis=1)
            if old is not None and np.array_equal(labels,old): break
            old=labels
            means=np.zeros((k,d),dtype=np.float32)
            np.add.at(means,labels,V)
            empty=(np.linalg.norm(means,axis=1)<1e-12)
            if empty.any():means[empty]=V[rng.choice(n,size=empty.sum(),replace=False)]
            cent=normalized(means)
        objective=float((1-(V@cent.T).max(axis=1)).sum())
        if objective<best:
            best=objective;best_labels=labels.copy()
    return best_labels


def knn_graph(V,n_neighbors=15):
    """Sparse symmetric weighted cosine kNN graph; never makes a dense NxN matrix."""
    V=normalized(V)
    n=len(V)
    if n<3:raise ValueError('At least 3 samples are required')
    nn=NearestNeighbors(n_neighbors=min(n,max(2,n_neighbors+1)),metric='euclidean',n_jobs=1)
    nn.fit(V)
    ds,ix=nn.kneighbors(V)
    rows=np.repeat(np.arange(n),ix.shape[1]-1)
    cols=ix[:,1:].reshape(-1)
    # Unit-norm Euclidean distances encode cos similarities: cos=1-d²/2.
    values=np.exp(-np.square(ds[:,1:]).reshape(-1)/.25).astype(np.float32)
    A=sparse.csr_matrix((values,(rows,cols)),shape=(n,n))
    A=A.maximum(A.T)
    A.setdiag(0);A.eliminate_zeros()
    return A


def spectral_features(A,k,seed=42):
    """Row-normalized leading eigenvectors of the symmetric normalized affinity.

    The eigenproblem (not K-Means) is cached and shared across clustering seeds.
    """
    n=A.shape[0]
    if A.shape!=(n,n):raise ValueError('A must be square')
    if k>=n-1:raise ValueError('K must be smaller than N-1')
    A=A.tocsr().astype(np.float64)
    A=A.maximum(A.T)
    A.setdiag(1.0)
    # scikit-learn uses the normalized graph Laplacian; more stable than
    # directly extracting leading normalized adjacency eigenvectors.
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore',message='Graph is not fully connected*')
        vecs=spectral_embedding(A,n_components=int(k),eigen_solver='arpack',
                                random_state=int(seed),drop_first=False)
    return normalized(vecs).astype(np.float32)


def spectral_eac_features(G,V,k,knn=15,backbone_weight=.03):
    """Spectral features from EAC with an explicitly tiny kNN geometric backbone.

    This is not pure evidence accumulation; report the backbone in the article.
    """
    base=knn_graph(V,n_neighbors=knn)
    if G is None or G.nnz==0: return spectral_features(base,k)
    return spectral_features(G + backbone_weight*base,k)


def density_predict(V,k,seed=42,pca_dim=32,min_cluster_size=None,min_samples=5):
    """HDBSCAN with prespecified PCA; -1 denotes noise and K is not enforced."""
    V=normalized(V)
    n=len(V)
    components=min(pca_dim,V.shape[1],n-1)
    if components<V.shape[1]:
        data=PCA(n_components=components,svd_solver='randomized',random_state=42).fit_transform(V)
    else:data=V
    size=min_cluster_size if min_cluster_size is not None else max(8,int(n/(max(k,2)*5)))
    size=min(max(size,2),n-1)
    return HDBSCAN(min_cluster_size=size,min_samples=min(min_samples,size),metric='euclidean',n_jobs=1,copy=True).fit_predict(data)


def eac_average_linkage(G,k,max_n=2000):
    """Exact Fred/Jain-style evidence accumulation, average linkage on 1-coassociation.

    A dense NxN distance matrix is intrinsically expensive; this baseline is
    intentionally restricted to <=max_n rows and is not included in the
    large-scale paper suite. No geometric kNN backbone is added.
    """
    n=G.shape[0]
    if n>max_n:
        raise ValueError(f'Exact average-linkage EAC requires N <= {max_n}; got {n}. '
                         'Use eac_classic (sparse spectral EAC) for large datasets.')
    A=G.toarray().astype(np.float32)
    A=np.maximum(A,A.T);np.fill_diagonal(A,1.)
    D=np.maximum(0,1-A)
    return AgglomerativeClustering(n_clusters=int(k),linkage='average',metric='precomputed').fit_predict(D)
