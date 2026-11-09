"""Classical DEC baseline, with AE pretraining then KL-only cluster training.

Torch is imported lazily, so the quick CPU benchmark needs no deep extras.
No weak-label supervision or ground truth enters the DEC training routine.
"""
from __future__ import annotations
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize


def dec_predict(V,k,seed=42,latent_dim=32,pretrain_epochs=8,dec_epochs=15,
                batch_size=256,learning_rate=1e-3,device='cpu',callback=None):
    try:
        import torch
        from torch import nn
        from torch.nn import functional as F
    except ImportError as ex:
        raise RuntimeError('DEC needs PyTorch: pip install -e ".[article]"') from ex
    torch.manual_seed(int(seed));np.random.seed(int(seed))
    # Deterministic runs and fixed preprocessing; no labels inspected.
    torch.set_num_threads(1)
    V=normalize(np.asarray(V,dtype=np.float32)).astype(np.float32)
    n,d=V.shape
    latent_dim=min(int(latent_dim),max(2,d//2))
    h1=min(256,max(64,d));h2=min(128,max(32,d//2))
    class AE(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder=nn.Sequential(nn.Linear(d,h1),nn.ReLU(),nn.Linear(h1,h2),nn.ReLU(),nn.Linear(h2,latent_dim))
            self.decoder=nn.Sequential(nn.Linear(latent_dim,h2),nn.ReLU(),nn.Linear(h2,h1),nn.ReLU(),nn.Linear(h1,d))
        def forward(self,x):
            z=self.encoder(x);return z,self.decoder(z)
    dev=torch.device(device if device and device!='auto' else ('cuda' if torch.cuda.is_available() else 'cpu'))
    model=AE().to(dev)
    x=torch.as_tensor(V,device=dev)
    opt=torch.optim.Adam(model.parameters(),lr=learning_rate)
    gen=torch.Generator(device='cpu').manual_seed(seed)
    for epoch in range(pretrain_epochs):
        model.train()
        perm=torch.randperm(n,generator=gen)
        losses=[]
        for idx in perm.split(batch_size):
            z,recon=model(x[idx.to(dev)]);loss=F.mse_loss(recon,x[idx.to(dev)])
            opt.zero_grad();loss.backward();opt.step();losses.append(float(loss.detach()))
        if callback:callback('pretrain',epoch+1,pretrain_epochs,float(np.mean(losses)))
    with torch.no_grad():
        z=model.encoder(x).cpu().numpy()
    centers=KMeans(n_clusters=k,n_init=5,random_state=int(seed)).fit(z).cluster_centers_
    centers=torch.nn.Parameter(torch.tensor(centers,dtype=torch.float32,device=dev))
    # Classical DEC: train the encoder and centroids using KL(P||Q), decoder frozen.
    opt=torch.optim.Adam(list(model.encoder.parameters())+[centers],lr=learning_rate/2)
    scale=float(np.median(np.linalg.norm(z-centers.detach().cpu().numpy()[np.argmin(((z[:,None,:]-centers.detach().cpu().numpy()[None,:,:])**2).sum(-1),axis=1)],axis=1)))
    scale=max(scale,1e-3)
    def assignments(zz):
        dist=((zz[:,None,:]-centers[None,:,:])**2).sum(-1)/scale**2
        q=(1+dist).pow(-1)
        return q/q.sum(1,keepdim=True).clamp_min(1e-10)
    for epoch in range(dec_epochs):
        model.encoder.eval()
        with torch.no_grad():
            q=assignments(model.encoder(x)).detach()
            freq=q.sum(0).clamp_min(1e-8)
            p=(q*q)/freq
            p=(p/p.sum(1,keepdim=True).clamp_min(1e-10)).detach()
        perm=torch.randperm(n,generator=gen)
        losses=[]
        for idx in perm.split(batch_size):
            indices=idx.to(dev)
            qmini=assignments(model.encoder(x[indices]))
            loss=F.kl_div(qmini.clamp_min(1e-10).log(),p[indices],reduction='batchmean')
            opt.zero_grad();loss.backward();opt.step();losses.append(float(loss.detach()))
        if callback:callback('dec_kl',epoch+1,dec_epochs,float(np.mean(losses)))
    with torch.no_grad():out=assignments(model.encoder(x)).argmax(1).cpu().numpy()
    return out
