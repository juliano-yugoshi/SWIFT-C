import numpy as np
from swiftc.clustering.runner import Tracker,run_one_dataset,source_groups,check_protocol
from swiftc.clustering.core import normalized
import pytest


def test_sources():
    names=['a','b','c']
    assert len(source_groups(names,'all,each'))==4
    assert source_groups(names,'a+b')['a+b']==[0,1]


def test_run_one_dataset_and_resume(tmp_path):
    n=96;d=24
    rng=np.random.default_rng(12)
    y=np.arange(n)%4
    centers=np.eye(4,d)
    X=normalized(centers[y]+rng.normal(0,.1,(n,d)))
    L=normalized((centers[y,None,:]+rng.normal(0,.1,(n,2,d))).reshape(-1,d)).reshape(n,2,d)
    M=np.ones((n,2),bool)
    tracker=Tracker(tmp_path,progress='plain',verbose=False)
    keys=set();rows=[]
    def save(r,*args):rows.append(r)
    a,g=run_one_dataset(X,L,M,y,np.arange(n),['a','b'],4,[42],
                        'all,each',tmp_path,tracker,'toy','signature',factors=(3,5),reps=1,
                        existing=keys,callback=save)
    assert len(a)==15
    assert len(keys)==15
    assert all('ari' in r and r['k']==4 for r in a)
    b,_=run_one_dataset(X,L,M,y,np.arange(n),['a','b'],4,[42],
                        'all,each',tmp_path,tracker,'toy','signature',factors=(3,5),reps=1,
                        existing=keys,callback=save)
    assert len(b)==0


def test_incompatible_resume_rejected(tmp_path):
    check_protocol(tmp_path,{'k':4},resume=True)
    with pytest.raises(ValueError):check_protocol(tmp_path,{'k':5},resume=True)
