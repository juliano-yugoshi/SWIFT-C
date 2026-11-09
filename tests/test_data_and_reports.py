import json
import numpy as np
import pandas as pd
from swiftc.clustering.data import inspect_columns,guess_mapping,read_csv,extract_texts,embed_texts
from swiftc.clustering.report import write_report


def test_generic_csv_mapping_and_cache(tmp_path):
    path=tmp_path/'generic.csv'
    pd.DataFrame({'uid':['a','b','c'],'description':['red apple','green pear','fruit'],
                  'sug_modelA':['fruit','pear',''],'ground_truth':['A','B','A']}).to_csv(path,index=False)
    cols=inspect_columns(path)
    mapping={'id_column':'uid','text_columns':['description'],'label_columns':{'modelA':'sug_modelA'},
             'target_column':'ground_truth','delimiter':',','encoding':'utf-8','text_joiner':'\n'}
    ds=extract_texts(read_csv(path,mapping),mapping)
    x,L,M,cache,hit=embed_texts(ds,encoder='hashing',cache_dir=tmp_path/'cache')
    xx,LL,MM,cc,hit2=embed_texts(ds,encoder='hashing',cache_dir=tmp_path/'cache')
    assert x.shape==(3,1024) and L.shape==(3,1,1024)
    assert bool(hit2) and not bool(hit)
    assert np.allclose(x,xx) and np.array_equal(M,MM)
    assert M[:,0].tolist()==[True,True,False]


def test_paired_report(tmp_path):
    f=tmp_path/'runs.csv'
    rows=[]
    for case in ['a','b','c']:
        for m,score in [('fusion',.7),('eac_adaptive',.72)]:
            rows.append({'data_kind':'synthetic','source_set':'all','k_factor':1,'k':4,
                         'scenario':case,'tier':'hard','generation_seed':99,'seed':42,
                         'method':m,'ari':score,'nmi':score})
    pd.DataFrame(rows).to_csv(f,index=False)
    summ,pairs=write_report(f,tmp_path/'report')
    assert (tmp_path/'report'/'REPORT.md').exists()
    assert round(float(pairs.loc[pairs.method=='eac_adaptive','mean_delta_ari'].iloc[0]),4)==.02
