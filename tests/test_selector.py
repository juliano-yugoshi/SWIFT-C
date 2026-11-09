import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from swiftc.selection import cli as s

class SelectorTest(unittest.TestCase):
    def make_fixture(self, root):
        methods=['input','labels_only','fusion','eac_uniform','eac_adaptive','spherical_fusion','spectral_fusion']
        rows=[]
        for sc in ['hard_one','hard_two','hard_three']:
            for gen in [100,200]:
                for seed in [42,123]:
                    for m in methods:
                        delta={'input':-.08,'labels_only':-.04,'fusion':0,'eac_uniform':-.01,
                               'eac_adaptive':.02,'spherical_fusion':.04,'spectral_fusion':.01}[m]
                        rows.append(dict(data_kind='synthetic',scenario=sc,tier='hard',generation_seed=gen,
                             source_set='all',k_factor=1.,k=8,seed=seed,method=m,
                             ari=.62+delta,nmi=.71+delta*.8,duration_s=2.))
        path=root/'runs.csv'
        pd.DataFrame(rows).to_csv(path,index=False)
        (root/'run_protocol.json').write_text(json.dumps(dict(kind='synthetic',scenarios=['hard_one','hard_two','hard_three'],generations=[100,200],seeds=[42,123],k_factors=[1.],sources='all',methods=methods)))
        return path
    def test_end_to_end(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            self.make_fixture(root)
            ret=s.main(['--results',str(root),'--output',str(root/'selection'),'--top-n','2','--device','cuda','--cache-dir','cached/embeddings'])
            self.assertEqual(ret,0)
            conf=json.loads((root/'selection'/'selected_methods.json').read_text())
            self.assertEqual(conf['selected_methods'][:4],list(s.MANDATORY))
            self.assertIn('spherical_fusion',conf['selected_methods'])
            script=(root/'selection'/'run_realworld_main.sh').read_text()
            self.assertIn('--methods',script)
            self.assertIn('--cache-dir cached/embeddings',script)
            self.assertIn('--clusters 20',script)
            self.assertIn('--seeds 42,123,2025',script)
            self.assertTrue((root/'selection'/'paired_comparisons.csv').exists())
    def test_partial_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            path=self.make_fixture(root)
            frame=pd.read_csv(path)
            frame=frame[~((frame.method=='fusion')&(frame.generation_seed==200))]
            frame.to_csv(path,index=False)
            with self.assertRaises(ValueError):
                s.main(['--results',str(root),'--output',str(root/'selection')])
    def test_paired_coverage(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            path=self.make_fixture(root)
            frame=pd.read_csv(path)
            frame=frame[~((frame.method=='spectral_fusion')&(frame.generation_seed==200))]
            frame.to_csv(path,index=False)
            data,_=s.load_runs(root,False,'all',1.)
            rank,_,_=s.metrics(data,.8)
            row=rank[rank.method=='spectral_fusion'].iloc[0]
            self.assertFalse(row.eligible)
    def test_sanity_excluded(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            path=self.make_fixture(root)
            frame=pd.read_csv(path)
            fake=frame.iloc[:1].copy()
            fake['scenario']='sanity_easy';fake['tier']='sanity'
            pd.concat([frame,fake]).to_csv(path,index=False)
            data,_=s.load_runs(root,False,'all',1.)
            self.assertNotIn('sanity_easy',set(data.scenario))

if __name__=='__main__':unittest.main()
