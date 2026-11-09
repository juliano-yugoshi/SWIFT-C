"""Reproducible 33-family synthetic suite from the established controlled generators."""
from __future__ import annotations
import json,csv
from pathlib import Path
import numpy as np
from .generators import legacy,extended

MAPPING={"id_column":"video_id","text_columns":["desc_smolvlm"],
         "label_columns":{"gemini":"sug_gemini","claude":"sug_claude","llama":"sug_llama"},
         "target_column":"category","delimiter":",","encoding":"utf-8-sig","text_joiner":"\n"}

SCENARIOS={cfg['name']:("legacy",cfg) for cfg in legacy.SCENARIOS}
SCENARIOS.update({cfg['name']:("extended",cfg) for cfg in extended.SCENARIOS})
assert len(SCENARIOS)==33, 'Expected 33 distinct scenario families'

QUICK_SCENARIOS=['sanity_clean_separable','sanity_paraphrase_labels',
                 'hard_generic_weak_labels','hard_labels_help_only_boundaries',
                 'hard_hierarchical_granularity','hard_correlated_llm_errors']

def choose_scenarios(selector):
    if selector=='all':return list(SCENARIOS)
    if selector=='quick':return QUICK_SCENARIOS
    if selector=='sanity':return [x for x,(group,cfg) in SCENARIOS.items() if cfg['tier'] in ('sanity','control')]
    if selector=='hard':return [x for x,(group,cfg) in SCENARIOS.items() if cfg['tier']=='hard']
    names=[s.strip() for s in selector.split(',') if s.strip()]
    unknown=set(names)-set(SCENARIOS)
    if unknown:raise ValueError(f'Unknown scenarios: {sorted(unknown)}')
    return names

def _write_records(path,records):
    if not records:return
    with open(path,'w',encoding='utf-8',newline='') as fp:
        w=csv.DictWriter(fp,fieldnames=list(records[0]));w.writeheader();w.writerows(records)

def generate(root,scenario_names,seed=20261107,dim=96,overwrite=False):
    root=Path(root);(root/'data').mkdir(parents=True,exist_ok=True)
    (root/'configs').mkdir(exist_ok=True)
    (root/'configs'/'column_mapping.json').write_text(json.dumps(MAPPING,indent=2))
    meta=[]
    for name in scenario_names:
        group,cfg=SCENARIOS[name]
        folder=root/'data'/name
        npz=folder/'synthetic_embeddings.npz'
        if overwrite or not npz.exists():
            folder.mkdir(parents=True,exist_ok=True)
            if group=='legacy':
                rows,diagnostics,X,L,M,y,_=legacy.generate(cfg,seed,dim)
            else:
                rows,diagnostics,X,L,M,y=extended.generate(cfg,seed,dim)
            _write_records(folder/'dataset.csv',rows)
            _write_records(folder/'diagnostics_DO_NOT_USE_FOR_TRAINING.csv',diagnostics)
            np.savez_compressed(npz,X=X,labels=L,label_mask=M,y=y,
                                ids=np.array([r['video_id'] for r in rows]),
                                source_names=np.array(['gemini','claude','llama']))
            (folder/'scenario.json').write_text(json.dumps({**cfg,'seed':seed,'dimension':dim,'type':'simulated embeddings; NOT Qwen'},indent=2))
        with np.load(npz) as d:
            n,k=len(d['X']),cfg['k']
        meta.append({'name':name,'tier':cfg['tier'],'k':k,'n':n,'generator_seed':seed,'generator':group})
    (root/'manifest.json').write_text(json.dumps(meta,indent=2))
    (root/'generation_provenance.json').write_text(json.dumps({'seed':seed,'dim':dim,'scenario_count':len(meta),
                                                             'embeddings':'synthetic, shared semantic space (not Qwen)'},indent=2))
    return meta
