#!/usr/bin/env python3
"""Deterministic synthetic weak-label clustering benchmark.

Produces paired CSV datasets for Qwen-compatible workflows and synthetic
co-embedded input/weak-label vectors for an offline sanity benchmark.
Ground truth is used only for generating controlled challenges/evaluation.
"""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
import numpy as np

FAMILIES = ["food", "sport", "travel", "animals"]
CLASSES = [
    ("cooking_tutorial", 0, ["cooking tutorial", "meal preparation lesson", "step-by-step recipe", "culinary instruction", "how to cook a dish"],
     ["A person demonstrates how to prepare a dish.", "Ingredients are combined as someone explains a recipe.", "An instructional segment shows meal preparation."]),
    ("food_review", 0, ["restaurant review", "food tasting", "dining experience", "reviewing a meal", "culinary critique"],
     ["A person tastes a dish and comments on it.", "Visitors sample food and discuss its quality.", "A dining review takes place at a restaurant."]),
    ("soccer_match", 1, ["soccer match", "football game highlights", "team soccer action", "goals and passing", "soccer gameplay"],
     ["Players pass a ball across a soccer field.", "Athletes compete in a soccer game.", "A soccer match includes shots towards the goal."]),
    ("basketball_match", 1, ["basketball match", "basketball game highlights", "court basketball action", "shooting hoops", "basketball gameplay"],
     ["Players shoot hoops on a basketball court.", "Athletes compete in a basketball game.", "A basketball match shows passes and baskets."]),
    ("urban_trip", 2, ["city walking tour", "urban sightseeing", "downtown travel", "exploring city landmarks", "city tourism"],
     ["A traveler walks through busy urban streets.", "People visit buildings and landmarks in a city.", "A tourist explores a downtown neighborhood."]),
    ("nature_hike", 2, ["mountain hiking", "forest trail adventure", "nature trekking", "outdoor wilderness walk", "trail exploration"],
     ["A person hikes along a natural trail.", "Travelers walk through a forest landscape.", "A hiker explores a mountain path."]),
    ("dog_training", 3, ["dog training", "pet obedience", "canine behavior lessons", "teaching a dog commands", "pet skills demonstration"],
     ["A dog follows commands given by a person.", "A trainer demonstrates obedience exercises with a pet.", "Someone teaches a dog a new behavior."]),
    ("wildlife_documentary", 3, ["wildlife documentary", "wild animal observation", "animals in natural habitat", "nature wildlife footage", "wildlife behavior"],
     ["Wild animals are observed in their habitat.", "Nature footage shows animals moving outdoors.", "A documentary follows wildlife behavior."]),
]
FAMILY_LABELS = {
    0:["food", "culinary arts", "food-related content", "meals and cuisine"],
    1:["sports", "athletic activity", "competitive sports", "team sports"],
    2:["travel", "exploration", "tourism", "places to visit"],
    3:["animals", "animal life", "nature and animals", "animal behavior"],
}
CROSS_LABELS = ["daily activities", "entertainment", "general interest", "people and events", "lifestyle", "outdoor activities"]
LLMS = ("gemini", "claude", "llama")
STYLES = ("handheld footage", "studio recording", "short compilation", "close-up shots", "narrated clip", "wide-angle footage")
GENERIC_DESC = {
    0:"People interact with a variety of items while discussing an activity.",
    1:"A group of people moves quickly in a competitive setting.",
    2:"A person moves through a location while observing surroundings.",
    3:"An animal appears in a sequence of recorded scenes.",
}

SCENARIOS = [
    dict(name="sanity_clean_separable",tier="sanity",n=480,k=8,input_family=.80,input_class=1.35,noise=.22,style=.10,correct=.98,generic=.02,missing=.00),
    dict(name="sanity_paraphrase_labels",tier="sanity",n=480,k=8,input_family=.80,input_class=1.35,noise=.24,style=.10,correct=.98,generic=.02,missing=.00,paraphrase=True),
    dict(name="sanity_one_noisy_llm",tier="sanity",n=480,k=8,input_family=.82,input_class=1.25,noise=.32,style=.10,correct=.98,generic=.02,missing=.00,adversary="llama"),
    dict(name="sanity_missing_labels",tier="sanity",n=480,k=8,input_family=.80,input_class=1.28,noise=.30,style=.10,correct=.92,generic=.06,missing=.42),
    dict(name="hard_overlap_fine_grained",tier="hard",n=800,k=8,input_family=1.30,input_class=.22,noise=1.10,style=.70,correct=.76,generic=.16,missing=.02),
    dict(name="hard_generic_weak_labels",tier="hard",n=800,k=8,input_family=1.20,input_class=.32,noise=.88,style=.72,correct=.35,generic=.58,missing=.02),
    dict(name="hard_correlated_llm_errors",tier="hard",n=800,k=8,input_family=1.12,input_class=.25,noise=1.05,style=.72,correct=.82,generic=.10,missing=.02,correlated_wrong=.35),
    dict(name="hard_imbalanced_rare_classes",tier="hard",n=900,k=8,input_family=1.18,input_class=.57,noise=.78,style=.45,correct=.76,generic=.18,missing=.04,imbalance=True),
    dict(name="hard_complementary_weak_labels",tier="hard",n=800,k=8,input_family=1.40,input_class=.25,noise=.78,style=.66,correct=.92,generic=.04,missing=.02,coarse_desc=.72),
    dict(name="hard_polysemy_cross_category",tier="hard",n=800,k=8,input_family=1.25,input_class=.24,noise=1.10,style=.78,correct=.70,generic=.18,missing=.02,polysemy=.26),
    dict(name="hard_missing_outliers",tier="hard",n=900,k=8,input_family=1.08,input_class=.60,noise=.84,style=.44,correct=.74,generic=.18,missing=.38,outliers=.10),
    dict(name="hard_style_nuisance",tier="hard",n=800,k=8,input_family=1.02,input_class=.48,noise=.63,style=1.40,correct=.84,generic=.10,missing=.02),
    dict(name="hard_mixed_granularity",tier="hard",n=800,k=8,input_family=1.22,input_class=.45,noise=.80,style=.48,correct=.60,generic=.32,missing=.03,mixed_granularity=True),
]


def normalize(a,axis=-1):
    return a / np.maximum(np.linalg.norm(a,axis=axis,keepdims=True),1e-9)


def generate(cfg, seed=20261107, dim=96):
    seed = seed + sum(ord(c)*i for i,c in enumerate(cfg['name'],1))
    rng=np.random.default_rng(seed)
    C=cfg['k']; n=cfg['n']
    family_vectors=normalize(rng.normal(size=(4,dim)))
    subtype_vectors=normalize(rng.normal(size=(C,dim)))
    style_vectors=normalize(rng.normal(size=(len(STYLES),dim)))
    # Semantic components shared between the synthetic input and label embeddings.
    # Thus semantic compatibility is controllable and no model download is needed.
    class_centers=normalize(.85*family_vectors[np.arange(C)//2] + 1.20*subtype_vectors)
    coarse_centers=family_vectors
    if cfg.get('imbalance'):
        proportions=np.array([.29,.23,.19,.13,.08,.045,.025,.01]); proportions/=proportions.sum()
    else:
        proportions=np.ones(C)/C
    counts=rng.multinomial(n,proportions)
    if cfg.get('imbalance'):
        counts[-1]=max(counts[-1],5)
        counts[0]=n-counts[1:].sum()
    y=np.concatenate([np.full(c,i,int) for i,c in enumerate(counts)])
    rng.shuffle(y)
    x=np.zeros((n,dim),np.float32)
    weak=np.zeros((n,len(LLMS),dim),np.float32)
    mask=np.zeros((n,len(LLMS)),dtype=bool)
    rows=[]; metadata=[]
    # Fixed phrase embeddings: equal text has equal semantic representation (up to embedding noise).
    lexicon={}
    phrase_semantic_type={}
    for k, (_,family, phrases,_) in enumerate(CLASSES):
        for phrase in phrases:
            lexicon[phrase]=normalize(1.0*family_vectors[family]+1.50*subtype_vectors[k]+.09*rng.normal(size=dim))
            phrase_semantic_type[phrase]=('specific',k)
    for family,phrases in FAMILY_LABELS.items():
        for phrase in phrases:
            lexicon[phrase]=normalize(1.30*family_vectors[family]+.08*rng.normal(size=dim))
            phrase_semantic_type[phrase]=('generic',family)
    for phrase in CROSS_LABELS:
        lexicon[phrase]=normalize(.30*rng.normal(size=dim))
        phrase_semantic_type[phrase]=('cross',-1)
    for i,k in enumerate(y):
        cls,family,phrases,descs=CLASSES[int(k)]
        nuisance=int(rng.integers(len(STYLES)))
        is_outlier=bool(rng.random()<cfg.get('outliers',0))
        base=cfg['input_family']*family_vectors[family]+cfg['input_class']*subtype_vectors[k]
        if is_outlier: base= .15*base+1.8*rng.normal(size=dim)/np.sqrt(dim)
        noise=cfg['noise']*rng.normal(size=dim)/np.sqrt(dim)
        x[i]=normalize(base+cfg['style']*style_vectors[nuisance]+noise).astype(np.float32)
        if rng.random()<cfg.get('coarse_desc',.08):
            description=GENERIC_DESC[family]
        else:
            description=descs[int(rng.integers(len(descs)))]
        if cfg.get('paraphrase'):
            description=description.replace('A person','Someone').replace('A ','One ').replace('a ','one ')
        description += f" The clip uses {STYLES[nuisance]}."
        selected=[]; types=[]
        # Correlated wrong labels emulate agreement between LLMs that is systematically incorrect.
        correlated=bool(rng.random()<cfg.get('correlated_wrong',0))
        wrong_k=(k^1) if cfg.get('correlated_wrong',0) else ((k+2)%C)
        for m,llm in enumerate(LLMS):
            if rng.random()<cfg['missing']:
                label=''; subtype='missing'
            elif cfg.get('adversary')==llm:
                subtype='wrong'; label=str(rng.choice(CLASSES[int((k+2)%C)][2]))
            elif correlated and m < 2:
                subtype='correlated_wrong';label=str(rng.choice(CLASSES[int(wrong_k)][2]))
            else:
                r=rng.random()
                if r<cfg['correct']:
                    subtype='specific'; label=str(rng.choice(phrases))
                elif r<cfg['correct']+cfg['generic']:
                    subtype='generic'; label=str(rng.choice(FAMILY_LABELS[family]))
                else:
                    subtype='wrong';
                    wrong=int(rng.choice([j for j in range(C) if j!=k]))
                    label=str(rng.choice(CLASSES[wrong][2]))
            if label and cfg.get('paraphrase'):
                base_label=label
                label=str(rng.choice([
                    f"short video showing {base_label}",
                    f"visual content about {base_label}",
                    f"footage of {base_label}",
                    f"{base_label} scene",
                    f"topic: {base_label}",
                    f"{base_label} activity",
                    f"a segment featuring {base_label}",
                    f"overview of {base_label}",
                    base_label.title(),
                ]))
                if label not in lexicon:
                    lexicon[label]=normalize(lexicon[base_label]+.025*rng.normal(size=dim))
            if label and cfg.get('polysemy',0) and rng.random()<cfg['polysemy']:
                label=str(rng.choice(["animal behavior", "sports", "exploration", "culinary arts", "outdoor activities"]))
                subtype='polysemy'
            if label and cfg.get('mixed_granularity') and m==0:
                label=str(rng.choice(FAMILY_LABELS[family])); subtype='generic'
            selected.append(label); types.append(subtype)
            if label:
                # The same text is encoded consistently, independent of the true label.
                # Label errors still refer to their (wrong) semantic concept.
                weak[i,m]=normalize(lexicon[label]+.035*rng.normal(size=dim)).astype(np.float32)
                mask[i,m]=True
        row={"video_id":f"{cfg['name']}_{i:05d}","category":cls,
             "desc_smolvlm":description,"sug_gemini":selected[0],
             "sug_claude":selected[1],"sug_llama":selected[2]}
        rows.append(row)
        metadata.append({"video_id":row['video_id'],"true_class_id":int(k),
                         "family":FAMILIES[family],"is_outlier":is_outlier,
                         "input_style":STYLES[nuisance],
                         "type_gemini":types[0],"type_claude":types[1],"type_llama":types[2]})
    return rows,metadata,x,weak,mask,y,lexicon


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',default='.')
    p.add_argument('--seed',type=int,default=20261107)
    p.add_argument('--dimension',type=int,default=96)
    p.add_argument('--scenario',default='all',help='all, sanity, hard, or a specific scenario name')
    a=p.parse_args();root=Path(a.output);(root/'data').mkdir(parents=True,exist_ok=True);(root/'configs').mkdir(exist_ok=True)
    mapping={"id_column":"video_id","text_columns":["desc_smolvlm"],"label_columns":{"gemini":"sug_gemini","claude":"sug_claude","llama":"sug_llama"},"target_column":"category","delimiter":",","encoding":"utf-8-sig","text_joiner":"\n"}
    (root/'configs'/'column_mapping.json').write_text(json.dumps(mapping,indent=2)+'\n')
    manifest=[]
    for cfg in SCENARIOS:
        if a.scenario not in ('all',cfg['tier'],cfg['name']): continue
        rows,meta,x,w,mask,y,lexicon=generate(cfg,a.seed,a.dimension)
        base=root/'data'/cfg['name']; base.mkdir(exist_ok=True)
        for fn,records in [('dataset.csv',rows),('diagnostics_DO_NOT_USE_FOR_TRAINING.csv',meta)]:
            with (base/fn).open('w',encoding='utf-8',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
        np.savez_compressed(base/'synthetic_embeddings.npz',X=x,labels=w,label_mask=mask,y=y,ids=np.array([r['video_id'] for r in rows]),source_names=np.array(LLMS),class_names=np.array([v[0] for v in CLASSES]))
        with (base/'scenario.json').open('w') as f: json.dump({**cfg,'seed':a.seed,'embedding_dim':a.dimension,'generated_rows':len(rows),'counts':{CLASSES[j][0]:int((y==j).sum()) for j in range(cfg['k'])}},f,indent=2)
        manifest.append({'name':cfg['name'],'tier':cfg['tier'],'n':len(y),'k':cfg['k'],'dim':a.dimension,'mechanism':scenario_description(cfg)})
        print(f"{cfg['name']:36s} n={len(y):4d} k={cfg['k']:2d} missing={1-mask.mean():.1%}")
    (root/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('Generated',len(manifest),'scenarios at',root)


def scenario_description(cfg):
    name=cfg['name']
    return {
      'sanity_clean_separable':'Separable instance clusters, high-quality synonymous labels.',
      'sanity_paraphrase_labels':'Many lexical paraphrases, same semantic concepts.',
      'sanity_one_noisy_llm':'One systematically incorrect weak label source.',
      'sanity_missing_labels':'Randomly missing suggestions from all labelers.',
      'hard_overlap_fine_grained':'Adjacent subcategories overlap in instance space.',
      'hard_generic_weak_labels':'Broad family labels shared by different fine classes.',
      'hard_correlated_llm_errors':'Two LLMs agree on the same wrong fine-grained class.',
      'hard_imbalanced_rare_classes':'Severely imbalanced cluster sizes, including rare categories.',
      'hard_complementary_weak_labels':'Instance embeddings emphasize coarse families while weak labels reveal subtypes.',
      'hard_polysemy_cross_category':'Labels are reused across categories and/or semantic contexts.',
      'hard_missing_outliers':'Many weak labels missing plus out-of-distribution instances.',
      'hard_style_nuisance':'Nonsemantic recording-style effects dominate embedding geometry.',
      'hard_mixed_granularity':'Different LLMs prefer broad family or specific subtype labels.',
    }[name]

if __name__=='__main__': main()
