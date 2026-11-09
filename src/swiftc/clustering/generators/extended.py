#!/usr/bin/env python3
"""Controlled multi-view weak-label benchmark, with no LLM/encoder calls.

Each artifact is self-contained, re-generable from seed, and suitable for
unsupervised training with ground truth isolated to evaluation.
"""
from __future__ import annotations
import argparse, csv, json, hashlib
from pathlib import Path
import numpy as np

LLMS=('gemini','claude','llama')
# Eight parent semantic families, with two subclasses each. No special
# classes are inserted into model features by the evaluator.
CLASSES=[
 ('cooking_tutorial','food',('cooking tutorial','meal preparation lesson','recipe instructions','culinary demonstration'),('A cook demonstrates a recipe, step by step.','Ingredients are prepared in a cooking lesson.')),
 ('restaurant_review','food',('restaurant review','food tasting review','dining critique','evaluating a restaurant'),('A visitor reviews food served at a restaurant.','A diner describes the taste of a meal.')),
 ('soccer_match','sports',('football match','soccer game','soccer highlights','players scoring a goal'),('Players compete in a soccer match on a field.','Athletes play football and aim for the goal.')),
 ('basketball_game','sports',('basketball game','shooting hoops','court basketball','basketball highlights'),('Players compete on a basketball court.','Athletes take shots at the hoop.')),
 ('city_sightseeing','travel',('city tourism','urban walking tour','downtown sightseeing','city landmark visit'),('A traveler explores streets and buildings in a city.','A tourist walks past urban landmarks.')),
 ('mountain_hiking','travel',('mountain hiking','forest trekking','outdoor trail hiking','nature trail expedition'),('Hikers explore a mountain trail.','A traveler hikes through the forest.')),
 ('dog_training','animals',('dog obedience training','pet dog commands','canine training','teaching a dog tricks'),('A trainer teaches commands to a dog.','A dog performs an obedience exercise.')),
 ('wildlife_observation','animals',('wildlife documentary','wild animals in nature','wildlife observation','animals in habitat'),('Wild animals are observed in their habitat.','A nature documentary follows wildlife.')),
 ('ceramic_craft','arts',('pottery making','ceramic crafting','clay modeling','pottery workshop'),('An artist shapes clay into a ceramic object.','A craftsperson demonstrates pottery making.')),
 ('painting_lesson','arts',('painting tutorial','drawing and painting','art painting lesson','brush painting techniques'),('An artist teaches a painting technique.','Paint is applied to a canvas in a lesson.')),
 ('medical_education','education',('health education','medical teaching','healthcare lecture','medicine demonstration'),('An instructor teaches a medical concept.','A lesson explains a healthcare procedure.')),
 ('technology_tutorial','education',('software tutorial','computer teaching','technology lesson','technical instruction'),('An instructor demonstrates software usage.','A teacher explains a computer program.')),
 ('train_transport','transport',('train journey','rail transport','railway travel','travel by train'),('A train passes along a railway.','Passengers board a train at a station.')),
 ('motorcycle_ride','transport',('motorcycle riding','motorbike journey','motorcycle transport','motorcycling adventure'),('A rider travels on a motorcycle.','A motorbike moves down a road.')),
 ('live_music','performance',('live music concert','musical performance','stage singing','live band performance'),('Musicians perform live on a stage.','A singer performs for an audience.')),
 ('standup_comedy','performance',('stand-up comedy show','comedian performance','live comedy act','humorous stage routine'),('A comedian tells jokes on stage.','An audience watches a stand-up comedy routine.')),
]
FAMILIES=['food','sports','travel','animals','arts','education','transport','performance']
FAMILY_LABELS={
 'food':['food content','culinary activity','food and dining','meals and cooking'],
 'sports':['sports video','athletics','competitive sports','sports highlights'],
 'travel':['travel footage','tourism','outdoor exploration','places to visit'],
 'animals':['animal content','animal behavior','nature and pets','wildlife and animals'],
 'arts':['creative activities','artistic demonstration','arts and crafts','visual arts'],
 'education':['educational video','informative tutorial','learning activity','instructional content'],
 'transport':['transportation','vehicles in motion','travel by vehicle','transport infrastructure'],
 'performance':['stage performance','entertainment show','performing arts','live performance'],
}
GENERIC_LABELS=('general interest','everyday life','people and activities','interesting video','educational and entertainment content','outdoor activities','life and culture')
STYLES=('studio footage','mobile camera','panoramic shots','narrated compilation','close-up view','low-light footage')

# The first four are explicit negative controls; the others stress particular
# hypotheses about EAC, multiple labelers, or discriminative weak supervision.
SCENARIOS=[
 dict(name='control_labels_random',tier='control',n=680,k=8,geometry='separable',label_mode='random',input_strength=1.6,label_noise=.07),
 dict(name='control_labels_perfect',tier='control',n=680,k=8,geometry='overlap',label_mode='perfect',input_strength=.42,label_noise=.03),
 dict(name='control_labels_all_generic',tier='control',n=680,k=8,geometry='overlap',label_mode='all_generic',input_strength=.65,label_noise=.03),
 dict(name='control_permuted_wrong_labels',tier='control',n=680,k=8,geometry='separable',label_mode='correlated_all_wrong',input_strength=1.8,label_noise=.03),
 dict(name='hard_eac_multimodal_subclusters',tier='hard',n=960,k=8,geometry='submodes',label_mode='good',input_strength=.52,label_noise=.08),
 dict(name='hard_eac_interleaved_moons',tier='hard',n=880,k=8,geometry='moons',label_mode='mixed',input_strength=.12,label_noise=.09),
 dict(name='hard_eac_bridge_chains',tier='hard',n=880,k=8,geometry='bridges',label_mode='boundary_help',input_strength=.75,label_noise=.07),
 dict(name='hard_heteroscedastic_densities',tier='hard',n=960,k=8,geometry='heteroscedastic',label_mode='mixed',input_strength=.9,label_noise=.07),
 dict(name='hard_shared_label_hubs',tier='hard',n=860,k=8,geometry='overlap',label_mode='hubs',input_strength=.55,label_noise=.05),
 dict(name='hard_conditional_llm_specialists',tier='hard',n=880,k=8,geometry='overlap',label_mode='specialists',input_strength=.55,label_noise=.07),
 dict(name='hard_confident_wrong_majority',tier='hard',n=880,k=8,geometry='overlap',label_mode='wrong_majority',input_strength=.65,label_noise=.07),
 dict(name='hard_labels_help_only_boundaries',tier='hard',n=880,k=8,geometry='bridges',label_mode='boundary_help',input_strength=.85,label_noise=.07),
 dict(name='hard_missing_not_at_random',tier='hard',n=900,k=8,geometry='overlap',label_mode='mnar',input_strength=.68,label_noise=.07),
 dict(name='hard_source_domain_shift',tier='hard',n=900,k=8,geometry='overlap',label_mode='domain_shift',input_strength=.63,label_noise=.07),
 dict(name='hard_duplicate_cross_class',tier='hard',n=900,k=8,geometry='duplicate',label_mode='good',input_strength=.74,label_noise=.07),
 dict(name='hard_hubness_anisotropy',tier='hard',n=920,k=8,geometry='anisotropic',label_mode='mixed',input_strength=.77,label_noise=.07),
 dict(name='hard_many_classes_16',tier='hard',n=1600,k=16,geometry='overlap',label_mode='mixed',input_strength=.72,label_noise=.07),
 dict(name='hard_extreme_imbalance',tier='hard',n=1400,k=12,geometry='heteroscedastic',label_mode='mixed',input_strength=.78,label_noise=.07,imbalanced=True),
 dict(name='hard_hierarchical_granularity',tier='hard',n=1000,k=12,geometry='overlap',label_mode='hierarchy',input_strength=.63,label_noise=.07),
 dict(name='hard_spurious_eac_consensus',tier='hard',n=960,k=8,geometry='spurious',label_mode='mixed',input_strength=.55,label_noise=.07),
]

def norm(v, axis=-1):
    v=np.asarray(v)
    return v/np.maximum(np.linalg.norm(v,axis=axis,keepdims=True),1e-9)

def stable_seed(name,seed):
    return (int.from_bytes(hashlib.blake2b(name.encode(),digest_size=8).digest(),'little')+seed)%(2**32-1)

def generated_phrase(base,rng,variation):
    if not variation: return base
    return str(rng.choice((base,'footage showing '+base,'a short clip about '+base,
                           'visual content featuring '+base,base+' segment',base.title(),
                           'video of '+base,'an activity involving '+base)))

def generate(cfg, seed=42, dim=96):
    rng=np.random.default_rng(stable_seed(cfg['name'],seed)); n=cfg['n']; k=cfg['k']
    nf=(k+1)//2
    # Represent concepts in one shared latent basis. Class and family vectors
    # are identical for all text paraphrases of the same concept.
    family=norm(rng.normal(size=(nf,dim)))
    subtype=norm(rng.normal(size=(k,dim)))
    style=norm(rng.normal(size=(len(STYLES),dim)))
    manifold=norm(rng.normal(size=(2,dim)))
    common=norm(rng.normal(size=dim))
    mode_vecs=norm(rng.normal(size=(k,4,dim)))
    class_dir=norm(0.9*family[np.arange(k)//2]+1.25*subtype)
    sizes=None
    if cfg.get('imbalanced'):
        proportions=np.geomspace(1,.006,k); proportions=proportions/proportions.sum()
        sizes=np.maximum(3,np.floor(n*proportions)).astype(int)
        sizes[0] += n-sizes.sum()
        counts=sizes
    else: counts=rng.multinomial(n,np.ones(k)/k)
    y=np.repeat(np.arange(k,dtype=np.int16),counts);rng.shuffle(y)
    labels_lex={}; phrase_types={}
    for j in range(k):
        for phrase in CLASSES[j][2]:
            labels_lex[phrase]=norm(1.05*family[j//2]+1.75*subtype[j]+.05*rng.normal(size=dim))
            phrase_types[phrase]='specific'
    for f in range(nf):
        for phrase in FAMILY_LABELS[FAMILIES[f]]:
            labels_lex[phrase]=norm(1.3*family[f]+.05*rng.normal(size=dim))
            phrase_types[phrase]='family'
    for phrase in GENERIC_LABELS:
        labels_lex[phrase]=norm(.6*common+.3*rng.normal(size=dim))
        phrase_types[phrase]='global'

    rows=[]; details=[]
    x=np.zeros((n,dim),np.float32)
    w=np.zeros((n,3,dim),np.float32)
    mask=np.zeros((n,3),bool)
    episode=rng.integers(0,2,size=n)
    # Per-scenario shared spuriously stable visual style can confuse EAC.
    nuisance_style=rng.integers(0,len(STYLES),size=n)
    for i,ki in enumerate(y):
        ki=int(ki); fi=ki//2; geom=cfg['geometry']; mode=int(rng.integers(4));
        near_boundary=False; shared_duplicate=False
        c=cfg['input_strength']*class_dir[ki]+.67*family[fi]
        if geom=='separable':
            base=c+1.1*subtype[ki]; noise=.25
        elif geom=='overlap':
            base=.53*family[fi]+.25*subtype[ki]+.08*c;noise=1.08
        elif geom=='submodes':
            # Multimodal local components, including cross-class mode directions.
            base=.6*c+1.5*mode_vecs[ki,mode] + .28*style[nuisance_style[i]]
            noise=.36
        elif geom=='moons':
            theta=float(rng.uniform(0,np.pi))
            u,v=np.cos(theta),np.sin(theta)
            if ki%2: u,v=1.0-u,-v-.35
            base=.8*family[fi]+2.2*(u*manifold[0]+v*manifold[1])+.25*subtype[ki]
            noise=.18
        elif geom=='bridges':
            near_boundary=bool(rng.random()<.30)
            other=ki^1
            t=float(rng.uniform(.36,.64)) if near_boundary else 1.0
            base=t*(.65*family[fi]+.42*subtype[ki])+(1-t)*(.65*family[fi]+.42*subtype[other])
            noise=1.02 if near_boundary else .83
        elif geom=='heteroscedastic':
            base=.57*family[fi]+.36*subtype[ki]+ (.22+.95*float(ki%4==0))*style[nuisance_style[i]]
            noise=.48+1.0*((ki%4)/3)
        elif geom=='duplicate':
            shared_duplicate=bool(rng.random()<.24)
            base=(.63*family[fi]+.02*subtype[ki]) if shared_duplicate else (.60*family[fi]+.35*subtype[ki])
            noise=.65 if shared_duplicate else .93
        elif geom=='anisotropic':
            base=1.5*common+.72*c+1.4*style[nuisance_style[i]]
            noise=.95
        elif geom=='spurious':
            base=.55*c+2.3*style[nuisance_style[i]]+.3*subtype[ki]
            noise=.54
        else: raise ValueError(geom)
        x[i]=norm(base+noise*rng.normal(size=dim)/np.sqrt(dim)).astype(np.float32)
        # Similar content must not be secretly made distinguishable by IDs.
        desc=CLASSES[ki][3][int(rng.integers(len(CLASSES[ki][3])))]
        if near_boundary: desc+=' Several activities occur in the same scene.'
        if shared_duplicate: desc='The scene is visually ambiguous and has similar objects across categories. '+desc
        if geom in ('anisotropic','spurious'): desc+=' Recorded in '+STYLES[nuisance_style[i]]+'.'
        if geom=='moons': desc+=' A sequence of transitions connects nearby activity scenes.'
        if geom=='submodes': desc+=' Seen in '+str(('close-up','wide shot','moving scene','demonstration')[mode])+'.'
        names=[];types=[]
        shared_false = bool(rng.random()<.52) and cfg['label_mode']=='wrong_majority'
        for m,source in enumerate(LLMS):
            lm=cfg['label_mode']; typ='specific';
            p_correct=.77; p_family=.14; miss=.035
            if lm=='perfect': p_correct=1.;p_family=0;miss=0
            elif lm=='random': p_correct=0;p_family=0;miss=0
            elif lm=='all_generic': p_correct=0;p_family=0;miss=0
            elif lm=='correlated_all_wrong':p_correct=0;p_family=0;miss=0
            elif lm=='good':p_correct=.91;p_family=.07;miss=.015
            elif lm=='mixed':p_correct=.68;p_family=.22;miss=.04
            elif lm=='hubs':p_correct=.16;p_family=.12;miss=.03
            elif lm=='specialists':
                p_correct=.93 if m==ki%3 else .30
                p_family=.04 if m==ki%3 else .48;miss=.02
            elif lm=='wrong_majority':
                p_correct=.91 if m==2 else .82
                p_family=.05;miss=.01
            elif lm=='boundary_help':
                p_correct=.96 if near_boundary else .20
                p_family=.02 if near_boundary else .68
                miss=.005 if near_boundary else .055
            elif lm=='mnar':
                p_correct=.78;p_family=.17
                miss=.82 if ki in (k-2,k-1) else (.50 if ki%2 else .02)
            elif lm=='domain_shift':
                if int(episode[i])==0:
                    p_correct=.95 if m in (0,1) else .20
                else:
                    p_correct=.95 if m==2 else .24
                p_family=.04 if p_correct>.9 else .43
                miss=.03
            elif lm=='hierarchy':
                p_correct=.92 if m==1 else (.2 if m==0 else .60)
                p_family=.76 if m==0 else (.06 if m==1 else .25)
                miss=.015
            if lm=='all_generic':
                label=str(rng.choice(GENERIC_LABELS));typ='global'
            elif lm=='random':
                label=str(rng.choice(CLASSES[int(rng.integers(k))][2]));typ='random'
            elif lm=='correlated_all_wrong':
                label=str(rng.choice(CLASSES[(ki^1)][2]));typ='wrong'
            elif lm=='wrong_majority' and shared_false and m<2:
                label=str(rng.choice(CLASSES[ki^1][2]));typ='correlated_wrong'
            elif rng.random()<miss:
                label='';typ='missing'
            else:
                u=rng.random()
                if u < p_correct:
                    label=str(rng.choice(CLASSES[ki][2]));typ='specific'
                elif u < p_correct+p_family:
                    label=str(rng.choice(FAMILY_LABELS[FAMILIES[fi]]));typ='family'
                elif lm=='hubs' and rng.random()<.78:
                    label=str(rng.choice(GENERIC_LABELS[:3]));typ='global'
                else:
                    wrong=int(rng.choice([j for j in range(k) if j!=ki]))
                    label=str(rng.choice(CLASSES[wrong][2]));typ='wrong'
            if label:
                original=label
                label=generated_phrase(original,rng,variation=True)
                if label not in labels_lex:
                    labels_lex[label]=norm(labels_lex[original]+.018*rng.normal(size=dim))
                w[i,m]=norm(labels_lex[label]+cfg['label_noise']*rng.normal(size=dim)/np.sqrt(dim)).astype(np.float32)
                mask[i,m]=True
            names.append(label);types.append(typ)
        row={'video_id':f"{cfg['name']}_{i:06d}",'category':CLASSES[ki][0],
             'desc_smolvlm':desc,'sug_gemini':names[0],
             'sug_claude':names[1],'sug_llama':names[2]}
        rows.append(row)
        details.append({'video_id':row['video_id'],'true_class_index':ki,'family':FAMILIES[fi],
                        'submode':mode,'boundary':near_boundary,'duplicate_ambiguous':shared_duplicate,
                        'domain':int(episode[i]),'input_style':STYLES[nuisance_style[i]],
                        'label_type_gemini':types[0],'label_type_claude':types[1],'label_type_llama':types[2]})
    return rows,details,x,w,mask,y

def save_dataset(base,cfg,seed,dim):
    base.mkdir(parents=True,exist_ok=True)
    rows,details,X,L,M,y=generate(cfg,seed,dim)
    for fn,rr in [('dataset.csv',rows),('diagnostics_DO_NOT_USE_FOR_TRAINING.csv',details)]:
        with (base/fn).open('w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=list(rr[0]));writer.writeheader();writer.writerows(rr)
    np.savez_compressed(base/'synthetic_embeddings.npz',X=X,labels=L,label_mask=M,y=y,
                       ids=np.array([r['video_id'] for r in rows]),source_names=np.array(LLMS),
                       class_names=np.array([c[0] for c in CLASSES[:cfg['k']]]))
    counts={CLASSES[t][0]:int((y==t).sum()) for t in range(cfg['k'])}
    info={**cfg,'seed':seed,'embedding_dim':dim,'generated_rows':len(y),'class_counts':counts,
          'missing_fraction':float(1-M.mean()),'source_kind':'synthetic controlled co-embeddings; not Qwen'}
    (base/'scenario.json').write_text(json.dumps(info,indent=2)+'\n')
    return {'name':cfg['name'],'tier':cfg['tier'],'n':len(y),'k':cfg['k'],
            'dim':dim,'geometry':cfg['geometry'],'label_mode':cfg['label_mode'],
            'missing_fraction':float(1-M.mean())}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',default='.')
    p.add_argument('--seed',type=int,default=20261107)
    p.add_argument('--dimension',type=int,default=96)
    p.add_argument('--scenario',default='all',help='all, control, hard, or a specific scenario')
    a=p.parse_args()
    root=Path(a.output)
    (root/'data').mkdir(parents=True,exist_ok=True);(root/'configs').mkdir(exist_ok=True)
    mapping={'id_column':'video_id','text_columns':['desc_smolvlm'],
             'label_columns':{'gemini':'sug_gemini','claude':'sug_claude','llama':'sug_llama'},
             'target_column':'category','delimiter':',','encoding':'utf-8','text_joiner':'\n'}
    (root/'configs'/'column_mapping.json').write_text(json.dumps(mapping,indent=2)+'\n')
    manifest=[]
    for cfg in SCENARIOS:
        if a.scenario not in ('all',cfg['tier'],cfg['name']): continue
        item=save_dataset(root/'data'/cfg['name'],cfg,a.seed,a.dimension)
        manifest.append(item)
        print(f"{cfg['name']:39s} n={item['n']:4d}  K={item['k']:2d}  missing={item['missing_fraction']:.1%}",flush=True)
    (root/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(f'Generated {len(manifest)} scenarios / {sum(x["n"] for x in manifest):,} instances at {root}',flush=True)

if __name__=='__main__':main()
