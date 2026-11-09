"""Mapped CSV/ZIP loading and encoder caching, with no ground-truth access during fitting."""
from __future__ import annotations
import csv, hashlib, io, json, zipfile
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.preprocessing import normalize

DEFAULT_COLUMNS={'id_column':'video_id','text_columns':['desc_smolvlm'],
                 'label_columns':{'gemini':'sug_gemini','claude':'sug_claude','llama':'sug_llama'},
                 'target_column':'category','delimiter':',','encoding':'utf-8-sig','text_joiner':'\n'}

def _open_csv(path:Path, encoding='utf-8-sig'):
    if str(path).lower().endswith('.zip'):
        z=zipfile.ZipFile(path)
        names=[name for name in z.namelist() if name.lower().endswith('.csv') and not name.startswith('__MACOSX/')]
        if not names: z.close(); raise ValueError(f'No CSV in {path}')
        data=z.open(names[0]);stream=io.TextIOWrapper(data,encoding=encoding,errors='replace')
        return z,stream
    return None,open(path,encoding=encoding,errors='replace',newline='')

def inspect_columns(path, encoding='utf-8-sig',delimiter=','):
    z,f=_open_csv(Path(path),encoding)
    try: return next(csv.reader(f,delimiter=delimiter))
    finally:
        f.close()
        if z:z.close()

def read_csv(path,mapping):
    z,f=_open_csv(Path(path),mapping.get('encoding','utf-8-sig'))
    try:
        # Pandas' Python parser is unnecessary for multiline/long fields;
        # increase field size for CSVs that contain exceptionally long descriptions.
        csv.field_size_limit(2**30)
        return pd.read_csv(f,sep=mapping.get('delimiter',','),low_memory=False,dtype=str,keep_default_na=False)
    finally:
        f.close()
        if z:z.close()

def guess_mapping(columns):
    cols=list(columns);low={x.casefold():x for x in cols}
    def choose(*names):
        for name in names:
            if name.casefold() in low:return low[name.casefold()]
        return None
    idcol=choose('video_id','id','document_id','item_id') or cols[0]
    text=choose('desc_smolvlm','description','text','content','document','title')
    target=choose('category','gold_class','ground_truth','target','y_true','class')
    labels={}
    for col in cols:
        if col.casefold().startswith('sug_'):
            labels[col[4:]]=col
    return {**DEFAULT_COLUMNS,'id_column':idcol,'text_columns':[text] if text else [],'target_column':target,'label_columns':labels}

def validate_mapping(columns,mapping):
    required=[mapping['id_column']]+list(mapping['text_columns'])+list(mapping['label_columns'].values())
    if mapping.get('target_column'):required.append(mapping['target_column'])
    missing=[s for s in required if s not in columns]
    if missing:raise ValueError(f'Unknown CSV columns: {missing}; available: {columns}')
    if not mapping['text_columns'] or not mapping['label_columns']:
        raise ValueError('At least one text column and one weak-label column are required')

@dataclass
class TextDataset:
    ids:np.ndarray
    texts:list[str]
    weak_texts:np.ndarray
    source_names:list[str]
    target:np.ndarray|None
    source_rows:np.ndarray


def extract_texts(df,mapping,max_chars=4096,sample=0,sample_seed=42):
    validate_mapping(df.columns,mapping)
    if sample and len(df)>sample:
        indices=np.sort(np.random.default_rng(sample_seed).choice(len(df),size=sample,replace=False))
        data=df.iloc[indices].copy()
    else:
        indices=np.arange(len(df));data=df.copy()
    names=list(mapping['label_columns'])
    joiner=mapping.get('text_joiner','\n')
    descriptions=[joiner.join(str(row[x]).strip() for x in mapping['text_columns'] if str(row[x]).strip())[:max_chars]
                  for _,row in data.iterrows()]
    labels=data[[mapping['label_columns'][name] for name in names]].to_numpy(dtype=str)
    labels=np.vectorize(lambda s:str(s).strip()[:max_chars],otypes=[str])(labels)
    targets=data[mapping['target_column']].to_numpy(dtype=str) if mapping.get('target_column') else None
    ids=data[mapping['id_column']].to_numpy(dtype=str)
    if len(set(ids))!=len(ids):raise ValueError('The configured ID column must contain unique IDs')
    return TextDataset(ids,descriptions,labels,names,targets,indices)

def _cache_key(ds,encoder,max_chars):
    h=hashlib.sha256()
    h.update(json.dumps({'encoder':encoder,'max_chars':max_chars,'sources':ds.source_names},sort_keys=True).encode())
    for txt in ds.texts:h.update(txt.encode('utf-8'));h.update(b'\0')
    for val in ds.weak_texts.ravel():h.update(str(val).encode('utf-8'));h.update(b'\0')
    return h.hexdigest()[:22]

def embed_texts(ds,encoder='Qwen/Qwen3-Embedding-0.6B',cache_dir='.cache/swiftc',batch_size=32,device=None,max_chars=4096,
                on_progress=None):
    """Encode description and each label using exactly the same encoder.

    `hashing` is a PIPELINE TEST ONLY, not a semantic Qwen substitute.
    """
    cache_dir=Path(cache_dir);cache_dir.mkdir(parents=True,exist_ok=True)
    key=_cache_key(ds,encoder,max_chars)
    cache=cache_dir/f'embeddings_{key}.npz'
    if cache.exists():
        with np.load(cache) as d:
            X,L,M=d['X'],d['labels'],d['label_mask'].astype(bool)
        if on_progress:on_progress('Embedding cache hit',len(ds.texts),len(ds.texts))
        return X,L,M,cache,True
    all_texts=ds.texts+list(ds.weak_texts.ravel())
    unique=list(dict.fromkeys(t for t in all_texts if t.strip()))
    lookup={t:i for i,t in enumerate(unique)}
    if not unique: raise ValueError('The dataset has no nonempty texts')
    if encoder=='hashing':
        vectorizer=HashingVectorizer(n_features=1024,alternate_sign=False,norm='l2',analyzer='char_wb',ngram_range=(3,5),dtype=np.float32)
        vectors=vectorizer.transform(unique).toarray().astype(np.float32)
        if on_progress:on_progress('Hashing embeddings (pipeline-only)',len(unique),len(unique))
    else:
        try:from sentence_transformers import SentenceTransformer
        except ImportError as ex:raise RuntimeError('Install the Qwen extra: pip install -e ".[qwen]"') from ex
        model=SentenceTransformer(encoder,device=None if device in (None,"auto") else device,trust_remote_code=True)
        # Control the max token length to avoid outlier descriptions consuming huge VRAM.
        model.max_seq_length=min(int(getattr(model,'max_seq_length',512) or 512),512)
        vectors=[]
        for start in range(0,len(unique),batch_size):
            enc=model.encode(unique[start:start+batch_size],batch_size=batch_size,
                             convert_to_numpy=True,show_progress_bar=False,normalize_embeddings=True)
            vectors.append(np.asarray(enc,dtype=np.float32))
            if on_progress:on_progress('Encoding text',min(len(unique),start+batch_size),len(unique))
        vectors=np.vstack(vectors)
        del model
    d=vectors.shape[1]
    X=np.zeros((len(ds.texts),d),dtype=np.float32)
    L=np.zeros((len(ds.texts),ds.weak_texts.shape[1],d),dtype=np.float32)
    M=np.zeros(ds.weak_texts.shape,dtype=bool)
    for i,text in enumerate(ds.texts):
        if text.strip():X[i]=vectors[lookup[text]]
    for i in range(L.shape[0]):
        for j in range(L.shape[1]):
            text=str(ds.weak_texts[i,j])
            if text.strip():L[i,j]=vectors[lookup[text]];M[i,j]=True
    tmp=cache.with_suffix('.tmp.npz')
    np.savez_compressed(tmp,X=X,labels=L,label_mask=M,ids=ds.ids,source_names=np.asarray(ds.source_names))
    tmp.replace(cache)
    return X,L,M,cache,False
