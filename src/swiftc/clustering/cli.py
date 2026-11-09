"""Command-line interface: all console text in English."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
from rich.console import Console
from rich.table import Table
from .data import inspect_columns,guess_mapping,DEFAULT_COLUMNS,validate_mapping
from .runner import METHODS,CORE_METHODS,QUICK_METHODS,PAPER_METHODS,experiment_synthetic,experiment_realworld
from .synthetic import choose_scenarios

console=Console()

def ints(s):return tuple(int(x) for x in s.split(',') if x.strip())
def floats(s):return tuple(float(x) for x in s.split(',') if x.strip())
def methods(s):
    vals=tuple(x.strip() for x in s.split(',') if x.strip())
    unknown=set(vals)-set(METHODS)
    if unknown:raise argparse.ArgumentTypeError(f'Unknown methods: {sorted(unknown)}')
    return vals

def common(p):
    p.add_argument('--output',required=True)
    p.add_argument('--progress',choices=['rich','plain'],default='rich')
    p.add_argument('--quiet',action='store_true')
    p.add_argument('--no-resume',action='store_true')
    p.add_argument('--dry-run',action='store_true')
    p.add_argument('--label-weight',type=float,default=.35)
    p.add_argument('--eta',type=float,default=.25)
    p.add_argument('--edge-threshold',type=float,default=.75)
    p.add_argument('--suite',choices=['core','quick','paper'],default=None,
                   help='core=5 methods; quick=10 methods; paper=14 methods including HDBSCAN and DEC')
    p.add_argument('--methods',type=methods,default=None,
                   help='Override suite; comma-separated methods. Run --help for presets.')
    p.add_argument('--knn-neighbors',type=int,default=15)
    p.add_argument('--spectral-backbone',type=float,default=.03,
                   help='Geometric backbone contribution in EAC spectral clustering')
    p.add_argument('--density-pca-dim',type=int,default=32)
    p.add_argument('--dec-latent-dim',type=int,default=32)
    p.add_argument('--dec-pretrain-epochs',type=int,default=None)
    p.add_argument('--dec-epochs',type=int,default=None)
    p.add_argument('--dec-batch-size',type=int,default=256)
    p.add_argument('--dec-device',default='cpu')

def parser():
    p=argparse.ArgumentParser(prog='swiftc',description='SWIFT-C semantic weak-label clustering experiments')
    sp=p.add_subparsers(dest='command',required=True)
    s=sp.add_parser('sanity',help='Fast offline pipeline/algorithm sanity check, using synthetic embeddings')
    s.add_argument('--output',default='results/sanity');s.add_argument('--progress',choices=['rich','plain'],default='rich')
    s.add_argument('--suite',choices=['core','quick','paper'],default='core')
    s.add_argument('--quiet',action='store_true')
    s.add_argument('--dry-run',action='store_true')
    s=sp.add_parser('synthetic',help='Run controlled synthetic evaluation')
    common(s)
    s.add_argument('--profile',choices=['quick','full'],default='quick')
    s.add_argument('--dataset-root',default=None,help='Where to cache generated scenario datasets')
    s.add_argument('--scenarios',default=None,help='all, quick, sanity, hard, or comma-separated scenario names')
    s.add_argument('--generations',default=None,help='Integer generator seeds, comma-separated')
    s.add_argument('--seeds',default=None,help='K-Means seeds')
    s.add_argument('--sources',default=None,help='all, each, all,each or named source combinations')
    s.add_argument('--k-factors',default=None,help='K relative to synthetic reference K: e.g. 0.75,1,1.25')
    s.add_argument('--sample',type=int,default=None,help='Fixed subsample per scenario; 0 means all')
    s=sp.add_parser('realworld',help='Run clustering on a mapped CSV or ZIP dataset using a text encoder')
    common(s)
    s.add_argument('--input',required=True,help='CSV or ZIP containing CSV')
    s.add_argument('--dataset-name',default=None,help='Dataset identifier used in run metadata (defaults to input filename stem)')
    s.add_argument('--mapping' ,required=True,help='JSON mapping for id, text, label, and target columns')
    s.add_argument('--profile',choices=['quick','full'],default='quick')
    s.add_argument('--encoder',default='Qwen/Qwen3-Embedding-0.6B',help='Qwen model or hashing (functional test only)')
    s.add_argument('--precomputed-embeddings',default=None,help='Validated NPZ with X, labels, label_mask, ids and source_names; bypass encoder')
    s.add_argument('--sample',type=int,default=None,help='Quick defaults to 600, full defaults to full dataset')
    s.add_argument('--sample-seed',type=int,default=42)
    s.add_argument('--clusters',default=None,help='Comma-separated K values')
    s.add_argument('--seeds',default=None,help='Comma-separated clustering seeds')
    s.add_argument('--sources',default=None,help='all, each, named sources from the JSON mapping, or combinations joined by +')
    s.add_argument('--batch-size',type=int,default=32)
    s.add_argument('--max-chars',type=int,default=4096)
    s.add_argument('--device',default=None,help='cuda, cpu, mps, or auto')
    s.add_argument('--cache-dir',default=None)
    s=sp.add_parser('inspect',help='Show columns in an input CSV/ZIP')
    s.add_argument('--input',required=True)
    s=sp.add_parser('configure',help='Interactively map arbitrary CSV columns')
    s.add_argument('--input',required=True)
    s.add_argument('--output',required=True)
    s.add_argument('--noninteractive',action='store_true',help='Save detected mapping without prompting')
    s=sp.add_parser('plot',help='Create publication PNG/SVG figures from completed evaluations')
    s.add_argument('--runs',required=True);s.add_argument('--output',required=True)
    s.add_argument('--sources',default='all');s.add_argument('--k-factor',type=float,default=1.)
    s.add_argument('--method',default='eac_adaptive')
    s=sp.add_parser('report',help='Rebuild aggregate reports and paired statistics from existing runs.csv')
    s.add_argument('--runs',required=True);s.add_argument('--output',required=True)
    return p

def setup_mapping(a):
    cols=inspect_columns(a.input)
    guess=guess_mapping(cols)
    console.print(f'Found {len(cols)} columns:')
    for i,c in enumerate(cols):console.print(f'  {i:>2}: {c}')
    if not a.noninteractive:
        def single(title,default,allow_blank=False):
            d='' if default is None else default
            answer=input(f'{title} [{d}]: ').strip() or d
            if answer=='' and allow_blank:return None
            if answer.isdigit() and 0<=int(answer)<len(cols):return cols[int(answer)]
            if answer not in cols:raise ValueError(f'Unknown column {answer}')
            return answer
        guess['id_column']=single('Unique ID column (name or number)',guess['id_column'])
        text=single('Text column (name or number)',guess['text_columns'][0] if guess['text_columns'] else None)
        guess['text_columns']=[text]
        labels=input('Weak-label columns (comma-separated names or indices) ['+','.join(guess['label_columns'].values())+']: ').strip()
        if labels:
            parsed=[]
            for item in labels.split(','):
                item=item.strip();parsed.append(cols[int(item)] if item.isdigit() else item)
            guess['label_columns']={c.removeprefix('sug_'):c for c in parsed}
        guess['target_column']=single('Reference target column (blank to disable)',guess['target_column'],allow_blank=True)
    validate_mapping(cols,guess)
    Path(a.output).parent.mkdir(parents=True,exist_ok=True)
    Path(a.output).write_text(json.dumps(guess,indent=2)+'\n')
    console.print(f'[green]Saved column mapping:[/green] {a.output}')


def chosen_methods(a):
    if getattr(a,'methods',None) is not None:return a.methods
    suite=a.suite or ('paper' if getattr(a,'profile','quick')=='full' else 'quick')
    return {'core':CORE_METHODS,'quick':QUICK_METHODS,'paper':PAPER_METHODS}[suite]


def extra_params(a):
    return dict(knn_neighbors=a.knn_neighbors,backbone_weight=a.spectral_backbone,
                density_pca_dim=a.density_pca_dim,dec_latent_dim=a.dec_latent_dim,
                dec_pretrain_epochs=a.dec_pretrain_epochs or (3 if a.profile=='quick' else 12),
                dec_epochs=a.dec_epochs or (5 if a.profile=='quick' else 25),
                dec_batch_size=a.dec_batch_size,dec_device=a.dec_device)


def main(argv=None):
    a=parser().parse_args(argv)
    if a.command=='inspect':
        console.print('\n'.join(inspect_columns(a.input)));return 0
    if a.command=='configure':setup_mapping(a);return 0
    if a.command=='plot':
        from .plots import plot_evaluations
        plot_evaluations(a.runs,a.output,source=a.sources,k_factor=a.k_factor,method=a.method)
        console.print(f'[green]Figures saved:[/green] {a.output}');return 0
    if a.command=='report':
        from .report import write_report
        write_report(a.runs,a.output)
        console.print(f'[green]Report saved:[/green] {a.output}');return 0
    if a.command=='sanity':
        experiment_synthetic(Path(a.output)/'generated',Path(a.output),
                             ['sanity_clean_separable','sanity_paraphrase_labels'],
                             seeds=(42,),generation_seeds=(20261107,),profile='quick',
                             sources='all',sample=160,methods=chosen_methods(a),
                             dec_pretrain_epochs=2,dec_epochs=3,progress=a.progress,verbose=not a.quiet,dry_run=a.dry_run)
        return 0
    if a.command=='synthetic':
        quick=a.profile=='quick'
        scenarios=choose_scenarios(a.scenarios or ('quick' if quick else 'all'))
        generations=ints(a.generations or ('20261107' if quick else '202611303,202611304,202611305'))
        seeds=ints(a.seeds or ('42' if quick else '42,123,2025'))
        factors=floats(a.k_factors or ('1'))
        sources=a.sources or ('all' if quick else 'all,each')
        output=Path(a.output)
        experiment_synthetic(a.dataset_root or (output/'generated'),output,scenarios,seeds,generations,
                             profile=a.profile,sources=sources,k_factors=factors,
                             sample=(320 if quick else 0) if a.sample is None else a.sample,
                             methods=chosen_methods(a),label_weight=a.label_weight,eac_eta=a.eta,
                             eac_threshold=a.edge_threshold,**extra_params(a),progress=a.progress,verbose=not a.quiet,
                             resume=not a.no_resume,dry_run=a.dry_run)
        return 0
    if a.command=='realworld':
        quick=a.profile=='quick'
        experiment_realworld(a.input,a.mapping,a.output,encoder=a.encoder,precomputed=a.precomputed_embeddings,profile=a.profile,
                             sample=(600 if quick else 0) if a.sample is None else a.sample,
                             sample_seed=a.sample_seed,k_values=ints(a.clusters or ('20' if quick else '20,40,60')),
                             seeds=ints(a.seeds or ('42' if quick else '42,123,2025')),
                             sources=a.sources or ('all' if quick else 'all,each'),methods=chosen_methods(a),
                             batch_size=a.batch_size,device=a.device,max_chars=a.max_chars,
                             cache_dir=a.cache_dir,label_weight=a.label_weight,eac_eta=a.eta,
                             eac_threshold=a.edge_threshold,**extra_params(a),progress=a.progress,
                             verbose=not a.quiet,resume=not a.no_resume,dry_run=a.dry_run,dataset_name=a.dataset_name)
        return 0
    return 2

if __name__=='__main__':sys.exit(main())
