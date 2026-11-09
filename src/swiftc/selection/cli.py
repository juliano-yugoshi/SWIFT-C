#!/usr/bin/env python3
"""Select real-world SWIFT-C comparisons from completed synthetic results.

Runs independently from weaklabel-eac; needs pandas/numpy only.
Ground truth is used here exclusively to select evaluation configurations on
synthetic datasets. It is not used in real-world model fitting.
"""
from __future__ import annotations

import argparse
import json
import math
import shlex
import stat
from pathlib import Path
import numpy as np
import pandas as pd

MANDATORY = ('input', 'fusion', 'eac_uniform', 'eac_adaptive')
AVAILABLE = (
    'input', 'labels_only', 'fusion', 'eac_uniform', 'eac_adaptive',
    'spherical_input', 'spherical_fusion', 'spectral_input',
    'eac_classic', 'eac_multiview', 'spectral_fusion',
    'hdbscan_fusion', 'dec_input', 'dec_fusion', 'eac_average_linkage',
)
FAMILY = {
    'input':'kmeans_input', 'labels_only':'labels_only', 'fusion':'fusion',
    'eac_uniform':'eac_adaptive_family', 'eac_adaptive':'eac_adaptive_family',
    'spherical_input':'spherical', 'spherical_fusion':'spherical',
    'spectral_input':'spectral', 'spectral_fusion':'spectral',
    'eac_classic':'eac_classic', 'eac_multiview':'eac_multiview',
    'hdbscan_fusion':'density', 'dec_input':'deep', 'dec_fusion':'deep',
    'eac_average_linkage':'eac_classic',
}
RUN_KEYS = ('scenario', 'generation_seed', 'source_set', 'k_factor', 'seed')


def _number(x, default=0.0):
    try:
        v=float(x)
        return v if math.isfinite(v) else default
    except (ValueError, TypeError):
        return default


def parse_args(argv=None):
    p=argparse.ArgumentParser(description='Rank synthetic SWIFT-C results and generate focused real-world scripts.')
    p.add_argument('--results', required=True, help='Directory containing runs.csv, or path to runs.csv')
    p.add_argument('--output', default='selection', help='Directory for ranking and generated Bash scripts')
    p.add_argument('--top-n', type=int, default=2, help='Additional competitors beyond four required controls')
    p.add_argument('--source', default='all', help='Synthetic source_set used to rank methods (default all)')
    p.add_argument('--k-factor', type=float, default=1.0, help='Synthetic K factor used to rank (default 1.0)')
    p.add_argument('--min-coverage', type=float, default=0.8, help='Required matched-run coverage vs fusion')
    p.add_argument('--allow-partial', action='store_true', help='Allow incomplete synthetic benchmark (reported explicitly)')
    p.add_argument('--include-sanity', action='store_true', help='Include trivial sanity scenarios in ranking')
    p.add_argument('--real-input', default='MSRVTT_dados_multillm.zip')
    p.add_argument('--mapping', default='configs/msrvtt.json')
    p.add_argument('--real-output', default='results/real_selected', help='Prefix for generated output directories')
    p.add_argument('--encoder', default='Qwen/Qwen3-Embedding-0.6B')
    p.add_argument('--cache-dir', default=None, help='Existing Qwen cache directory, e.g. results/real_paper/embedding_cache')
    p.add_argument('--device', default=None, help='cuda/cpu/mps/auto; passed to encoder')
    p.add_argument('--dec-device', default=None, help='cuda/cpu/mps for DEC if selected')
    p.add_argument('--progress', choices=('rich','plain'), default='rich')
    p.add_argument('--seeds', default='42,123,2025', help='Real-world clustering seeds')
    p.add_argument('--main-k', type=int, default=20)
    p.add_argument('--sensitivity-k', default='40,60')
    p.add_argument('--no-sensitivity', action='store_true')
    return p.parse_args(argv)


def load_runs(path: Path, include_sanity: bool, source: str, factor: float):
    file=path / 'runs.csv' if path.is_dir() else path
    if not file.exists():
        raise FileNotFoundError(f'Not found: {file}')
    df=pd.read_csv(file, low_memory=False)
    required={'scenario','method','ari','nmi','generation_seed','seed','source_set'}
    missing=required-set(df.columns)
    if missing:
        raise ValueError(f'runs.csv is missing columns: {sorted(missing)}')
    if 'k_factor' not in df: df['k_factor']=1.0
    df['k_factor']=pd.to_numeric(df['k_factor'], errors='coerce')
    df=df[(df.source_set.astype(str)==source)&np.isclose(df.k_factor,factor,atol=1e-7,rtol=0)]
    if 'tier' in df and not include_sanity:
        df=df[~df.tier.astype(str).str.lower().eq('sanity')]
    elif not include_sanity:
        df=df[~df.scenario.astype(str).str.startswith('sanity_')]
    if 'data_kind' in df:
        df=df[df.data_kind.eq('synthetic')]
    df=df[df.method.astype(str).isin(AVAILABLE)].copy()
    for col in ('ari','nmi','duration_s'):
        if col in df:df[col]=pd.to_numeric(df[col],errors='coerce')
    df=df[np.isfinite(df.ari)&np.isfinite(df.nmi)].copy()
    if df.empty:raise ValueError('No usable synthetic records for the selected source and K factor.')
    if df.duplicated(list(RUN_KEYS)+['method']).any():
        dupe=int(df.duplicated(list(RUN_KEYS)+['method']).sum())
        raise ValueError(f'{dupe} duplicate method/run keys in runs.csv. Use a clean output directory.')
    return df, file


def verify_completeness(df, file, args):
    """Inspect run_protocol.json, but never assume unobserved runs are successes."""
    notes=[]
    proto_path=file.parent/'run_protocol.json'
    if not proto_path.exists():
        notes.append('No run_protocol.json found; planned experiment completeness cannot be established.')
        return notes
    try:proto=json.loads(proto_path.read_text())
    except Exception as e:
        notes.append(f'Could not read run_protocol.json: {e}')
        return notes
    if proto.get('kind')!='synthetic':
        notes.append('Protocol is not marked as synthetic; verify input directory.')
        return notes
    scenarios=[s for s in proto.get('scenarios',[]) if args.include_sanity or not str(s).startswith('sanity_')]
    generations=proto.get('generations',[])
    seeds=proto.get('seeds',[])
    factors=[_number(x, -100) for x in proto.get('k_factors',[1])]
    if not any(abs(f-args.k_factor)<1e-7 for f in factors):
        notes.append('Requested K factor was not scheduled in run_protocol.json.')
    all_sources=proto.get('sources','all')
    if args.source != 'all' and args.source not in str(all_sources) and 'each' not in str(all_sources):
        notes.append(f'Source {args.source} may not be planned in protocol: {all_sources}')
    planned=len(scenarios)*len(generations)*len(seeds)
    observed=len(df[df.method.eq('fusion')])
    if planned and observed < planned:
        notes.append(f'Incomplete baseline: fusion has {observed}/{planned} scheduled synthetic records for source={args.source}, k_factor={args.k_factor}.')
    unknown=set(df.method)-set(proto.get('methods',[]))
    if unknown:notes.append('Methods absent from the recorded protocol: '+','.join(sorted(unknown)))
    return notes


def metrics(df, min_coverage):
    # Pair each model with the 65/35 fusion on identical run keys.
    baseline=df[df.method.eq('fusion')][list(RUN_KEYS)+['ari','nmi']].rename(columns={'ari':'base_ari','nmi':'base_nmi'})
    if baseline.empty:raise ValueError('Mandatory baseline fusion not found for the requested conditions.')
    rows=[]; paired=[]; scen=[]
    base_n=len(baseline)
    for method,g in df.groupby('method', sort=True):
        joined=g.merge(baseline, on=list(RUN_KEYS),how='inner',validate='one_to_one').copy()
        if joined.empty:continue
        joined['delta_ari']=joined.ari-joined.base_ari
        joined['delta_nmi']=joined.nmi-joined.base_nmi
        joined['method']=method
        joined['win']=(joined.delta_ari>0.001).astype(float)
        # Scenario-level macro statistics: repeated seeds/generations do not overweight a scenario.
        sc=joined.groupby('scenario',as_index=False).agg(
            ari=('ari','mean'),nmi=('nmi','mean'),delta_ari=('delta_ari','mean'),
            delta_nmi=('delta_nmi','mean'),win=('win','mean'),
            time_s=('duration_s','median') if 'duration_s' in joined else ('ari','count'))
        row={
            'method':method,'family':FAMILY.get(method,'other'),
            'mean_ari':float(sc.ari.mean()), 'mean_nmi':float(sc.nmi.mean()),
            'mean_delta_ari':float(sc.delta_ari.mean()), 'mean_delta_nmi':float(sc.delta_nmi.mean()),
            'scenario_win_rate':float((sc.delta_ari>0.001).mean()),
            'scenario_loss_rate':float((sc.delta_ari < -0.001).mean()),
            'worst_scenario_delta_ari':float(sc.delta_ari.min()),
            'p10_scenario_delta_ari':float(sc.delta_ari.quantile(.10)),
            'time_median_s':float(g.duration_s.median()) if 'duration_s' in g else float('nan'),
            'paired_runs':len(joined),'baseline_runs':base_n,
            'coverage':len(joined)/base_n,
            'n_scenarios':int(sc.scenario.nunique()),
            'n_generations':int(joined.generation_seed.nunique()),
        }
        # Conservative ranking: emphasize macro ARI and NMI; penalize worst regressions.
        row['selection_score']=(
            .60*row['mean_ari']+.25*row['mean_nmi']
            +.10*row['mean_delta_ari'] + .05*row['scenario_win_rate']
            -.10*max(0.,-row['p10_scenario_delta_ari'])
        )
        row['eligible']=bool(row['coverage']>=min_coverage)
        rows.append(row)
        paired.append(joined[list(RUN_KEYS)+['method','ari','nmi','base_ari','base_nmi','delta_ari','delta_nmi']])
        sc['method']=method
        scen.append(sc)
    result=pd.DataFrame(rows)
    if result.empty:raise ValueError('No methods could be paired to the fusion baseline.')
    # Penalize partially-covered methods in ranking to prevent cherry-picking easy datasets.
    result['selection_score']=result.selection_score * (0.75+0.25*result.coverage.clip(0,1))
    result=result.sort_values(['eligible','selection_score','mean_ari','mean_nmi'],ascending=[False,False,False,False]).reset_index(drop=True)
    result.insert(0,'rank',np.arange(1,len(result)+1))
    return result,pd.concat(paired,ignore_index=True),pd.concat(scen,ignore_index=True)


def choose_methods(ranking, top_n):
    found=set(ranking.method)
    absent=set(MANDATORY)-found
    if absent:raise ValueError(f'Mandatory control(s) missing in synthetic results: {sorted(absent)}')
    ineligible=set(ranking.loc[~ranking.eligible,'method'])&set(MANDATORY)
    if ineligible:
        raise ValueError(f'Mandatory method(s) have insufficient matched coverage: {sorted(ineligible)}')
    picked=list(MANDATORY)
    available=ranking[ranking.eligible & ~ranking.method.isin(picked)].to_dict('records')
    used=set(FAMILY[x] for x in picked)
    reasons={m:'mandatory methodological control' for m in picked}
    for _ in range(min(top_n,len(available))):
        preferred=[r for r in available if r['family'] not in used]
        options=preferred or available
        chosen=options[0]
        picked.append(chosen['method']);used.add(chosen['family'])
        reasons[chosen['method']]='best eligible competitor with complementary method family' if preferred else 'highest remaining eligible score'
        available=[r for r in available if r['method']!=chosen['method']]
    return picked,reasons


def make_command(args, methods, output, k_values, sources):
    cmd=['swiftc','realworld','--profile','full','--methods',','.join(methods),
         '--input',args.real_input,'--mapping',args.mapping,
         '--output',output,'--encoder',args.encoder,
         '--clusters',','.join(str(x) for x in k_values),'--seeds',args.seeds,
         '--sources',sources,'--progress',args.progress]
    if args.device:cmd.extend(['--device',args.device])
    if args.dec_device:cmd.extend(['--dec-device',args.dec_device])
    if args.cache_dir:cmd.extend(['--cache-dir',args.cache_dir])
    return cmd


def write_shell(path, commands, description):
    lines=['#!/usr/bin/env bash','set -euo pipefail','','# '+description,
           '# Run from the SWIFT-C project directory, where input and mapping paths resolve.',
           '# The existing embedding cache can be shared with --cache-dir.',
           'command -v swiftc >/dev/null || { echo "ERROR: swiftc is not installed" >&2; exit 127; }','']
    lines+=[' '.join(shlex.quote(str(x)) for x in cmd) for cmd in commands]
    path.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    path.chmod(path.stat().st_mode|stat.S_IXUSR|stat.S_IXGRP|stat.S_IXOTH)


def main(argv=None):
    args=parse_args(argv)
    if args.top_n<0:raise ValueError('--top-n must be non-negative')
    if not 0<args.min_coverage<=1:raise ValueError('--min-coverage must be in (0,1]')
    df,file=load_runs(Path(args.results),args.include_sanity,args.source,args.k_factor)
    notes=verify_completeness(df,file,args)
    if notes and not args.allow_partial:
        critical=[x for x in notes if 'Incomplete baseline' in x or 'Requested K factor was not scheduled' in x]
        if critical:raise ValueError('\n'.join(critical)+'\nPass --allow-partial for an exploratory selection.')
    ranking, paired, scenarios=metrics(df,args.min_coverage)
    methods,reasons=choose_methods(ranking,args.top_n)
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    ranking.to_csv(output/'method_ranking.csv',index=False)
    paired.to_csv(output/'paired_comparisons.csv',index=False)
    scenarios.to_csv(output/'scenario_performance.csv',index=False)
    config={'selected_methods':methods,'method_reasons':reasons,
            'synthetic_results':str(file.resolve()),'synthetic_source_set':args.source,
            'synthetic_k_factor':args.k_factor,'sanity_included':args.include_sanity,
            'selection_type':'exploratory' if args.allow_partial else 'synthetic development selection',
            'notes':notes,'n_synthetic_scenarios':int(df.scenario.nunique()),
            'n_synthetic_generations':int(df.generation_seed.nunique()),
            'realworld_main_k':args.main_k,'realworld_seeds':args.seeds,
            'realworld_main_sources':'all'}
    (output/'selected_methods.json').write_text(json.dumps(config,indent=2,ensure_ascii=False)+'\n')
    main_cmd=make_command(args,methods,args.real_output+'_main',[args.main_k],'all')
    write_shell(output/'run_realworld_main.sh',[main_cmd], 'Main real-world comparison: fixed K, combined weak-label sources')
    if not args.no_sensitivity:
        ks=[int(s.strip()) for s in args.sensitivity_k.split(',') if s.strip() and int(s.strip())!=args.main_k]
        cmds=[]
        if ks:cmds.append(make_command(args,['fusion','eac_adaptive'],args.real_output+'_k_sensitivity',ks,'all'))
        cmds.append(make_command(args,['input','fusion','eac_adaptive'],args.real_output+'_llm_sensitivity',[args.main_k],'each'))
        write_shell(output/'run_realworld_sensitivity.sh',cmds,'Sensitivity: K and individual LLMs, keeping central controls')
    lines=['# Synthetic-based selection for the real-world benchmark','',
           f'Source: `{file}`. Filter: source_set=`{args.source}`, k_factor={args.k_factor}.',
           f'Scenarios: {df.scenario.nunique()}; generations: {df.generation_seed.nunique()}.',
           'Ranked using mean ARI/NMI macro-averaged across scenarios, paired deltas vs fusion, '
           'scenario win rate and downside risk. Runs are paired on scenario, generation, seed, source and K-factor.',
           'Four methodological controls are always included when observed with sufficient coverage.','',
           '## Selected methods','']
    for m in methods:
        r=ranking[ranking.method.eq(m)].iloc[0]
        lines.append(f'- **{m}** — {reasons[m]}; ARI {r.mean_ari:.4f}; NMI {r.mean_nmi:.4f}; '
                     f'ΔARI {r.mean_delta_ari:+.4f}; coverage {r.coverage:.1%}.')
    lines+=['','## Important limitations',
            '- This selects a manageable *real-world comparison*, not an unbiased estimator of final performance.',
            '- Synthetic evidence may not transfer to the video dataset. ARI/NMI for MSR-VTT must not influence this selection.',
            '- Synthetic generators have been examined repeatedly; these synthetic results may be affected by method selection bias.',
            '- Pairwise matching excludes missing/unavailable method runs; see coverage and warnings.',
            '- The 65/35 fusion hyperparameter follows the fixed protocol and is not tuned here.',
            '- Different methods can have very different runtimes; inspect method_ranking.csv.',
            '- Run scripts are separate from results of an earlier full real-world benchmark, so they will not mix protocols.',
            '', '## Warnings']
    lines += ['- '+note for note in notes] if notes else ['- No protocol-completeness warnings detected.']
    lines+=['','## Run','', '```bash','bash run_realworld_main.sh','bash run_realworld_sensitivity.sh','```','',
            'The generated scripts use the installed `swiftc` executable. Adjust paths if executing from a different directory.']
    (output/'selection_report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(f'Selected {len(methods)} methods: {", ".join(methods)}')
    for n in notes:print('WARNING:',n)
    print('Output:',output.resolve())
    print('Main command:', ' '.join(shlex.quote(x) for x in main_cmd))
    return 0

if __name__=='__main__':
    try:raise SystemExit(main())
    except (ValueError,FileNotFoundError) as e:
        print(f'ERROR: {e}',file=__import__('sys').stderr)
        raise SystemExit(2)
