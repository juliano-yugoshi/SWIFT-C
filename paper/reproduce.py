#!/usr/bin/env python3
"""Optional article reproduction driver using the public SWIFT-C CLI.

This file only orchestrates existing commands. It does not implement clustering,
weak-label fusion, or information retrieval algorithms.
"""
from __future__ import annotations
import argparse
import datetime
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = json.loads((ROOT / 'paper/protocol.json').read_text())
STEPS = ('synthetic', 'selection', 'realworld', 'discrete', 'retrieval', 'tables')

def parser():
    p = argparse.ArgumentParser(description='Reproduce SWIFT-C article experiments using the installed package')
    p.add_argument('--stage', choices=STEPS + ('all',), required=True,
                   help='Stage to run. all runs synthetic, realworld, discrete, retrieval, tables (selection is exploratory).')
    p.add_argument('--workspace', type=Path, default=ROOT / 'results/paper')
    p.add_argument('--input', type=Path, default=ROOT / PROTOCOL['dataset'])
    p.add_argument('--mapping', type=Path, default=ROOT / PROTOCOL['mapping'])
    p.add_argument('--encoder', default=PROTOCOL['realworld']['encoder'])
    p.add_argument('--device', default=None)
    p.add_argument('--dec-device', default='cpu')
    p.add_argument('--progress', choices=('rich', 'plain'), default='rich')
    p.add_argument('--dry-run', action='store_true', help='Print full commands but do not execute')
    return p

def command_for(stage, a):
    s = PROTOCOL['synthetic']
    r = PROTOCOL['realworld']
    w = a.workspace
    base = ['swiftc']
    if stage == 'synthetic':
        return base + [
            'synthetic', '--profile', s['profile'], '--suite', 'paper',
            '--output', str(w/'synthetic'),
            '--scenarios', ','.join(s['scenarios']),
            '--generations', ','.join(map(str,s['generation_seeds'])),
            '--seeds', ','.join(map(str,s['clustering_seeds'])),
            '--sources', s['sources'], '--k-factors', ','.join(map(str,s['k_factors'])),
            '--progress', a.progress, '--dec-device', a.dec_device,
        ]
    if stage == 'selection':
        return base + [
            'select', '--results', str(w/'synthetic'), '--output', str(w/'selection'),
            '--top-n', '2', '--no-sensitivity', '--real-input', str(a.input),
            '--mapping', str(a.mapping), '--encoder', a.encoder,
            '--real-output', str(w/'selected'), '--progress', a.progress,
        ]
    if stage == 'realworld':
        cmd = base + [
            'realworld', '--input', str(a.input), '--mapping', str(a.mapping),
                '--dataset-name', 'MSRVTT',
            '--output', str(w/'realworld'), '--profile', r['profile'],
            '--methods', ','.join(r['methods']), '--encoder', a.encoder,
            '--clusters', ','.join(map(str,r['k'])),
            '--seeds', ','.join(map(str,r['clustering_seeds'])),
            '--sources', r['sources'], '--progress', a.progress,
            '--dec-device', a.dec_device,
        ]
        if a.device: cmd += ['--device', a.device]
        return cmd
    if stage == 'discrete':
        return base + [
            'discrete', '--input', str(a.input), '--output', str(w/'discrete'),
            '--embeddings', str(w/'realworld/embedding_cache'),
            '--k', str(PROTOCOL['discrete']['k']),
            '--seeds', ','.join(map(str,PROTOCOL['discrete']['seeds'])),
        ]
    if stage == 'retrieval':
        return base + [
            'retrieval', '--results', str(w/'realworld'), '--discrete', str(w/'discrete'),
            '--input', str(a.input), '--embeddings', str(w/'realworld/embedding_cache'),
            '--output', str(w/'retrieval'),
            '--source-sets', r['sources'], '--k', ','.join(map(str,r['k'])),
            '--topk', ','.join(map(str,PROTOCOL['retrieval']['topk'])),
            '--strict', '--no-plots',
        ]
    if stage == 'tables':
        return [sys.executable, str(ROOT/'paper/aggregate.py'), '--workspace', str(w)]
    raise ValueError(stage)

def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as fh:
        for part in iter(lambda: fh.read(8 * 1024 * 1024), b''):
            digest.update(part)
    return digest.hexdigest()


def provenance(a):
    packages = ('swiftc-clustering','numpy','scipy','scikit-learn','pandas',
                'torch','transformers','sentence-transformers','matplotlib')
    versions = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return {
        'protocol_sha256': sha256_file(ROOT/'paper/protocol.json'),
        'dataset_sha256': sha256_file(a.input) if a.input.is_file() else None,
        'mapping_sha256': sha256_file(a.mapping) if a.mapping.is_file() else None,
        'encoder': a.encoder,
        'python': platform.python_version(),
        'platform': platform.platform(),
        'package_versions': versions,
        'stages': [],
    }


def write_manifest(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n')
    temporary.replace(path)


def main(argv=None):
    a = parser().parse_args(argv)
    steps = ('synthetic', 'realworld', 'discrete', 'retrieval', 'tables') if a.stage == 'all' else (a.stage,)
    a.workspace = a.workspace.expanduser().resolve()
    a.input = a.input.expanduser().resolve()
    a.mapping = a.mapping.expanduser().resolve()

    manifest_path = a.workspace / 'reproduction_manifest.json'
    metadata = None
    if not a.dry_run:
        current = provenance(a)
        if manifest_path.exists():
            metadata = json.loads(manifest_path.read_text())
            for field in ('protocol_sha256', 'dataset_sha256', 'mapping_sha256', 'encoder'):
                if metadata.get(field) != current.get(field):
                    raise SystemExit(
                        f'Workspace provenance changed for {field}. Use a new workspace.'
                    )
        else:
            metadata = current
            write_manifest(manifest_path, metadata)

    for stage in steps:
        cmd = command_for(stage, a)
        print(f'STAGE {stage}', flush=True)
        print('COMMAND ' + ' '.join(__import__('shlex').quote(str(x)) for x in cmd), flush=True)
        if a.dry_run:
            continue
        if stage in ('realworld', 'discrete', 'retrieval'):
            if not a.input.exists() or not a.mapping.exists():
                raise SystemExit('Dataset or column mapping not found')
        entry = {
            'stage': stage,
            'command': [str(x) for x in cmd],
            'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'status': 'running',
        }
        metadata['stages'].append(entry)
        write_manifest(manifest_path, metadata)
        try:
            subprocess.run(cmd, check=True, cwd=ROOT)
        except subprocess.CalledProcessError as error:
            entry['status'] = 'failed'
            entry['exit_code'] = error.returncode
            entry['finished_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
            write_manifest(manifest_path, metadata)
            raise
        entry['status'] = 'completed'
        entry['exit_code'] = 0
        entry['finished_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        write_manifest(manifest_path, metadata)
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
