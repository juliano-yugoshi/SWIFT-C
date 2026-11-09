"""Tests for integration and article orchestration, not algorithm reimplementations."""
from pathlib import Path
import json
import subprocess
import sys

from swiftc.cli import main
from swiftc.registry import METHOD_FAMILIES
from swiftc.clustering.runner import PAPER_METHODS
from swiftc.clustering.synthetic import SCENARIOS

ROOT = Path(__file__).resolve().parents[1]


def test_methods_cover_all_paper_algorithms(capsys):
    listed = {mid for group in METHOD_FAMILIES.values() for mid, _ in group}
    assert set(PAPER_METHODS) <= listed
    assert main(['methods']) == 0
    text = capsys.readouterr().out
    assert 'eac_adaptive' in text and 'Representation-based' in text


def test_paper_protocol_separates_publication_from_sanity():
    protocol = json.loads((ROOT/'paper/protocol.json').read_text())
    expected = {n for n, (_, cfg) in SCENARIOS.items() if cfg['tier'] in ('control','hard')}
    assert set(protocol['synthetic']['scenarios']) == expected
    assert len(expected) == 29
    assert len(protocol['synthetic']['methods']) == 14
    assert protocol['retrieval']['topk'] == [1,3,5,10]


def test_paper_script_is_dry_run_without_outputs(tmp_path):
    script = ROOT/'paper/reproduce.py'
    result = subprocess.run(
        [sys.executable, str(script), '--stage', 'all', '--dry-run', '--workspace', str(tmp_path)],
        cwd=ROOT, capture_output=True, text=True, check=True
    )
    assert 'swiftc synthetic' in result.stdout
    assert 'swiftc realworld' in result.stdout
    assert 'swiftc discrete' in result.stdout
    assert 'swiftc retrieval' in result.stdout
    assert 'paper/aggregate.py' in result.stdout
    assert not (tmp_path/'synthetic').exists()


def test_discrete_allows_custom_label_columns_but_rejects_leaky_columns(tmp_path):
    import pandas as pd
    from swiftc.discrete.core import open_real_data
    data = pd.DataFrame({
        'item_id':['a','b'], 'description':['one','two'],
        'suggestion_a':['topic','genre'], 'reference_category':['x','y'],
        'cat_model_a':['x','y']
    })
    file = tmp_path/'custom.csv'
    data.to_csv(file,index=False)
    frame = open_real_data(file, ['suggestion_a'], 'item_id', 'reference_category')
    assert frame.columns.tolist() == ['item_id','suggestion_a','reference_category']
    import pytest
    with pytest.raises(ValueError):
        open_real_data(file, ['cat_model_a'], 'item_id', 'reference_category')
    with pytest.raises(ValueError):
        open_real_data(file, ['reference_category'], 'item_id', 'reference_category')
