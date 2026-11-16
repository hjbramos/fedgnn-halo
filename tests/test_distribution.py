"""A distribuição executa seus métodos sem dados ou caminhos externos."""
import copy
import json
from pathlib import Path
import numpy as np
import pytest
import yaml
from fedgnn_halo.cli import run, METHODS
from fedgnn_halo.data import synthetic
from fedgnn_halo.partition import assignments

BASE = yaml.safe_load((Path(__file__).resolve().parents[1] / 'examples/synthetic.yaml').read_text())

@pytest.mark.parametrize('method', METHODS)
def test_methods_write_finite_metrics(tmp_path, method):
    cfg = copy.deepcopy(BASE)
    cfg.update(method=method, output=str(tmp_path / method))
    cfg['training'].update(max_epochs=1, max_rounds=1)
    result = run(cfg)
    assert np.isfinite(result['test']['MAE'])
    assert (tmp_path / method / 'environment.json').exists()
    assert json.loads((tmp_path / method / 'summary.json').read_text())['test']['MAE'] == result['test']['MAE']
    with pytest.raises(ValueError, match='não está vazio'):
        run(cfg)

@pytest.mark.parametrize('architecture', ['gru', 'lstm', 'dcrnn'])
def test_alternative_architecture(tmp_path, architecture):
    cfg = copy.deepcopy(BASE)
    cfg.update(method='fedavg', output=str(tmp_path / architecture))
    cfg['training']['max_rounds'] = 1
    cfg['model'].update(name=architecture, num_layers=1, max_diffusion_step=1, cl_decay_steps=100)
    assert np.isfinite(run(cfg)['test']['MAE'])

@pytest.mark.parametrize('method', ['contiguous', 'random', 'leiden'])
def test_partitions_are_complete_and_deterministic(method):
    d = synthetic()
    opts = {'method': method, 'clients': 3, 'resolution': 1.0}
    a = assignments(d.adj, opts, 7)
    assert a.shape == (d.nodes,)
    np.testing.assert_array_equal(a, assignments(d.adj, opts, 7))
    assert set(a) == set(range(len(set(a))))


def test_synthetic_targets_do_not_overlap():
    d = synthetic()
    targets = [set((o[:, None] + np.arange(d.horizon)).ravel()) for o in d.origins.values()]
    assert targets[0].isdisjoint(targets[1]) and targets[1].isdisjoint(targets[2])
