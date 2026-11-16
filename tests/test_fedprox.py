"""Regressões CPU/sintéticas, sem carregar datasets ou executar campanhas."""
import copy
import io
import json
import tempfile
import unittest
from contextlib import ExitStack, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn
from torch.utils.data import DataLoader

from experiments import fedprox, federated, run_grid
from experiments.train import train_epoch


class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.tensor(0., dtype=torch.float64))

    def forward(self, x, mask):
        return self.w * x


def batch(observed=True):
    return dict(x=torch.ones(2, 1, 1, 1, dtype=torch.float64),
                x_mask=torch.ones(2, 1, 1, 1, dtype=torch.bool),
                y=torch.tensor([10., 999.], dtype=torch.float64).reshape(2, 1, 1, 1),
                y_mask=torch.tensor([observed, False]).reshape(2, 1, 1, 1))


class ProximalEpochTests(unittest.TestCase):
    def test_analytic_update_and_fixed_anchor(self):
        for mu, expected in [(0., .2), (2., .18)]:
            with self.subTest(mu=mu):
                model = Toy()
                anchors = [p.detach().clone() for p in model.parameters()]
                optimizer = torch.optim.SGD(model.parameters(), lr=.1)
                _, seen = fedprox.train_epoch_proximal(model, anchors, mu, [batch(), batch()], optimizer, 'cpu', 100.)
                self.assertAlmostEqual(model.w.item(), expected)
                self.assertEqual(seen, 2)
                self.assertEqual(anchors[0].item(), 0.)
                self.assertFalse(anchors[0].requires_grad)

    def test_zero_mu_matches_fedavg_with_adam_and_masked_batches(self):
        model = Toy()
        other = copy.deepcopy(model)
        batches = [batch(), batch(False), batch(), batch()]
        a = train_epoch(model, batches, torch.optim.Adam(model.parameters(), lr=.01), 'cpu', 5.)
        b = fedprox.train_epoch_proximal(other, [other.w.detach().clone()], 0., batches,
                                       torch.optim.Adam(other.parameters(), lr=.01), 'cpu', 5.)
        self.assertEqual(a, b)
        self.assertTrue(torch.equal(model.w, other.w))

    def test_empty_epoch_fails_without_updating(self):
        for batches in [[], [batch(False)]]:
            model = Toy()
            with self.assertRaisesRegex(ValueError, 'sem alvos observados'):
                fedprox.train_epoch_proximal(model, [model.w.detach().clone()], .1, batches,
                                            torch.optim.Adam(model.parameters()), 'cpu', 5.)
            self.assertEqual(model.w.item(), 0.)

    def test_invalid_mu_rejected_at_all_entry_points(self):
        for mu in [-.1, float('nan'), float('inf'), -float('inf')]:
            with self.subTest(mu=mu):
                with self.assertRaisesRegex(ValueError, 'finito'):
                    fedprox.mu_token(mu)
                model = Toy()
                with self.assertRaisesRegex(ValueError, 'finito'):
                    fedprox.train_epoch_proximal(model, [], mu, [], None, 'cpu', 5.)
                with patch.object(fedprox, 'load_experiment') as load:
                    with self.assertRaisesRegex(ValueError, 'finito'):
                        fedprox.train_fedprox('synthetic', 1, mu, 42,
                                             {'fedprox': {}, 'windows': {'synthetic': {}}}, 'cpu')
                    load.assert_not_called()

    def test_tokens_preserve_grid_and_distinguish_adjacent_values(self):
        for mu, expected in [(0., 'mu0'), (-0., 'mu0'), (.001, 'mu0p001'), (.01, 'mu0p01'), (.1, 'mu0p1')]:
            self.assertEqual(fedprox.mu_token(mu), expected)
        values = [.12345671, .12345672, .123457, 1., torch.nextafter(
            torch.tensor(1., dtype=torch.float64), torch.tensor(2., dtype=torch.float64)).item()]
        self.assertEqual(len(set(map(fedprox.mu_token, values))), len(values))


class FederatedIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.pdir = self.root / 'partition'
        self.pdir.mkdir()
        (self.pdir / 'metadata.json').write_text('{"num_clients": 1}')
        self.data = SimpleNamespace(nodes=1, input_steps=1, horizon=1,
                                    feature_names=['signal'], target_feature_index=0)
        self.cfg = {'method': 'fedprox', 'output': 'runs',
                    'windows': {'synthetic': {'input_steps': 1, 'horizon': 1}},
                    'data': {'batch_size': 1, 'num_workers': 0}, 'model': {'name': 'toy'},
                    'fedprox': {'partition_seed': 42, 'max_rounds': 2, 'local_epochs': 2,
                                'gradient_clip_norm': 5., 'learning_rate': .01,
                                'patience': 10, 'early_stopping_min_delta': 0.}}
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        patches = dict(ROOT=self.root, load_experiment=lambda *a: self.data,
                       partition_dir=lambda *a: self.pdir, load_clients=lambda *a: [self.client()],
                       make_client_model=lambda *a: Toy(),
                       evaluate_federated=lambda state, *a, **kw: {'MAE': 10.-state['w'].item(), 'macro_client_MAE': 10.-state['w'].item()})
        for name, replacement in patches.items():
            self.stack.enter_context(patch.object(fedprox, name, replacement))

    def client(self, observations=2):
        loader = DataLoader([batch(), batch()], batch_size=None,
                            generator=torch.Generator().manual_seed(42))
        return SimpleNamespace(client_id=0, nodes=1, train_observations=observations,
                               loaders={'train': loader})

    def train(self, output='runs', **kw):
        with redirect_stdout(io.StringIO()):
            return fedprox.train_fedprox('synthetic', 1, .1, 42, self.cfg, 'cpu', output=output, **kw)

    def folder(self, output='runs'):
        return self.root / output / 'synthetic/k001/mu0p1/seed_042'

    def test_empty_client_rejected_before_model_creation(self):
        with patch.object(fedprox, 'load_clients', return_value=[self.client(0)]), patch.object(fedprox, 'make_client_model') as model:
            with self.assertRaisesRegex(ValueError, r'clientes inválidos: \[0\]'):
                self.train()
            model.assert_not_called()

    def test_resume_matches_uninterrupted_cpu_run(self):
        full = self.train('full')
        save = fedprox._save_atomic
        def interrupt_after_checkpoint(obj, path):
            save(obj, path)
            if path.name == 'last.pt':
                raise RuntimeError('simulated interruption')
        with patch.object(fedprox, '_save_atomic', interrupt_after_checkpoint):
            with self.assertRaisesRegex(RuntimeError, 'simulated'):
                self.train('resume')
        st = torch.load(self.folder('resume') / 'last.pt', weights_only=False)
        self.assertEqual(st['proximal_mu'], .1)
        resumed = self.train('resume')
        self.assertEqual(full['validation'], resumed['validation'])
        self.assertEqual(full['rounds_completed'], resumed['rounds_completed'])
        for name in ['full', 'resume']:
            self.assertEqual(json.loads((self.folder(name) / 'metrics.json').read_text())['proximal_mu'], .1)
        a = torch.load(self.folder('full') / 'best_model.pt', weights_only=True)
        b = torch.load(self.folder('resume') / 'best_model.pt', weights_only=True)
        self.assertTrue(torch.equal(a['model_state_dict']['w'], b['model_state_dict']['w']))

    def test_resume_rejects_unknown_or_different_mu(self):
        self.folder().mkdir(parents=True)
        for st in [{}, {'proximal_mu': .2}]:
            torch.save(st, self.folder() / 'last.pt')
            with self.assertRaisesRegex(ValueError, 'mu verificável'):
                self.train()

    def test_grid_cache_validates_mu(self):
        self.folder().mkdir(parents=True)
        with patch.object(run_grid, 'ROOT', self.root), patch.object(federated, 'partition_dir', return_value=self.pdir):
            for data, status in [({'proximal_mu': .1}, 'cached'), ({'proximal_mu': .2}, 'failed'), ({}, 'failed')]:
                (self.folder() / 'metrics.json').write_text(json.dumps(data))
                result = run_grid.execute(('synthetic', 42, 1, .1), self.cfg, False, None)
                self.assertEqual(result['status'], status)
                if status == 'failed':
                    self.assertIn('mu verificável', result['error'])


if __name__ == '__main__':
    unittest.main()
