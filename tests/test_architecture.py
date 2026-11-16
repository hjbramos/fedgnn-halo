"""Contratos independentes do domínio e preservação dos estados compartilhados."""
import ast
import copy
from pathlib import Path
import unittest

import numpy as np
import torch
from torch import nn

from federation import ObjectiveResult, cpu_state, fedavg_round, ditto_round, train_local_epoch
from federation.fedpub import MaskedModel, fedpub_round
from models import ModelSpec, EdgeAttributes, build_model


class IndependenceTests(unittest.TestCase):
    # `rglob` e não `glob`: subpastas futuras precisam herdar a regra sem que ninguém se lembre.
    # E as duas direções são bloqueadas — a premissa declarada é que a federação opera sobre estados
    # e não conhece os modelos, logo `federation → models` também é violação.
    PROIBIDO = {
        'federation': ('experiments', 'data_loader', 'models'),
        'models': ('experiments', 'data_loader', 'federation'),
    }

    def test_core_has_no_experiment_or_dataset_imports(self):
        root = Path(__file__).resolve().parents[1]
        vistos = 0
        for package, proibidos in self.PROIBIDO.items():
            for path in (root / package).rglob('*.py'):
                if '__pycache__' in path.parts:
                    continue
                vistos += 1
                tree = ast.parse(path.read_text())
                for node in ast.walk(tree):
                    modules = ([node.module or ''] if isinstance(node, ast.ImportFrom) else
                               [alias.name for alias in node.names] if isinstance(node, ast.Import) else [])
                    for module in modules:
                        self.assertNotIn(module.split('.')[0], proibidos,
                                         f"{path} importa {module}, proibido em {package}/")
        self.assertGreater(vistos, 3, "o varrimento não encontrou arquivos — glob errado?")


class GenericRoundTests(unittest.TestCase):
    def model(self, client=None):
        model = nn.Linear(1, 1, bias=False, dtype=torch.float64)
        with torch.no_grad():
            model.weight.zero_()
        return model

    def optimizer(self, model):
        return torch.optim.SGD(model.parameters(), lr=.1)

    def epoch(self, model, client, optimizer, anchors, seen):
        # Ordinary (input, target) pairs: no graph, window, mask, scaler or traffic dataset.
        x, y = torch.ones(1, 1, dtype=torch.float64), torch.tensor([[client]], dtype=torch.float64)
        def objective(model, batch, step):
            x, y = batch
            loss = (model(x) - y).square().mean()
            return ObjectiveResult(loss, loss.item(), 1)
        return train_local_epoch(model, [(x, y)], optimizer, objective, 100., seen)

    def test_weighted_fedavg_with_arbitrary_task(self):
        state = cpu_state(self.model())
        updated, metric, seen = fedavg_round(state, [1., 3.], [1, 3], self.model, self.optimizer, self.epoch, 1)
        self.assertAlmostEqual(updated['weight'].item(), .5)
        self.assertEqual(metric, 7.)
        self.assertEqual(seen, 2)
        self.assertEqual(state['weight'].item(), 0.)

    def test_ditto_persists_personal_models_and_anchors_to_received_global(self):
        state = cpu_state(self.model())
        personal = [{0.: {'weight': torch.tensor([[2.]], dtype=torch.float64)}},
                    {0.: {'weight': torch.tensor([[4.]], dtype=torch.float64)}}]
        before = copy.deepcopy(personal)
        def local(model, client, optimizer, anchors, candidate, seen):
            self.assertEqual(anchors[0].item(), 0.)
            return self.epoch(model, client, optimizer, anchors, seen)
        global_state, updated, _, _, seen, personal_seen = ditto_round(
            state, personal, [1., 3.], [1, 3], self.model, self.optimizer, self.epoch, local, 1)
        self.assertAlmostEqual(global_state['weight'].item(), .5)
        self.assertAlmostEqual(updated[0][0.]['weight'].item(), 1.8)
        self.assertAlmostEqual(updated[1][0.]['weight'].item(), 3.8)
        self.assertEqual((seen, personal_seen), (2, 2))
        for old, current in zip(before, personal):
            self.assertTrue(torch.equal(old[0.]['weight'], current[0.]['weight']))

    def test_fedpub_uniform_similarity_and_local_masks(self):
        state = cpu_state(self.model())
        mask = {'weight': torch.ones_like(state['weight'])}
        update = fedpub_round([state, state], [mask, mask], [1., 3.], [1, 3], self.model,
                              self.optimizer, self.epoch, lambda state: np.ones(2), .1, 1., 1)
        for received in update.received:
            self.assertAlmostEqual(received['weight'].item(), .4)  # FED-PUB uses similarity, not client size.
        self.assertEqual(update.sent_counts, [1, 1])
        self.assertEqual(update.batches_seen, 2)
        self.assertTrue(torch.equal(mask['weight'], torch.ones_like(mask['weight'])))

    def test_masked_upload_is_not_effective_parameter(self):
        model = self.model()
        with torch.no_grad():
            model.weight.fill_(4.)
        masked = MaskedModel(model, {'weight': torch.tensor([[.5]], dtype=torch.float64)}, .1)
        self.assertEqual(masked.sent_state()[0]['weight'].item(), 4.)
        self.assertEqual(masked.effective_state()['weight'].item(), 2.)


class ModelBoundaryTests(unittest.TestCase):
    def spec(self, n):
        index = torch.stack([torch.arange(n), torch.arange(n)])
        return ModelSpec(n, 2, 1, 1, index, torch.ones(n), np.eye(n, dtype=np.float32))

    def test_shared_state_loads_on_different_client_graphs(self):
        a, b = [build_model('leconv', self.spec(n), hidden_size=4) for n in (2, 5)]
        b.load_state_dict(cpu_state(a), strict=True)
        self.assertNotIn('edge_index', cpu_state(a))
        self.assertEqual(b(torch.ones(1, 2, 5, 1), torch.ones(1, 2, 5, 1)).shape, (1, 1, 5, 1))
        self.assertEqual(b.edge_index.shape[1], 5)

    def test_edge_model_constructs_from_explicit_tensors(self):
        spec = self.spec(2)
        attrs = EdgeAttributes(torch.ones(2, 1), torch.ones(168, 2, 2), torch.ones(168, 2, 2, dtype=torch.bool),
                               torch.zeros(3), torch.ones(3), torch.ones(2, dtype=torch.bool))
        from dataclasses import replace
        model = build_model('leconv_edge', replace(spec, edge_attributes=attrs), hidden_size=4)
        result = model(torch.ones(1, 2, 2, 1), torch.ones(1, 2, 2, 1), torch.zeros(1, 1, dtype=torch.long))
        self.assertEqual(result.shape, (1, 1, 2, 1))
        self.assertTrue(torch.isfinite(result).all())
        with self.assertRaisesRegex(ValueError, 'edge_attributes'):
            build_model('leconv_edge', spec, hidden_size=4)


if __name__ == '__main__':
    unittest.main()
