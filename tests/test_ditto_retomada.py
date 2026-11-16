"""Retomada do Ditto: interromper depois de gravar o last.pt e retomar dá o mesmo resultado que rodar direto.

Regressão CPU/sintética (lote 5, plano em results/lote5_sunt_k166/plano.md), no molde de
tests/test_fedprox.py::test_resume_matches_uninterrupted_cpu_run: modelo de um parâmetro, dois clientes,
λ por cliente com dois candidatos. Verifica a implementação em CPU; não garante equivalência na GPU.
"""
import io
import tempfile
import unittest
from contextlib import ExitStack, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from experiments import ditto


class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.tensor(0., dtype=torch.float64))

    def forward(self, x, mask):
        return self.w * x


def batch(target):
    return dict(x=torch.ones(2, 1, 1, 1, dtype=torch.float64),
                x_mask=torch.ones(2, 1, 1, 1, dtype=torch.bool),
                y=torch.tensor([target, 999.], dtype=torch.float64).reshape(2, 1, 1, 1),
                y_mask=torch.tensor([True, False]).reshape(2, 1, 1, 1))


def fake_personalized(personal, clients, data, cfg, split, device, chosen=None, per_node=False, per_client=False):
    """Métrica determinística que depende dos estados: cada cliente escolhe o candidato mais próximo do seu alvo."""
    picked, maes = [], []
    for k, cands in enumerate(personal):
        target = 10. if k == 0 else 4.
        err = {l: abs(target - s["w"].item()) for l, s in cands.items()}
        lam = chosen[k] if chosen is not None else min(err, key=err.get)
        picked.append(lam)
        maes.append(err[lam])
    mae = sum(maes) / len(maes)
    return {"MAE": mae, "macro_client_MAE": mae}, picked


class DittoResumeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.pdir = self.root / "partition"
        self.pdir.mkdir()
        (self.pdir / "metadata.json").write_text('{"num_clients": 2}')
        self.data = SimpleNamespace(nodes=2, input_steps=1, horizon=1, feature_names=["signal"],
                                    target_feature_index=0)
        self.cfg = {"method": "ditto", "output": "runs",
                    "windows": {"synthetic": {"input_steps": 1, "horizon": 1}},
                    "data": {"batch_size": 1, "num_workers": 0}, "model": {"name": "toy"},
                    "ditto": {"partition_seed": 42, "max_rounds": 4, "local_epochs": 2, "gradient_clip_norm": 5.,
                              "learning_rate": .05, "patience": 100, "early_stopping_min_delta": 0.,
                              "lambda_candidates": [.01, 1.]}}
        stack = ExitStack()
        self.addCleanup(stack.close)
        patches = dict(ROOT=self.root, load_experiment=lambda *a: self.data, partition_dir=lambda *a: self.pdir,
                       load_clients=lambda *a: [self.client(0, 10., 3), self.client(1, 4., 1)],
                       make_client_model=lambda *a: Toy(), evaluate_personalized=fake_personalized,
                       evaluate_federated=lambda state, *a, **kw: {"MAE": abs(7. - state["w"].item()),
                                                                   "macro_client_MAE": abs(7. - state["w"].item())})
        for name, replacement in patches.items():
            stack.enter_context(patch.object(ditto, name, replacement))

    @staticmethod
    def client(cid, target, observations):
        # embaralhamento de verdade, com alvos dos dois lados do parâmetro (o sinal do gradiente do MAE muda entre
        # lotes, então a ordem altera o Adam): a retomada só iguala se restaurar o estado do gerador
        loader = DataLoader([batch(target), batch(-target), batch(3. * target)], batch_size=None, shuffle=True,
                            generator=torch.Generator().manual_seed(100 + cid))
        return SimpleNamespace(client_id=cid, nodes=1, train_observations=observations, loaders={"train": loader})

    def train(self, output):
        with redirect_stdout(io.StringIO()):
            return ditto.train_ditto("synthetic", 2, "per_client", 42, self.cfg, "cpu", output=output)

    def folder(self, output):
        return self.root / output / "synthetic/k002/lam_per_client/seed_042"

    def test_resume_matches_uninterrupted_cpu_run(self):
        full = self.train("full")
        save, calls = ditto._save_atomic, {"last": 0}

        def interrupt_after_second_checkpoint(obj, path):
            save(obj, path)
            if path.name == "last.pt":
                calls["last"] += 1
                if calls["last"] == 2:
                    raise RuntimeError("simulated interruption")

        with patch.object(ditto, "_save_atomic", interrupt_after_second_checkpoint):
            with self.assertRaisesRegex(RuntimeError, "simulated"):
                self.train("resume")
        self.assertEqual(torch.load(self.folder("resume") / "last.pt", weights_only=False)["round"], 2)
        resumed = self.train("resume")

        for key in ["validation", "test", "validation_global", "test_global", "client_lambda", "best_round",
                    "rounds_completed"]:
            self.assertEqual(full[key], resumed[key], key)
        hist = {name: pd.read_csv(self.folder(name) / "history.csv").drop(columns="elapsed_s") for name in ("full", "resume")}
        pd.testing.assert_frame_equal(hist["full"], hist["resume"], check_exact=True)
        a = torch.load(self.folder("full") / "best_model.pt", weights_only=False)
        b = torch.load(self.folder("resume") / "best_model.pt", weights_only=False)
        self.assertTrue(torch.equal(a["global_state"]["w"], b["global_state"]["w"]))
        for pa, pb in zip(a["personal"], b["personal"]):
            self.assertEqual(pa.keys(), pb.keys())
            for lam in pa:
                self.assertTrue(torch.equal(pa[lam]["w"], pb[lam]["w"]), lam)
        self.assertFalse((self.folder("resume") / "last.pt").exists())

    def test_two_uninterrupted_runs_match(self):
        """Controle: sem interrupção, duas execuções independentes coincidem (determinismo em CPU)."""
        self.assertEqual(self.train("a")["validation"], self.train("b")["validation"])


if __name__ == "__main__":
    unittest.main()
