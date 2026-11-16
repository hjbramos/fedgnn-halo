"""Referência Local (Ditto com lambda = 0) e o caminho `ditto.local_only` (results/lote6_sunt_local/).

Conferem, em CPU e com modelo sintético:
- que uma época consome do gerador do DataLoader o mesmo que `consume_like_epoch`;
- que, com lambda = 0, a atualização pessoal não depende do estado global recebido;
- que o caminho Local reproduz bit a bit os modelos pessoais, o histórico e o estado dos geradores do caminho original
  (e que, sem consumir o gerador, não reproduz: o teste é sensível à ordem dos lotes);
- que lambda = 0 não ganha termo proximal; que local_only recusa lambda > 0 e last.pt de outro caminho;
- que a retomada do Local reproduz a execução contínua.
"""
import copy
import io
import json
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

import pandas as pd
import torch
from torch.utils.data import DataLoader

from experiments import ditto
from federation import ditto_round
from tests.test_ditto_retomada import DittoResumeTests, Toy, batch


def _gen_state(loader):
    return loader.generator.get_state()


class ConsumoDoGerador(unittest.TestCase):
    def test_epoca_real_e_emulada_consomem_igual(self):
        for kw in ({"batch_size": 3}, {"batch_size": 32}, {"batch_size": 3, "drop_last": True}):
            real = DataLoader(list(range(10)), shuffle=True, generator=torch.Generator().manual_seed(5), **kw)
            emul = DataLoader(list(range(10)), shuffle=True, generator=torch.Generator().manual_seed(5), **kw)
            for _ in range(3):
                for _ in real:
                    pass
                ditto.consume_like_epoch(emul)
                self.assertTrue(torch.equal(_gen_state(real), _gen_state(emul)), kw)
            self.assertEqual([b.tolist() for b in real], [b.tolist() for b in emul])   # mesma ordem dali em diante

    def test_so_o_primeiro_lote_nao_basta(self):
        """Ao se esgotar, o RandomSampler sorteia outra permutação: consumir só o início deixaria o gerador atrasado."""
        real = DataLoader(list(range(10)), batch_size=3, shuffle=True, generator=torch.Generator().manual_seed(5))
        parcial = DataLoader(list(range(10)), batch_size=3, shuffle=True, generator=torch.Generator().manual_seed(5))
        for _ in real:
            pass
        next(iter(parcial))
        self.assertFalse(torch.equal(_gen_state(real), _gen_state(parcial)))


class IndependenciaDoGlobal(unittest.TestCase):
    def test_lambda_zero_sem_termo_proximal(self):
        self.assertEqual(ditto.parse_lambda("0"), 0.0)
        seen = {}
        with patch.object(ditto, "train_local_epoch", lambda *a: seen.setdefault("reg", a[6]) or (0.0, 0)):
            ditto.train_epoch_personalized(Toy(), [torch.tensor(9.)], 0.0, [], None, "cpu", 5.)
        self.assertIsNone(seen["reg"])
        seen.clear()
        with patch.object(ditto, "train_local_epoch", lambda *a: seen.setdefault("reg", a[6]) and (0.0, 0)):
            ditto.train_epoch_personalized(Toy(), [torch.tensor(9.)], 0.1, [], None, "cpu", 5.)
        self.assertIsNotNone(seen["reg"])

    def test_estado_global_nao_altera_o_pessoal(self):
        """ditto_round com lambda = 0 e estados globais diferentes: mesmos modelos pessoais (mesmo gerador)."""
        def run(w_global):
            client = DittoResumeTests.client(0, 10., 3)
            personal = [{0.0: {"w": torch.tensor(1.5, dtype=torch.float64)}}]
            train = lambda model, c, opt, anchors, seen: ditto.train_epoch(model, c.loaders["train"], opt, "cpu", 5., seen)
            train_p = lambda model, c, opt, anchors, lam, seen: ditto.train_epoch_personalized(
                model, anchors, lam, c.loaders["train"], opt, "cpu", 5., seen)
            out = ditto_round({"w": torch.tensor(w_global, dtype=torch.float64)}, personal, [client], [3], lambda c: Toy(),
                              lambda m: torch.optim.Adam(m.parameters(), lr=.05), train, train_p, 2)
            return out[1][0][0.0]["w"], _gen_state(client.loaders["train"])
        (pa, ga), (pb, gb) = run(0.0), run(-37.0)
        self.assertTrue(torch.equal(pa, pb))
        self.assertTrue(torch.equal(ga, gb))


class LocalIgualAoDittoLambdaZero(DittoResumeTests):
    """Mesmo arnês sintético da retomada do Ditto, com lambda = 0 nos dois caminhos."""

    def setUp(self):
        super().setUp()
        self.cfg["ditto"]["lambda_candidates"] = [0.0]

    def run_path(self, output, local_only, evaluate_test=True):
        cfg = copy.deepcopy(self.cfg)
        cfg["ditto"].update(local_only=local_only, evaluate_test=evaluate_test, keep_last_checkpoint=True)
        clients = [self.client(0, 10., 3), self.client(1, 4., 1)]
        with patch.object(ditto, "load_clients", lambda *a: clients), redirect_stdout(io.StringIO()):
            res = ditto.train_ditto("synthetic", 2, 0, 42, cfg, "cpu", output=output)
        root = self.root / output / "synthetic/k002/lam0/seed_042"
        return res, root, [_gen_state(c.loaders["train"]) for c in clients]

    def test_mesma_trajetoria_pessoal(self):
        ro, do, go = self.run_path("orig", False)
        rl, dl, gl = self.run_path("local", True)
        self.assertEqual(rl["method"], "local")
        self.assertEqual(ro["best_round"], rl["best_round"])
        self.assertEqual(ro["validation"], rl["validation"])
        self.assertEqual(ro["test"], rl["test"])
        self.assertEqual(rl["model_communication_bytes"], 0)
        self.assertNotIn("validation_global", rl)
        ho, hl = (pd.read_csv(d / "history.csv") for d in (do, dl))
        cols = ["round", "train_scaled_MAE", "validation_MAE", "validation_macro_client_MAE"]
        pd.testing.assert_frame_equal(ho[cols], hl[cols], check_exact=True)
        for name in ("best_model.pt", "last.pt"):
            a, b = (torch.load(d / name, weights_only=False) for d in (do, dl))
            for pa, pb in zip(a["personal"], b["personal"]):
                self.assertTrue(torch.equal(pa[0.0]["w"], pb[0.0]["w"]), name)
            self.assertIsNone(b["global_state"])
        for a, b in zip(go, gl):
            self.assertTrue(torch.equal(a, b))

    def test_sem_consumir_o_gerador_diverge(self):
        _, do, _ = self.run_path("orig", False)
        with patch.object(ditto, "consume_like_epoch", lambda loader: None):
            _, dl, _ = self.run_path("local", True)
        a, b = (torch.load(d / "last.pt", weights_only=False) for d in (do, dl))
        self.assertFalse(all(torch.equal(pa[0.0]["w"], pb[0.0]["w"]) for pa, pb in zip(a["personal"], b["personal"])))

    def test_recusas(self):
        cfg = copy.deepcopy(self.cfg)
        cfg["ditto"].update(local_only=True)
        with self.assertRaisesRegex(ValueError, "local_only"), redirect_stdout(io.StringIO()):
            ditto.train_ditto("synthetic", 2, 0.1, 42, cfg, "cpu", output="x")
        self.run_path("misto", False)
        with self.assertRaisesRegex(ValueError, "last.pt"):
            self.run_path("misto", True)

    def test_teste_adiado(self):
        r, d, _ = self.run_path("adiado", True, evaluate_test=False)
        self.assertNotIn("test", r)
        self.assertFalse(r["test_evaluated"])
        cfg = copy.deepcopy(self.cfg)
        cfg["ditto"].update(local_only=True)
        clients = [self.client(0, 10., 3), self.client(1, 4., 1)]
        with patch.object(ditto, "load_clients", lambda *a: clients), redirect_stdout(io.StringIO()):
            m = ditto.evaluate_test_from_checkpoint("synthetic", 2, 0, 42, cfg, "cpu", output="adiado")
            self.assertTrue(m["test_evaluated"])
            self.assertEqual(m["validation_recomputed"]["MAE"], r["validation"]["MAE"])
            self.assertIn("test", json.loads((d / "metrics.json").read_text()))
            with self.assertRaisesRegex(ValueError, "já avaliado"):
                ditto.evaluate_test_from_checkpoint("synthetic", 2, 0, 42, cfg, "cpu", output="adiado")

    def test_retomada_local(self):
        full, dfull, _ = self.run_path("full", True)
        save, calls = ditto._save_atomic, {"last": 0}

        def interrupt(obj, path):
            save(obj, path)
            if path.name == "last.pt":
                calls["last"] += 1
                if calls["last"] == 2:
                    raise RuntimeError("simulated interruption")

        with patch.object(ditto, "_save_atomic", interrupt):
            with self.assertRaisesRegex(RuntimeError, "simulated"):
                self.run_path("resume", True)
        resumed, dres, _ = self.run_path("resume", True)
        for key in ("validation", "test", "best_round", "rounds_completed"):
            self.assertEqual(full[key], resumed[key], key)
        hist = [pd.read_csv(d / "history.csv").drop(columns="elapsed_s") for d in (dfull, dres)]
        pd.testing.assert_frame_equal(*hist, check_exact=True)
        a, b = (torch.load(d / "best_model.pt", weights_only=False) for d in (dfull, dres))
        for pa, pb in zip(a["personal"], b["personal"]):
            self.assertTrue(torch.equal(pa[0.0]["w"], pb[0.0]["w"]))

    # os testes herdados do Ditto per_client não se aplicam a este arnês
    test_resume_matches_uninterrupted_cpu_run = None
    test_two_uninterrupted_runs_match = None


if __name__ == "__main__":
    unittest.main()
