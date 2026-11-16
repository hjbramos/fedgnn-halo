"""Regressão do caminho REAL de treino: `train_fedgnnhet` ponta a ponta.

Por que existe. `tests/test_regressao_equivalencia.py` fixa a agregação e a construção do halo, mas
o treino local dali é escrito dentro do próprio teste — não passa pelos runners. Ele não protegeria
uma mudança na sequência de rodadas, na avaliação ou na gravação de checkpoint, que é justamente o
que a reorganização de 24/09 mexeu. Este arquivo fecha essa lacuna: monta um dataset e uma partição
sintéticos em disco e chama `train_fedgnnhet`.

O que ele fixa são **relações**, não constantes: determinismo em CPU, o halo alterando a previsão, a
reposição da aresta cortada, a exatidão contra o centralizado, e as identidades de contabilização de
comunicação. Relações sobrevivem a mudanças legítimas de hiperparâmetro e continuam falhando se o
comportamento mudar — constantes fixadas só sobreviveriam ao primeiro ajuste de `lr`.

Escopo e limite. CPU e determinístico. Não reproduz os artefatos das campanhas, que vieram de GPU com
scatter-add não determinístico, e **não** fixa o MAE de `halo_none`/`halo_1hop` do METR-LA — a
comparação com a versão pré-reorganização foi feita à mão, com worktrees, e não é reproduzida aqui.
"""
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from experiments.data import ExperimentData

T, N, F, W, H = 40, 6, 1, 3, 2
K = 2


def dados() -> ExperimentData:
    """Série determinística, grafo em caminho 0–1–2–3–4–5 com pesos distintos."""
    rng = np.random.default_rng(0)
    series = (np.sin(np.arange(T)[:, None, None] / 3.0 + np.arange(N)[None, :, None])
              + 0.1 * rng.standard_normal((T, N, F))).astype(np.float32)
    mask = np.ones_like(series, dtype=bool)
    adj = np.zeros((N, N), np.float32)
    for i in range(N - 1):
        adj[i, i + 1] = adj[i + 1, i] = 0.5 + 0.1 * i
    ei = np.stack(np.nonzero(adj))
    n_janelas = T - W - H + 1
    origens = np.arange(n_janelas) + W
    return ExperimentData(
        name="toy", series=series, mask=mask, feature_names=["v"], target_feature_index=0,
        timestamps=pd.date_range("2024-01-01", periods=T, freq="h"),
        node_ids=[str(i) for i in range(N)], adj=adj,
        edge_index=torch.from_numpy(ei).long(), edge_weight=torch.from_numpy(adj[ei[0], ei[1]]).float(),
        origins={"train": origens[:24], "validation": origens[24:30], "test": origens[30:]},
        input_steps=W, horizon=H,
        scaler_mean=series.reshape(-1, F).mean(0), scaler_std=series.reshape(-1, F).std(0))


def particao(raiz: Path, d: ExperimentData) -> Path:
    """Dois clientes contíguos: {0,1,2} e {3,4,5}. A aresta 2–3 é a que cruza a fronteira."""
    pdir = raiz / "part"
    pdir.mkdir(parents=True, exist_ok=True)
    assign = np.array([0, 0, 0, 1, 1, 1])
    (pdir / "metadata.json").write_text(json.dumps({"num_clients": K}))
    np.save(pdir / "client_assignments.npy", assign)
    np.save(pdir / "node_ids.npy", np.array(d.node_ids, dtype=object), allow_pickle=True)
    for c in range(K):
        cdir = pdir / f"client_{c:03d}"
        cdir.mkdir(exist_ok=True)
        nodes = np.flatnonzero(assign == c)
        np.save(cdir / "node_indices.npy", nodes)
        local = {int(n): i for i, n in enumerate(nodes)}
        pares = [(local[u], local[v]) for u in nodes for v in nodes if d.adj[u, v] > 0]
        ei = np.array(pares, dtype=np.int64).T if pares else np.zeros((2, 0), np.int64)
        np.save(cdir / "edge_index.npy", ei)
        np.save(cdir / "edge_weight.npy",
                np.array([d.adj[nodes[a], nodes[b]] for a, b in pares], np.float32))
    return pdir


def config(saida: str) -> dict:
    return {
        "windows": {"toy": {"input_steps": W, "horizon": H}},
        "data": {"batch_size": 8, "num_workers": 0},
        "model": {"name": "leconv", "hidden_size": 4},
        "fedgnnhet": {"partition_seed": 42, "learning_rate": 0.01, "local_epochs": 1,
                      "max_rounds": 2, "patience": 30, "early_stopping_min_delta": 0.0,
                      "gradient_clip_norm": 5.0},
        "output": saida,
    }


class TreinoPontaAPonta(unittest.TestCase):
    """Invariantes do caminho real, verificadas em CPU determinística.

    Fixa relações (determinismo, efeito do halo, exatidão, identidades de contabilização) em vez de
    constantes numéricas: as relações sobrevivem a mudanças legítimas de hiperparâmetro e continuam
    falhando se o comportamento mudar.
    """

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.d = dados()
        self.pdir = particao(self.tmp, self.d)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def roda(self, halo: str) -> dict:
        from experiments import fedgnnhet

        cfg = config("out")
        with patch.object(fedgnnhet, "ROOT", self.tmp), \
             patch.object(fedgnnhet, "load_experiment", lambda *a, **k: self.d), \
             patch.object(fedgnnhet, "partition_dir", lambda *a, **k: self.pdir), \
             patch.object(fedgnnhet, "gpu_name", lambda *a, **k: None):
            torch.manual_seed(0)
            return fedgnnhet.train_fedgnnhet("toy", K, halo, 7, cfg, torch.device("cpu"), resume=False)

    def test_none_e_1hop_sao_deterministicos_e_diferentes(self):
        """Duas execuções idênticas dão o mesmo; e o halo muda o resultado (senão não faz nada)."""
        a1, a2 = self.roda("none"), self.roda("none")
        self.assertEqual(a1["test"]["MAE"], a2["test"]["MAE"], "o caminho em CPU deve ser determinístico")
        b = self.roda("1hop")
        self.assertNotEqual(a1["test"]["MAE"], b["test"]["MAE"], "o halo tem de alterar a previsão")

    def test_halo_repoe_a_aresta_cortada(self):
        """A partição corta 2–3; cada cliente ganha exatamente um nó virtual."""
        r = self.roda("1hop")
        self.assertEqual(r["halo_nodes_total"], 2)
        self.assertEqual(r["client_halo_nodes"], [1, 1])
        # cada nó de fronteira tem UM vizinho externo -> exposição máxima, e min2 suprime os dois
        self.assertEqual(r["halo_single_neighbour_pct"], 100.0)

    def test_min2_suprime_tudo_neste_grafo_e_nao_quebra(self):
        """Caso-limite real: todos os nós de fronteira têm grau externo 1, logo o min2 zera o halo."""
        r = self.roda("1hop_min2")
        self.assertEqual(r["halo_nodes_total"], 0)
        self.assertEqual(r["halo_suppressed_nodes"], 2)
        self.assertEqual(r["halo_bytes"], 0)

    def test_exatidao_do_1hop_reproduz_o_centralizado(self):
        """A primeira linha de evidência do método, verificada no caminho real."""
        r = self.roda("1hop")
        self.assertLess(r["halo_exactness_max_abs_error"]["com_halo"], 1e-5)

    def test_contabilizacao_de_comunicacao_confere(self):
        r = self.roda("1hop")
        self.assertEqual(r["model_communication_bytes"],
                         2 * r["model_state_bytes"] * r["num_clients"] * r["rounds_completed"])
        self.assertEqual(r["halo_bytes"], T * r["halo_nodes_total"] * F * 4 * 2)
        self.assertEqual(r["total_communication_bytes"],
                         r["model_communication_bytes"] + r["halo_bytes"])


if __name__ == "__main__":
    unittest.main()
