"""Ablação `1hop_min2`: seleção, normalização, exposição e contabilização. Sintético, sem dataset."""
import unittest

import numpy as np
import pandas as pd
import torch

from types import SimpleNamespace
from unittest.mock import patch

from torch import nn

from experiments.data import ExperimentData
from experiments.fedgnnhet import HALOS, build_halo, format_exactness

T, F = 6, 1


def grafo(adj: np.ndarray) -> ExperimentData:
    """ExperimentData mínimo: só `adj`, `series`, `mask` e o scaler importam para o `build_halo`."""
    n = adj.shape[0]
    # série distinta por nó, para conseguir identificar quem foi agregado
    series = (np.arange(T)[:, None, None] * 100 + np.arange(n)[None, :, None] + 1).astype(np.float32)
    mask = np.ones_like(series, dtype=bool)
    ei = np.stack(np.nonzero(adj))
    return ExperimentData(
        name="toy", series=series, mask=mask, feature_names=["v"], target_feature_index=0,
        timestamps=pd.date_range("2024-01-01", periods=T, freq="h"),
        node_ids=[str(i) for i in range(n)], adj=adj.astype(np.float32),
        edge_index=torch.from_numpy(ei).long(), edge_weight=torch.from_numpy(adj[ei[0], ei[1]]).float(),
        origins={"train": np.array([0]), "validation": np.array([]), "test": np.array([])},
        input_steps=1, horizon=1,
        scaler_mean=np.zeros(F, np.float32), scaler_std=np.ones(F, np.float32))


class SelecaoMin2(unittest.TestCase):
    """Cliente = {0}. O nó 0 recebe arestas de fora conforme o caso."""

    def caso(self, n_vizinhos_externos: int):
        n = 1 + max(n_vizinhos_externos, 1)
        adj = np.zeros((n, n), np.float64)
        for u in range(1, 1 + n_vizinhos_externos):
            adj[u, 0] = u                      # pesos distintos, para testar a normalização
        return grafo(adj), np.array([0])

    def test_grau_zero_nao_gera_halo_em_nenhuma_variante(self):
        d, nodes = self.caso(0)
        for v in ("1hop", "1hop_min2"):
            h = build_halo(d, nodes, v)
            self.assertEqual(h.size, 0, v)
            self.assertEqual(h.n_suppressed, 0, v)

    def test_grau_um_gera_halo_no_1hop_e_e_suprimido_no_min2(self):
        d, nodes = self.caso(1)
        a = build_halo(d, nodes, "1hop")
        self.assertEqual(a.size, 1)
        self.assertEqual(a.n_suppressed, 0)
        self.assertEqual(int(a.n_external[0]), 1)

        b = build_halo(d, nodes, "1hop_min2")
        self.assertEqual(b.size, 0, "grau 1 não deve receber nó virtual no min2")
        self.assertEqual(b.n_suppressed, 1)
        self.assertEqual(list(b.suppressed_local), [0])

    def test_grau_um_no_1hop_entrega_a_serie_do_vizinho_identica(self):
        """É a razão de existir o min2: com um vizinho, o agregado NÃO é uma soma, é a própria série."""
        d, nodes = self.caso(1)
        h = build_halo(d, nodes, "1hop")
        esperado = ((d.series[:, 1, :] - d.scaler_mean) / d.scaler_std).astype(np.float32)
        np.testing.assert_allclose(h.values[:, 0, :], esperado, atol=0)
        np.testing.assert_allclose(h.mask[:, 0, :], d.mask[:, 1, :].astype(np.float32), atol=0)

    def test_grau_dois_mantem_halo_nas_duas_e_normaliza_igual(self):
        d, nodes = self.caso(2)
        a, b = build_halo(d, nodes, "1hop"), build_halo(d, nodes, "1hop_min2")
        self.assertEqual((a.size, b.size), (1, 1))
        self.assertEqual(b.n_suppressed, 0)
        # W_v e o agregado não mudam: o min2 só decide QUEM entra, não como se normaliza
        np.testing.assert_allclose(a.weight, b.weight, atol=0)
        np.testing.assert_allclose(a.values, b.values, atol=0)
        # média ponderada explícita: pesos 1 e 2 sobre os nós 1 e 2
        x = (d.series - d.scaler_mean) / d.scaler_std
        esperado = (1 * x[:, 1, :] + 2 * x[:, 2, :]) / 3.0
        np.testing.assert_allclose(a.values[:, 0, :], esperado.astype(np.float32), rtol=1e-6)

    def test_min2_nao_altera_static_nem_none(self):
        """A variante nova não pode mexer no caminho das existentes."""
        d, nodes = self.caso(1)
        self.assertIn("1hop_min2", HALOS)
        s = build_halo(d, nodes, "static")
        self.assertEqual(s.size, 1, "o static continua criando halo para grau 1")
        self.assertEqual(s.n_suppressed, 0)
        # static é constante no tempo: todas as linhas iguais
        self.assertTrue(np.allclose(s.values, s.values[:1]))


class ContabilizacaoEExposicao(unittest.TestCase):
    def cenario(self):
        """Cliente {0,1}: nó 0 com um vizinho externo, nó 1 com dois. Um suprimido, um mantido."""
        adj = np.zeros((5, 5), np.float64)
        adj[2, 0] = 1.0
        adj[3, 1] = 1.0
        adj[4, 1] = 3.0
        return grafo(adj), np.array([0, 1])

    def test_contagem_de_nos_e_supressao(self):
        d, nodes = self.cenario()
        a, b = build_halo(d, nodes, "1hop"), build_halo(d, nodes, "1hop_min2")
        self.assertEqual((a.size, a.n_suppressed), (2, 0))
        self.assertEqual((b.size, b.n_suppressed), (1, 1))
        self.assertEqual(list(b.boundary_local), [1], "deve sobrar o nó de grau 2")
        self.assertEqual(list(b.suppressed_local), [0], "deve cair o nó de grau 1")

    def test_payload_cai_na_proporcao_dos_nos_retidos(self):
        """halo_bytes = T × n_b × F × 4 × 2 (valores + máscara). É o que a auditoria recalcula."""
        d, nodes = self.cenario()
        for v, n_esperado in (("1hop", 2), ("1hop_min2", 1)):
            h = build_halo(d, nodes, v)
            bytes_ = h.values.nbytes + h.mask.nbytes
            self.assertEqual(bytes_, T * n_esperado * F * 4 * 2, v)

    def test_exposicao_de_vizinho_unico_some_no_min2(self):
        d, nodes = self.cenario()
        a, b = build_halo(d, nodes, "1hop"), build_halo(d, nodes, "1hop_min2")
        self.assertEqual(int((a.n_external == 1).sum()), 1)
        self.assertEqual(int((b.n_external == 1).sum()), 0,
                         "nenhum agregado entregue pelo min2 pode ter um único vizinho")



class FormatacaoDoDiagnostico(unittest.TestCase):
    """Chama a função, não procura a string no código-fonte.

    Foi um `{dict:.2e}` que derrubou as execuções de `1hop` em 24/09. Um teste de `inspect.getsource`
    passaria com o comportamento quebrado; estes não.
    """

    def test_sem_nos_suprimidos_omite_a_segunda_parte(self):
        s = format_exactness({"todos": 7e-7, "com_halo": 7e-7, "sem_halo": None})
        self.assertIn("7.00e-07", s)
        self.assertNotIn("sem halo", s)

    def test_com_nos_suprimidos_reporta_os_dois(self):
        s = format_exactness({"todos": 0.5, "com_halo": 6e-7, "sem_halo": 0.5})
        self.assertIn("6.00e-07", s)
        self.assertIn("5.00e-01", s)

    def test_nao_aceita_float_solto(self):
        """Regressão do sentido inverso: se alguém voltar a passar um float, tem de falhar aqui."""
        with self.assertRaises((TypeError, AttributeError)):
            format_exactness(7e-7)


class DiagnosticoDeExatidao(unittest.TestCase):
    """Executa `check_exactness` de verdade, com os pesados trocados por dublês."""

    def cenario(self, todos_suprimidos: bool):
        """Duas saídas idênticas: o erro é zero por construção, o que interessa é o fatiamento."""
        H, n_local = 2, 3
        class Modelo(nn.Module):
            def forward(self, x, x_mask):
                b = x.shape[0]
                return torch.zeros(b, H, x.shape[2], 1)

        janela = {"x": torch.zeros(1, 5, 1), "x_mask": torch.zeros(1, 5, 1)}
        conjunto = [janela]
        sup = np.arange(n_local) if todos_suprimidos else np.array([0])
        halo = SimpleNamespace(suppressed_local=sup)
        cliente = SimpleNamespace(n_local=n_local, node_indices=np.arange(n_local), halo=halo,
                                  loaders={"train": SimpleNamespace(dataset=conjunto)})
        return Modelo(), conjunto, [cliente]

    def roda(self, todos_suprimidos: bool):
        from experiments import fedgnnhet, train as train_mod

        modelo, conjunto, clientes = self.cenario(todos_suprimidos)
        modelo.load_state_dict = lambda *a, **k: None
        # `make_model` é importado dentro de check_exactness, logo o alvo é o módulo de origem
        with patch.object(train_mod, "make_model", return_value=modelo), \
             patch.object(fedgnnhet, "make_client_model", return_value=modelo), \
             patch.object(fedgnnhet, "cpu_state", return_value={}), \
             patch.object(fedgnnhet, "WindowDataset", return_value=conjunto):
            return fedgnnhet.check_exactness(SimpleNamespace(), clientes, {}, "cpu", n_windows=1)

    def test_cliente_parcialmente_suprimido_reporta_os_dois_estratos(self):
        r = self.roda(todos_suprimidos=False)
        self.assertEqual(set(r), {"todos", "com_halo", "sem_halo"})
        self.assertIsNotNone(r["sem_halo"])

    def test_cliente_totalmente_suprimido_nao_levanta(self):
        """Sem a guarda, `err[:, ~m].max()` opera sobre tensor vazio e levanta RuntimeError."""
        r = self.roda(todos_suprimidos=True)
        self.assertEqual(r["com_halo"], 0.0, "não há nó com halo para verificar")
        self.assertIsNotNone(r["sem_halo"])

    def test_min2_tambem_recebe_verificacao_de_exatidao(self):
        """É a variante onde a equivalência é parcial; sem isto a ablação sai sem o número."""
        import inspect

        from experiments.fedgnnhet import train_fedgnnhet
        self.assertIn('halo in ("1hop", "1hop_min2")', inspect.getsource(train_fedgnnhet))

if __name__ == "__main__":
    unittest.main()
