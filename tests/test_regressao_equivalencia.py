"""Regressão numérica: valores fixados para que um refatoramento não mude os números em silêncio.

Motivação. Em 24/09 o repositório foi reorganizado (`federation/`, `models/`, `experiments/`) e a
equivalência com a versão anterior foi verificada à mão, em `/tmp`, com worktrees. A verificação
passou — checkpoints e métricas idênticos — mas não ficou no repositório, logo não protegia nada dali
para frente. Este arquivo é essa proteção.

Escopo e limite. Fixa o caminho **determinístico em CPU**: agregação ponderada, uma rodada local com
estado compartilhado, e a construção do halo. Não reproduz os artefatos das campanhas, que vieram de
GPU, onde o scatter-add da LEConv usa atomics e não é determinístico (~1e-3 de MAE ao fim de ~50
rodadas). O que ele detecta é mudança de comportamento do código, não ruído de hardware.

Se um destes testes falhar depois de uma mudança intencional, os valores devem ser regerados **e a
mudança justificada no commit** — nunca atualizados por conveniência.
"""
import unittest

import numpy as np
import torch
from torch import nn

from experiments.federated import aggregate_states, cpu_state
from tests.test_halo_min2 import grafo


class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(3, 2)

    def forward(self, x):
        return self.lin(x)


def treino_local_determinista() -> list[dict]:
    """Três clientes, mesma inicialização, três passos de Adam cada. Sem aleatoriedade de dados.

    Usa MSE: com L1 o gradiente é de sinal e o Adam normaliza, então os três clientes convergiriam
    para estados idênticos a ~1e-9 e o teste de ponderação não distinguiria nada.
    """
    torch.manual_seed(0)
    inicial = cpu_state(Toy())
    estados = []
    for i, seed in enumerate((1, 2, 3)):
        torch.manual_seed(seed)
        m = Toy()
        m.load_state_dict(inicial)
        opt = torch.optim.Adam(m.parameters(), lr=0.01)
        x = torch.arange(12, dtype=torch.float32).reshape(4, 3) + i
        y = (x.sum(1, keepdim=True).repeat(1, 2)) / 10
        for _ in range(3):
            opt.zero_grad()
            nn.functional.mse_loss(m(x), y).backward()
            opt.step()
        estados.append(cpu_state(m))
    return estados


class AgregacaoPonderada(unittest.TestCase):
    """Valores gerados em 24/09, no commit 872b08e + correções, antes do refatoramento."""

    REFERENCIA = {
        "lin.bias": [0.0184853002, 0.4877135754],
        "lin.weight": [0.0255999602, 0.3396385312, -0.4452624023,
                       -0.3949511051, -0.192426011, 0.1847632527],
    }

    def test_estado_agregado_bate_com_a_referencia(self):
        saida = aggregate_states(treino_local_determinista(), [1.0, 2.0, 3.0])
        self.assertEqual(sorted(saida), sorted(self.REFERENCIA))
        for k, esperado in self.REFERENCIA.items():
            # A referência resulta de treino em float32; plataformas podem diferir em poucos ULPs.
            tolerance = 2 * torch.finfo(saida[k].dtype).eps
            np.testing.assert_allclose(saida[k].flatten().numpy(), esperado, rtol=tolerance, atol=1e-9,
                                       err_msg=f"{k} mudou — refatoramento alterou o comportamento?")

    def test_pesos_diferentes_dao_resultado_diferente(self):
        """Guarda contra uma agregação que ignore os pesos e faça média simples."""
        estados = treino_local_determinista()
        a = aggregate_states(estados, [1.0, 2.0, 3.0])
        b = aggregate_states(estados, [1.0, 1.0, 1.0])
        # igualdade EXATA: se os pesos fossem ignorados, o cálculo seria o mesmo e bit a bit igual.
        # `allclose` não serve — a separação entre clientes é ~1e-5 e cai na tolerância padrão.
        self.assertFalse(torch.equal(a["lin.weight"], b["lin.weight"]),
                         "a agregação deve ser ponderada, não média simples")


class ConstrucaoDoHalo(unittest.TestCase):
    """O agregado do halo é determinístico e não deve mudar com reorganização de código."""

    def cenario(self):
        adj = np.zeros((5, 5), np.float64)
        adj[2, 0] = 1.0            # nó 0: um vizinho externo
        adj[3, 1] = 1.0            # nó 1: dois vizinhos externos, pesos 1 e 3
        adj[4, 1] = 3.0
        return grafo(adj), np.array([0, 1])

    def test_valores_do_agregado_sao_estaveis(self):
        from experiments.fedgnnhet import build_halo

        d, nodes = self.cenario()
        h = build_halo(d, nodes, "1hop")
        x = (d.series - d.scaler_mean) / d.scaler_std
        # nó 0: o agregado é a própria série do vizinho 2 (peso normalizado = 1)
        np.testing.assert_allclose(h.values[:, 0, 0], x[:, 2, 0], rtol=0, atol=0)
        # nó 1: média ponderada de 3 e 4 com pesos 1 e 3
        np.testing.assert_allclose(h.values[:, 1, 0], (x[:, 3, 0] + 3 * x[:, 4, 0]) / 4.0, rtol=1e-6)
        # W_v é a soma dos pesos externos, e é o peso da aresta virtual
        np.testing.assert_allclose(h.weight, [1.0, 4.0], rtol=0, atol=1e-9)


if __name__ == "__main__":
    unittest.main()
