"""Agregação ponderada de sinais externos em nós virtuais, sem conhecimento de splits ou scalers."""
from dataclasses import dataclass
import numpy as np


@dataclass
class Halo:
    boundary_local: np.ndarray   # (n_b,) posição LOCAL (0..n-1) dos nós de fronteira do cliente
    weight: np.ndarray           # (n_b,) W_v = soma dos pesos das arestas externas que entram em v
    values: np.ndarray           # (T, n_b, F) float32: Σ w x / W (sinais preparados pelo chamador)
    mask: np.ndarray             # (T, n_b, F) float32: Σ w m / W
    n_external: np.ndarray       # (n_b,) nº de vizinhos externos agregados
    suppressed_local: np.ndarray = None  # (n_s,) nós que teriam halo no `1hop` e não têm nesta variante

    @property
    def size(self) -> int:
        return len(self.boundary_local)

    @property
    def n_suppressed(self) -> int:
        return 0 if self.suppressed_local is None else len(self.suppressed_local)


def aggregate_halo(adj: np.ndarray, nodes: np.ndarray, values: np.ndarray, mask: np.ndarray,
                   min_neighbors: int = 1, static_until: int | None = None) -> Halo:
    """adj[u,v] representa u→v; values/mask têm shape (tempo, nós, features).

    O chamador fornece sinais já preparados. static_until é um corte exclusivo opcional.
    """
    A = adj.astype(np.float64)
    local = np.zeros(adj.shape[0], dtype=bool)
    local[nodes] = True
    ext = A[:, nodes].copy()          # (N, n): peso de u -> v para cada nó local v
    ext[local, :] = 0.0               # só vizinhos externos
    W = ext.sum(0)                    # (n,)
    grau = (ext > 0).sum(0)           # (n,) nº de vizinhos externos de cada nó local
    candidatos = np.flatnonzero(W > 0)
    # `1hop_min2`: só recebe nó virtual quem tem >= 2 vizinhos externos. A normalização por coluna
    # é a mesma — W_v continua a soma sobre TODOS os vizinhos externos do nó retido — e o peso da
    # aresta virtual também. Muda só quem entra.
    boundary = np.flatnonzero((W > 0) & (grau >= min_neighbors)) if min_neighbors > 1 else candidatos
    suprimidos = np.setdiff1d(candidatos, boundary, assume_unique=True)
    ext = ext[:, boundary] / W[boundary]  # (N, n_b), colunas somam 1
    std_x = values.astype(np.float32)
    m = mask.astype(np.float32)
    values = np.einsum("tnf,nb->tbf", std_x, ext.astype(np.float32))
    mask = np.einsum("tnf,nb->tbf", m, ext.astype(np.float32))
    if static_until is not None:
        if not 0 < static_until <= len(values):
            raise ValueError("static_until deve delimitar um intervalo temporal não vazio")
        t_end = static_until
        values = np.broadcast_to(values[:t_end].mean(0, keepdims=True), values.shape).copy()
        mask = np.broadcast_to(mask[:t_end].mean(0, keepdims=True), mask.shape).copy()
    return Halo(boundary_local=boundary, weight=W[boundary].astype(np.float32), values=np.ascontiguousarray(values),
                mask=np.ascontiguousarray(mask), n_external=(ext > 0).sum(0),
                suppressed_local=suprimidos)


