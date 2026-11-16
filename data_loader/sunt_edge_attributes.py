"""Atributos operacionais de aresta do SUNT, da representação oficial `passenger_flow_edge_dynamics_v1`.

Traz, por aresta, a distância Haversine (estática) e os perfis de tempo de segmento e velocidade
derivada por hora da semana (168 valores), com máscara de observação. São os atributos que o
`sunt-fedstgnn` usa para condicionar o peso de cada aresta na janela prevista.

Alinhamento: a representação oficial tem 2882 nós e 9232 arestas dirigidas; o nosso grafo `transit`
tem 2903 nós e 9332 arestas. Os 2882 são subconjunto dos 2903 (os 21 excluídos não têm coordenadas) e
as 9232 arestas mapeiam todas para arestas nossas. As restantes ficam sem atributos e o modelo
aplica-lhes um peso por defeito.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

def default_root() -> Path:
    import os
    root = os.environ.get("FEDGNN_EDGE_ATTRIBUTES_DIR")
    if root:
        return Path(root).expanduser()
    raise FileNotFoundError(
        "A variante leconv_edge exige atributos externos preparados. "
        "Defina model.edge_attributes_root ou FEDGNN_EDGE_ATTRIBUTES_DIR; "
        "essa variante não integra os exemplos autocontidos.")


def load_edge_attributes(node_ids, edge_index, root: str | Path | None = None) -> dict:
    """Mapeia os atributos oficiais para a ordem das nossas arestas.

    Devolve `static` (E,1), `profiles` (168,E,2), `mask` (168,E,2), `scaler_mean/std` (3,) e
    `has_attributes` (E,) — False nas arestas sem correspondência na representação oficial.
    """
    root = Path(root) if root else default_root()
    if not root.exists():
        raise FileNotFoundError(f"representação oficial não encontrada em {root}")
    ids = np.load(root / "stop_ids.npy").astype(str)
    off_ei = np.load(root / "edge_index.npy")
    static = np.load(root / "edge_static_features.npy")
    profiles = np.load(root / "edge_temporal_profiles.npy")
    mask = np.load(root / "edge_temporal_profile_mask.npy")
    scaler = np.load(root / "edge_scaler.npz")

    position = {s: i for i, s in enumerate(np.asarray(node_ids, dtype=str))}
    src = np.array([position.get(ids[u], -1) for u in off_ei[0]])
    dst = np.array([position.get(ids[v], -1) for v in off_ei[1]])
    official = {(int(a), int(b)): j for j, (a, b) in enumerate(zip(src, dst)) if a >= 0 and b >= 0}

    E = edge_index.shape[1]
    out_static = np.zeros((E, 1), dtype=np.float32)
    out_profiles = np.zeros((168, E, 2), dtype=np.float32)
    out_mask = np.zeros((168, E, 2), dtype=bool)
    has = np.zeros(E, dtype=bool)
    ours = np.asarray(edge_index)
    for e in range(E):
        j = official.get((int(ours[0, e]), int(ours[1, e])))
        if j is None:
            continue
        out_static[e] = static[j]
        out_profiles[:, e] = profiles[:, j]
        out_mask[:, e] = mask[:, j]
        has[e] = True
    return {"static": out_static, "profiles": out_profiles, "mask": out_mask,
            "scaler_mean": scaler["mean"].astype(np.float32), "scaler_std": scaler["std"].astype(np.float32),
            "has_attributes": has}
