"""Com uma LEConv e os mesmos parâmetros, o cliente com halo reproduz o grafo completo nos nós locais."""
import numpy as np
import torch

from federation.halo import aggregate_halo
from models import ModelSpec, build_model

W, H = 6, 2


def _edges(matrix):
    src, dst = np.nonzero(matrix)
    return torch.tensor(np.stack([src, dst]), dtype=torch.long), torch.tensor(matrix[src, dst], dtype=torch.float32)


def _leconv(edge_index, edge_weight, nodes):
    spec = ModelSpec(nodes=nodes, input_steps=W, horizon=H, n_features=1, edge_index=edge_index, edge_weight=edge_weight)
    return build_model("leconv", spec, hidden_size=8).eval()


def test_halo_reproduces_full_graph_and_cut_does_not():
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    edges = [(0, 1, 1.0), (0, 2, 0.5), (1, 2, 2.0), (2, 3, 1.0), (4, 5, 1.0), (4, 7, 0.5),
             (5, 6, 2.0), (6, 7, 1.0), (2, 5, 1.0), (3, 4, 1.5), (3, 5, 1.0)]
    adj = np.eye(8, dtype=np.float32)
    for u, v, w in edges:
        adj[u, v] = adj[v, u] = w
    series = rng.normal(size=(20, 8, 1)).astype(np.float32)
    mask = np.ones_like(series)
    x, m = torch.from_numpy(series[:W])[None], torch.from_numpy(mask[:W])[None]

    central = _leconv(*_edges(adj), 8)
    with torch.no_grad():
        reference = central(x, m)[0, :, :, 0]

    nodes = np.arange(4)
    sub = adj[np.ix_(nodes, nodes)]
    halo = aggregate_halo(adj, nodes, series, mask)
    assert halo.boundary_local.tolist() == [2, 3]
    assert np.allclose(halo.values[:, 1], (1.5 * series[:, 4] + series[:, 5]) / 2.5, atol=1e-6)

    cut = _leconv(*_edges(sub), 4)
    cut.load_state_dict(central.state_dict())
    virtual = 4 + np.arange(halo.size)
    ei, ew = _edges(sub)
    ei = torch.cat([ei, torch.tensor(np.stack([np.r_[virtual, virtual], np.r_[halo.boundary_local, virtual]]))], 1)
    ew = torch.cat([ew, torch.tensor(np.r_[halo.weight, np.ones(halo.size, dtype=np.float32)])])
    with_halo = _leconv(ei, ew, 4 + halo.size)
    with_halo.load_state_dict(central.state_dict())
    with torch.no_grad():
        out_cut = cut(x[:, :, nodes], m[:, :, nodes])[0, :, :, 0]
        xh = torch.cat([x[:, :, nodes], torch.from_numpy(halo.values[:W])[None]], 2)
        mh = torch.cat([m[:, :, nodes], torch.from_numpy(halo.mask[:W])[None]], 2)
        out_halo = with_halo(xh, mh)[0, :, :4, 0]

    assert torch.allclose(out_halo, reference[:, nodes], atol=1e-5)
    assert (out_cut[:, 2:] - reference[:, 2:4]).abs().max() > 1e-3     # nós com entrada externa
    assert torch.allclose(out_cut[:, :2], reference[:, :2], atol=1e-6)  # nós interiores
