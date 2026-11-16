"""Agrupamentos intercambiáveis e exportação para os treinadores."""
import json
from pathlib import Path
import numpy as np


def assignments(adj, options, seed):
    n = len(adj)
    method = options.get('method', 'contiguous')
    k = int(options.get('clients', 3))
    if not 1 <= k <= n:
        raise ValueError('clients precisa estar entre 1 e o número de nós')
    if method in ('contiguous', 'random'):
        order = np.arange(n)
        if method == 'random':
            np.random.default_rng(seed).shuffle(order)
        labels = np.empty(n, dtype=np.int64)
        for label, nodes in enumerate(np.array_split(order, k)):
            labels[nodes] = label
    elif method == 'leiden':
        import igraph as ig
        import leidenalg as la
        sym = np.maximum(adj, adj.T)
        u, v = np.nonzero(np.triu(sym, 1))
        graph = ig.Graph(n=n, edges=list(zip(u.tolist(), v.tolist())), directed=False)
        partition = la.find_partition(graph, la.RBConfigurationVertexPartition,
            weights=sym[u, v].tolist(), resolution_parameter=float(options.get('resolution', 1)), seed=seed)
        labels = np.asarray(partition.membership)
    elif method == 'file':
        labels = np.load(Path(options['path']), allow_pickle=False)
        if labels.shape != (n,) or not np.issubdtype(labels.dtype, np.integer) or np.any(labels < 0):
            raise ValueError('Partição deve conter um inteiro não negativo por nó, na ordem da série')
    else:
        raise ValueError(f'Agrupamento desconhecido: {method}')
    return np.unique(labels, return_inverse=True)[1].astype(np.int64)


def save_partition(data, labels, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    np.save(directory / 'client_assignments.npy', labels)
    np.save(directory / 'node_ids.npy', np.asarray(data.node_ids, dtype=str))
    k = len(np.unique(labels))
    (directory / 'metadata.json').write_text(json.dumps({'num_clients': k}, indent=2)+'\n')
    for c in range(k):
        nodes = np.flatnonzero(labels == c)
        sub = data.adj[np.ix_(nodes, nodes)]
        src, dst = np.nonzero(sub)
        out = directory / f'client_{c:03d}'
        out.mkdir(exist_ok=True)
        np.save(out / 'node_indices.npy', nodes)
        np.save(out / 'edge_index.npy', np.stack([src, dst]).astype(np.int64))
        np.save(out / 'edge_weight.npy', sub[src, dst].astype(np.float32))
    return k
