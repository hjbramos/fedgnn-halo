"""LEConv — cópia do modelo centralizado de sunt-fedstgnn/src/fedstgnn/models.py (sem fronteira)."""
from __future__ import annotations

import torch
from torch import nn
from torch_geometric.nn import LEConv as PyGLEConv


def normalize_incoming_edge_weights(edge_index: torch.Tensor, edge_weight: torch.Tensor, nodes: int,
                                    allow_no_incoming: bool = False) -> torch.Tensor:
    """Divide cada peso pela soma dos pesos que entram no nó destino.

    allow_no_incoming=True aceita nós sem aresta de entrada (grafo direcionado; no federado, também nós cuja
    única entrada vem de outro cliente): eles não são destino de nenhuma aresta, então a mensagem dos
    vizinhos é zero e a LEConv usa só o termo do próprio nó. Nenhuma aresta é criada. O padrão continua
    exigindo entrada em todo nó, como antes.
    """
    if not torch.isfinite(edge_weight).all() or torch.any(edge_weight < 0):
        raise ValueError("Edge weights must be finite and nonnegative")
    incoming = torch.zeros(nodes, dtype=edge_weight.dtype, device=edge_weight.device)
    incoming.index_add_(0, edge_index[1], edge_weight)
    if not allow_no_incoming and not torch.all(incoming > 0):
        raise ValueError("Every node must have at least one incoming edge")
    if not torch.all(incoming[edge_index[1]] > 0):
        raise ValueError("Edge with zero weight sum at its destination")
    return edge_weight / incoming[edge_index[1]]


class LEConv(nn.Module):
    """Uma LEConv sobre a janela inteira (valores ‖ máscara) + cabeça linear para o horizonte."""

    def __init__(self, edge_index, edge_weight, input_steps, hidden_size, horizon, node_features=1):
        super().__init__()
        self.register_buffer("edge_index", edge_index, persistent=False)
        self.register_buffer("edge_weight", edge_weight, persistent=False)
        self.horizon = int(horizon)
        self.convolution = PyGLEConv(2 * int(input_steps) * int(node_features), int(hidden_size))
        self.activation = nn.ReLU()
        self.output = nn.Linear(int(hidden_size), self.horizon)

    def spatial_preactivation(self, x, x_mask):
        """Saída da convolução ANTES da ReLU — (B, N, hidden).

        É o ponto que o FED-PUB oficial lê para o embedding funcional (`is_proxy` devolve a segunda
        convolução antes da ativação). A previsão também passa por aqui, via `spatial_embedding`, que
        aplica a ReLU; ler este ponto para o embedding não altera a previsão.
        """
        batch_size, steps, nodes, _ = x.shape
        values = x.permute(0, 2, 1, 3).reshape(batch_size * nodes, -1)          # (B·N, W·F)
        masks = x_mask.float().permute(0, 2, 1, 3).reshape(batch_size * nodes, -1)
        features = torch.cat([values, masks], dim=-1)
        offsets = torch.arange(batch_size, device=x.device) * nodes
        batch_edges = (self.edge_index[:, None, :] + offsets[None, :, None]).reshape(2, -1)
        batch_weights = self.edge_weight.repeat(batch_size)
        return self.convolution(features, batch_edges, batch_weights).reshape(batch_size, nodes, -1)

    def spatial_embedding(self, x, x_mask):
        """Representação pós-convolução e pós-ReLU, ANTES da cabeça linear — (B, N, hidden).

        É a entrada da cabeça de previsão e o embedding FED-PUB com `similarity_source: representation`.
        """
        return self.activation(self.spatial_preactivation(x, x_mask))

    def forward(self, x, x_mask):
        batch_size, steps, nodes, _ = x.shape
        spatial = self.spatial_embedding(x, x_mask).reshape(batch_size * nodes, -1)
        prediction = self.output(spatial).reshape(batch_size, nodes, self.horizon)
        return prediction.permute(0, 2, 1).unsqueeze(-1)
