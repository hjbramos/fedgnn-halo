"""LEConv com peso de aresta derivado dos atributos operacionais (porte do sunt-fedstgnn).

Motivação: no SUNT a topologia `transit` com peso fixo piora a previsão — nem `log1p(contagem)` nem
peso uniforme batem "sem arestas" (ver README). Ambos são pesos constantes no tempo, e a
representação oficial do dataset traz atributos que variam com a hora da semana: tempo de segmento e
velocidade derivada por aresta.

Duas regras, ambas SEM parâmetros treináveis (== EdgeDynamicLEConvForecaster de sunt-fedstgnn):

    score = (−distância − tempo_médio + velocidade_média) / 3
    peso  = softplus(score) + 1e-8

com os atributos padronizados em log1p sobre o período de treino. Arestas curtas, rápidas e de
percurso breve pesam mais.

  fixed_calendar  médias sobre as H horas previstas → UM peso por aresta e por janela. Com H = 24 a
                  média cobre um dia inteiro, por isso o que varia entre janelas é sobretudo o dia da
                  semana, não a hora do dia: não representa "hora de ponta vs fora dela" dentro da
                  janela, e o peso não reage ao tráfego corrente — depende do calendário e dos perfis
                  históricos do treino.
  fixed_constant  a MÉDIA DAS AFINIDADES das 168 origens possíveis da semana → peso constante,
                  pré-computado uma vez. É o CONTROLO: comparar `fixed_calendar` só com
                  `log1p(contagem)` e peso uniforme mudaria duas variáveis ao mesmo tempo (que
                  informação se usa E se ela varia no tempo).
                  Atenção à ordem das operações: é `mean_o(softplus(score_o))`, não
                  `softplus(mean_o(score_o))`. Como o softplus é não-linear e as máscaras pesam cada
                  origem de forma diferente, as duas divergem (medido: até 0.24 no peso normalizado,
                  2394 das 9232 arestas acima de 0.001). Só a primeira é "a afinidade dinâmica
                  congelada na sua média", que é o que o controlo precisa de ser.

Como o peso é função apenas dos dados, continua pré-computável. A variante `mlp` do projeto original
quebraria isso: o peso passaria a depender de uma rede em treino, logo o agregado de fronteira teria
de ser recalculado e retransmitido a cada rodada. (A identidade algébrica da LEConv sobreviveria se o
agregado fosse recalculado com os pesos vigentes — o que se perde é a troca única, não a exatidão.)

Arestas sem atributos recebem `fallback_weight`. São 100 no SUNT: 92 tocam os 21 nós que a
representação oficial descarta por falta de coordenadas, e 8 ligam nós com coordenadas mas foram
removidas por tempo de segmento ou velocidade implausíveis — ou seja, `fallback_weight` também dá
peso a ligações sem suporte operacional válido.

Limitação: o halo de fronteira (experiments/fedgnnhet.py) **não** suporta este modelo. O agregado
externo teria de ser ponderado pelos mesmos pesos da janela,
`a_v^(o)(t) = Σ_u w_uv^(o)·x_u(t) / Σ_u w_uv^(o)`, com a máscara e o peso da aresta virtual a
acompanhar. É pré-computável, mas o halo atual usa os pesos fixos da adjacência e não pode ser
reutilizado como está.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import LEConv as PyGLEConv


class LEConvEdgeDynamic(nn.Module):
    """Uma LEConv sobre a janela inteira, com pesos de aresta recalculados por janela."""

    def __init__(self, edge_index, edge_static, edge_profiles, edge_profile_mask, edge_scaler_mean,
                 edge_scaler_std, has_attributes, input_steps, hidden_size, horizon, node_features=1,
                 fallback_weight: float = 1.0, rule: str = "fixed_calendar"):
        super().__init__()
        if rule not in ("fixed_calendar", "fixed_constant"):
            raise ValueError(f"rule deve ser fixed_calendar ou fixed_constant, não {rule!r}")
        self.rule = rule
        self.register_buffer("edge_index", edge_index, persistent=False)
        self.register_buffer("edge_static", edge_static, persistent=False)          # (E, 1)
        self.register_buffer("edge_profiles", edge_profiles, persistent=False)      # (168, E, 2)
        self.register_buffer("edge_profile_mask", edge_profile_mask, persistent=False)
        self.register_buffer("scaler_mean", edge_scaler_mean, persistent=False)     # (3,)
        self.register_buffer("scaler_std", edge_scaler_std, persistent=False)
        self.register_buffer("has_attributes", has_attributes, persistent=False)    # (E,) bool
        self.horizon = int(horizon)
        self.fallback_weight = float(fallback_weight)
        self.convolution = PyGLEConv(2 * int(input_steps) * int(node_features), int(hidden_size))
        self.activation = nn.ReLU()
        self.output = nn.Linear(int(hidden_size), self.horizon)
        if rule == "fixed_constant":
            # média das afinidades das 168 origens possíveis (== sunt-fedstgnn), calculada uma vez:
            # é constante no treino, e expandi-la a cada minibatch custaria ~400 MB por batch de 32.
            with torch.no_grad():
                origins = torch.arange(168)
                indices = (origins[:, None] + torch.arange(self.horizon)[None, :]) % 168   # (168, H)
                self.register_buffer("constant_affinity", self.affinity(indices).mean(dim=0), persistent=False)

    def affinity(self, profile_indices: torch.Tensor) -> torch.Tensor:
        """(B, E) afinidade bruta (antes do fallback e da normalização) para janelas com estes alvos."""
        profiles = self.edge_profiles[profile_indices]                      # (B, H, E, 2)
        masks = self.edge_profile_mask[profile_indices].float()
        temporal = (torch.log1p(profiles) - self.scaler_mean[None, None, None, 1:]) / self.scaler_std[None, None, None, 1:]
        observed = torch.where(masks.bool(), temporal, torch.zeros_like(temporal))
        means = observed.sum(dim=1) / masks.sum(dim=1).clamp_min(1)          # (B, E, 2)
        distance = (torch.log1p(self.edge_static) - self.scaler_mean[0]) / self.scaler_std[0]
        score = (-distance.squeeze(-1)[None, :] - means[..., 0] + means[..., 1]) / 3
        return F.softplus(score) + 1e-8

    def edge_weights(self, profile_indices: torch.Tensor, nodes: int) -> torch.Tensor:
        """(B, E) normalizado por nó destino. `profile_indices` (B, H) = hora da semana de cada alvo."""
        batch = profile_indices.shape[0]
        if self.rule == "fixed_constant":
            weights = self.constant_affinity[None, :].expand(batch, -1)      # pré-computado no __init__
        else:
            weights = self.affinity(profile_indices)
        weights = torch.where(self.has_attributes[None, :], weights,
                              torch.full_like(weights, self.fallback_weight))
        incoming = torch.zeros((batch, nodes), dtype=weights.dtype, device=weights.device)
        target = self.edge_index[1]
        incoming.scatter_add_(1, target[None, :].expand(batch, target.numel()), weights)
        if not torch.isfinite(weights).all() or torch.any(incoming[:, target] <= 0):
            raise ValueError("pesos de aresta devem ser finitos e somar > 0 em cada nó destino")
        return weights / incoming[:, target]

    def spatial_embedding(self, x, x_mask, y_how):
        batch_size, steps, nodes, _ = x.shape
        values = x.permute(0, 2, 1, 3).reshape(batch_size * nodes, -1)
        masks = x_mask.float().permute(0, 2, 1, 3).reshape(batch_size * nodes, -1)
        features = torch.cat([values, masks], dim=-1)
        offsets = torch.arange(batch_size, device=x.device) * nodes
        batch_edges = (self.edge_index[:, None, :] + offsets[None, :, None]).reshape(2, -1)
        batch_weights = self.edge_weights(y_how, nodes).reshape(-1)          # (B·E,)
        spatial = self.activation(self.convolution(features, batch_edges, batch_weights))
        return spatial.reshape(batch_size, nodes, -1)

    def forward(self, x, x_mask, y_how):
        batch_size, steps, nodes, _ = x.shape
        spatial = self.spatial_embedding(x, x_mask, y_how).reshape(batch_size * nodes, -1)
        prediction = self.output(spatial).reshape(batch_size, nodes, self.horizon)
        return prediction.permute(0, 2, 1).unsqueeze(-1)
