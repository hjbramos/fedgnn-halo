"""Adaptação da previsão de tráfego: configuração, batches, ablações e MAE mascarada."""
import torch
from torch import nn
from models import DCRNN, LEConvEdgeDynamic, RNNSeq2Seq, build_model, ModelSpec, EdgeAttributes
from federation import ObjectiveResult


def cut_partition_edges(data, partition: str, edge_index, edge_weight):
    """Controle centralizado com os cortes de uma partição: mantém só as arestas entre nós do mesmo cliente.

    Os pesos mantidos são os brutos do grafo; a normalização por destino (`build_model`) acontece depois,
    sobre as arestas que restam — a mesma operação que cada cliente do FedAvg faz no seu subgrafo.
    """
    from pathlib import Path

    import numpy as np

    pdir = Path(__file__).resolve().parents[1] / partition
    ids = np.load(pdir / "node_ids.npy", allow_pickle=True).astype(str).tolist()
    if ids != list(data.node_ids):
        raise ValueError(f"{partition}: ordem dos nós da partição difere da série")
    assign = torch.from_numpy(np.load(pdir / "client_assignments.npy")).to(edge_index.device)
    keep = assign[edge_index[0]] == assign[edge_index[1]]
    return edge_index[:, keep], edge_weight[keep]


def model_spec(data, options, name):
    """Resolve ablações e atributos do domínio antes de entrar na camada de modelos."""
    edge_index, edge_weight = data.edge_index, data.edge_weight
    attrs = None
    if name == "leconv":
        if bool(options.get("self_loops_only", False)):
            loop = torch.arange(data.nodes, device=edge_index.device)
            edge_index = torch.stack([loop, loop])
            edge_weight = torch.ones(data.nodes, device=edge_weight.device)
        elif options.get("edge_weight_rule", "dataset") == "uniform":
            edge_weight = torch.ones_like(edge_weight)
        if options.get("cut_partition"):
            if bool(options.get("self_loops_only", False)):
                raise ValueError("cut_partition e self_loops_only são controles distintos; use um de cada vez")
            edge_index, edge_weight = cut_partition_edges(data, options["cut_partition"], edge_index, edge_weight)
    if name == "leconv_edge":
        from data_loader.sunt_edge_attributes import load_edge_attributes
        raw = load_edge_attributes(data.node_ids, edge_index.cpu().numpy(), options.get("edge_attributes_root"))
        attrs = EdgeAttributes(**{key: torch.as_tensor(value, dtype=torch.bool if key in ("mask", "has_attributes")
                                                       else torch.float32) for key, value in raw.items()
                                  if key in EdgeAttributes.__dataclass_fields__})
    return ModelSpec(data.nodes, data.input_steps, data.horizon, data.n_features,
                     edge_index, edge_weight, data.adj, attrs)


def make_model(cfg: dict, data, device) -> nn.Module:
    m = dict(cfg["model"])
    name = m.pop("name", "leconv")
    m["gradient_checkpointing"] = bool(m.pop("gradient_checkpointing_per_dataset", {}).get(data.name, False))
    return build_model(name, model_spec(data, m, name), **m).to(device)


def forward(model, batch, device, teacher=False, batches_seen=0):
    x, x_mask = batch["x"].to(device, non_blocking=True), batch["x_mask"].to(device, non_blocking=True)
    if isinstance(model, (DCRNN, RNNSeq2Seq)):
        if isinstance(model, DCRNN):
            return model(x, batch["x_tod"].to(device), batch["y"].to(device) if teacher else None,
                         batch["y_tod"].to(device), batches_seen, x_mask=x_mask)
        return model(x, batch["x_tod"].to(device), batch["y"].to(device) if teacher else None,
                     batch["y_tod"].to(device), batches_seen)
    if isinstance(model, LEConvEdgeDynamic):
        return model(x, x_mask, batch["y_how"].to(device))
    return model(x, x_mask)


def masked_mae_objective(device, predict=forward, target_scale=None):
    """MAE só nos alvos observados (y_mask). target_scale=None: escala padronizada (y). target_scale=(média, desvio)
    do alvo: a previsão volta à escala original e a perda é contra y_raw, como na DCRNN de referência (masked_mae_loss
    sobre o inverse_transform). Sem clamp: o recorte em zero é só das métricas."""
    def objective(model, batch, batches_seen):
        mask = batch["y_mask"].to(device, non_blocking=True)
        count = int(mask.sum().item())
        if count == 0:
            return None
        prediction = predict(model, batch, device, teacher=True, batches_seen=batches_seen)
        if target_scale is None:
            y = batch["y"].to(device, non_blocking=True)
        else:
            y = batch["y_raw"].to(device, non_blocking=True)
            prediction = prediction * float(target_scale[1]) + float(target_scale[0])
        errors = torch.abs(prediction - y)[mask]
        return ObjectiveResult(errors.mean(), errors.detach().sum().item(), count)
    return objective
