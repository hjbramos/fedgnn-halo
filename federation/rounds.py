"""Rodadas síncronas com modelo, otimizador e treino local fornecidos pelo chamador.

Clientes são identificadores/objetos opacos. Pesos e métricas são escolhas do chamador.
Nenhuma rodada lê dados, avalia splits, escreve arquivos ou controla early stopping.
"""
import hashlib

import torch

from .aggregation import aggregate_states, cpu_state


def fedavg_round(global_state, clients, weights, make_model, make_optimizer, train_epoch,
                 local_epochs, batches_seen=0, optimizer_states=None):
    """train_epoch(model, client, optimizer, anchors, step) -> (metric, step).

    Cada cliente começa no estado global; o otimizador é recriado por cliente/rodada.
    O callback pode usar anchors para FedProx ou ignorá-las para FedAvg.

    optimizer_states (opcional, uma entrada por cliente): estado do otimizador de cada cliente preservado entre
    rodadas. Os parâmetros continuam vindo do estado global; só os momentos e contadores do otimizador do
    cliente são restaurados antes do treino local, e a lista é atualizada no lugar ao fim. None (padrão)
    mantém o comportamento original, com o otimizador reiniciado a cada rodada.
    """
    if not clients or len(clients) != len(weights) or local_epochs < 1 or sum(weights) <= 0:
        raise ValueError("rodada requer clientes, pesos correspondentes e épocas positivas")
    if optimizer_states is not None and len(optimizer_states) != len(clients):
        raise ValueError("optimizer_states precisa de uma entrada por cliente")
    states, error, observations = [], 0.0, 0
    for i, (client, weight) in enumerate(zip(clients, weights)):
        model = make_model(client)
        model.load_state_dict(global_state, strict=True)
        anchors = [p.detach().clone() for p in model.parameters()]
        optimizer = make_optimizer(model)
        if optimizer_states is not None and optimizer_states[i] is not None:
            optimizer.load_state_dict(optimizer_states[i])
            check_restored(optimizer, optimizer_states[i])
        for _ in range(local_epochs):
            metric, batches_seen = train_epoch(model, client, optimizer, anchors, batches_seen)
            error += metric * weight
            observations += weight
        states.append(cpu_state(model))
        if optimizer_states is not None:
            optimizer_states[i] = cpu_optimizer_state(optimizer)
        del model, optimizer, anchors
    return aggregate_states(states, weights), error / observations, batches_seen


def check_restored(optimizer, saved: dict):
    """Confere que o estado efetivamente carregado no otimizador é o salvo (mesmas chaves e valores, finitos)."""
    loaded = optimizer.state_dict()["state"]
    if set(loaded) != set(saved["state"]):
        raise RuntimeError("estado do otimizador restaurado com parâmetros diferentes dos salvos")
    for k, st in saved["state"].items():
        if set(loaded[k]) != set(st):
            raise RuntimeError("estado do otimizador restaurado com campos diferentes dos salvos")
        for n, v in st.items():
            got = loaded[k][n]
            if not hasattr(v, "detach"):
                if got != v:
                    raise RuntimeError(f"campo {n} do otimizador difere do salvo")
                continue
            got = got.detach().to(v.device)
            if not torch.isfinite(v).all() or not torch.equal(got.to(v.dtype), v):
                raise RuntimeError(f"tensor {n} do otimizador não finito ou diferente do salvo")


def optimizer_states_sha256(states) -> str:
    """Hash determinístico dos estados por cliente (cliente, parâmetro, campo, bytes)."""
    d = hashlib.sha256()
    for c, st in enumerate(states):
        if st is None:
            d.update(f"{c}:None".encode())
            continue
        for k in sorted(st["state"]):
            for n in sorted(st["state"][k]):
                v = st["state"][k][n]
                d.update(f"{c}:{k}:{n}".encode())
                d.update(v.detach().cpu().contiguous().numpy().tobytes() if hasattr(v, "detach") else repr(v).encode())
    return d.hexdigest()


def cpu_optimizer_state(optimizer) -> dict:
    """Cópia independente do state_dict do otimizador, com tensores em CPU."""
    sd = optimizer.state_dict()
    return {"state": {k: {n: (v.detach().cpu().clone() if hasattr(v, "detach") else v) for n, v in s.items()}
                      for k, s in sd["state"].items()},
            "param_groups": [dict(g) for g in sd["param_groups"]]}


def ditto_round(global_state, personal, clients, weights, make_model, make_optimizer,
                train_global, train_personal, local_epochs, batches_seen=0, personal_seen=0):
    """Atualiza w por FedAvg e v_k por candidato, ancorado em w recebido.

    Preserva a ordem global/pessoal por cliente e os contadores separados.
    train_personal recebe (model, client, optimizer, anchors, candidate, step).
    Os estados pessoais retornados substituem os anteriores, sem alterá-los in-place.
    """
    if (not clients or len(clients) != len(weights) or len(personal) != len(clients)
            or any(not p for p in personal) or local_epochs < 1 or sum(weights) <= 0):
        raise ValueError("rodada Ditto requer clientes, candidatos, pesos e épocas válidos")
    states, updated, err_g, err_p, observations = [], [], 0.0, 0.0, 0
    for client, weight, candidates in zip(clients, weights, personal):
        model = make_model(client)
        model.load_state_dict(global_state, strict=True)
        anchors = [p.detach().clone() for p in model.parameters()]
        optimizer = make_optimizer(model)
        for _ in range(local_epochs):
            metric, batches_seen = train_global(model, client, optimizer, anchors, batches_seen)
            err_g += metric * weight
        states.append(cpu_state(model))
        del model, optimizer
        local = {}
        for candidate, state in candidates.items():
            model = make_model(client)
            model.load_state_dict(state, strict=True)
            optimizer = make_optimizer(model)
            for _ in range(local_epochs):
                metric, personal_seen = train_personal(model, client, optimizer, anchors, candidate, personal_seen)
                err_p += metric * weight / len(candidates)
            local[candidate] = cpu_state(model)
            del model, optimizer
        updated.append(local)
        observations += weight * local_epochs
    return aggregate_states(states, weights), updated, err_g / observations, err_p / observations, batches_seen, personal_seen
