"""Arquiteturas espaço-temporais; build_model recebe um ModelSpec com entradas explícitas."""
from __future__ import annotations

from torch import nn
from .spec import ModelSpec, EdgeAttributes

from .dcrnn import DCRNN
from .leconv import LEConv, normalize_incoming_edge_weights
from .leconv_edge import LEConvEdgeDynamic
from .rnn import RNNSeq2Seq

MODELS = ("leconv", "leconv_edge", "dcrnn", "gru", "lstm")


def build_model(name: str, data: ModelSpec, **kw) -> nn.Module:
    if name == "leconv":
        edge_index, ew = data.edge_index, data.edge_weight
        edge_weight = normalize_incoming_edge_weights(edge_index, ew, data.nodes,
                                                      bool(kw.get("allow_nodes_without_incoming", False)))
        return LEConv(edge_index, edge_weight, data.input_steps, kw["hidden_size"], data.horizon,
                                node_features=data.n_features)
    if name == "leconv_edge":
        attrs = data.edge_attributes
        if attrs is None:
            raise ValueError("leconv_edge requer ModelSpec.edge_attributes fornecido pelo chamador")
        return LEConvEdgeDynamic(
            data.edge_index, attrs.static, attrs.profiles, attrs.mask,
            attrs.scaler_mean, attrs.scaler_std, attrs.has_attributes,
            data.input_steps, kw["hidden_size"], data.horizon, node_features=data.n_features,
            fallback_weight=float(kw.get("fallback_weight", 1.0)),
            rule=kw.get("edge_rule", "fixed_calendar"))
    if name == "dcrnn":
        if data.adj is None:
            raise ValueError("dcrnn requer ModelSpec.adj")
        masks, tod = bool(kw.get("input_masks", False)), bool(kw.get("time_of_day", True))
        return DCRNN(
            data.adj, input_dim=data.n_features * (2 if masks else 1) + int(tod), output_dim=1, hidden_dim=int(kw["hidden_size"]),
            num_layers=int(kw["num_layers"]), max_diffusion_step=int(kw["max_diffusion_step"]), horizon=data.horizon,
            cl_decay_steps=int(kw["cl_decay_steps"]), use_curriculum_learning=bool(kw.get("use_curriculum_learning", True)),
            gradient_checkpointing=bool(kw.get("gradient_checkpointing", False)),
            input_masks=masks, time_of_day=tod, no_messages=bool(kw.get("no_messages", False)),
        )
    if name in ("gru", "lstm"):
        return RNNSeq2Seq(
            input_dim=data.n_features + 1, output_dim=1, hidden_dim=int(kw["hidden_size"]),
            num_layers=int(kw.get("num_layers", 1)), horizon=data.horizon, cell=name,
            cl_decay_steps=int(kw.get("cl_decay_steps", 2000)),
            use_curriculum_learning=bool(kw.get("use_curriculum_learning", True)),
        )
    raise ValueError(f"modelo desconhecido: {name!r}; opções: {MODELS}")


__all__ = ["ModelSpec", "EdgeAttributes", "build_model", "MODELS", "LEConv", "LEConvEdgeDynamic", "DCRNN", "RNNSeq2Seq", "normalize_incoming_edge_weights"]
