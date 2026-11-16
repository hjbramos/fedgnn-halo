"""Mecanismos federados independentes de modelos, datasets e campanhas."""
from .aggregation import aggregate_states, cpu_state
from .rounds import fedavg_round, ditto_round
from .optimization import ObjectiveResult, train_local_epoch, proximal_penalty

__all__ = ["aggregate_states", "cpu_state", "fedavg_round", "ditto_round",
           "ObjectiveResult", "train_local_epoch", "proximal_penalty"]
