"""Instrumentação do treino centralizado (opt-in, `training.instrumentation: true`). Só observa.

A época instrumentada executa exatamente as mesmas operações de `federation.train_local_epoch`, na mesma ordem e com os
mesmos argumentos: zero_grad, objetivo, backward, `clip_grad_norm_(model.parameters(), clip)`, `optimizer.step()`. As
medições são feitas sob `torch.no_grad()`, não consomem números aleatórios e não alteram os lotes. A equivalência é
verificada em `tests/test_sunt_montevideo.py::Instrumentacao`.

Por passo (`instrumentacao/passos.csv`):
- perda do lote e número de alvos;
- norma global do gradiente antes e depois do clipping, e se o clipping atuou;
- gradiente² por grupo de parâmetros (antes do clipping): em cada DCGRU, porta de reset (r), porta de atualização (u) e
  candidato (c); projeção;
- norma dos parâmetros (global e `encoder.0`); norma da atualização efetivamente aplicada (global e por grupo) e a razão
  atualização/parâmetros;
- lr do agendador; mediana do passo efetivo por parâmetro do Adam, lr/(√v̂ + eps), e fração com √v̂ < eps;
- magnitude das previsões (escala original): máximo |previsão|, média |previsão| nos alvos, fração negativa;
- não finitos (perda, gradiente, parâmetros, previsão); hash do lote (conferência da sequência de lotes);
- tempo gasto nas medições.

Resumo por época (`instrumentacao/epocas.csv`): mediana, p90, p99 e máximo das grandezas por passo. Em um valor não
finito o passo e a época parcial são gravados, o estado é salvo em `instrumentacao/nao_finito.{pt,json}` e o treino para antes
do `optimizer.step()`.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from .task import forward, masked_mae_objective


class NaoFinito(RuntimeError):
    pass


def grupos(model: nn.Module) -> dict[str, list[tuple[torch.nn.Parameter, slice | None]]]:
    """Grupos de parâmetros: por célula DCGRU, r/u (metades da saída da porta) e c (candidato); projeção; resto."""
    g: dict[str, list] = {}
    vistos = set()
    for nome_mod, mod in model.named_modules():
        if hasattr(mod, "gate") and hasattr(mod, "candidate") and hasattr(mod, "hidden_dim"):
            H = mod.hidden_dim
            for p in (mod.gate.weight, mod.gate.bias):
                g.setdefault(f"{nome_mod}.r", []).append((p, slice(0, H)))
                g.setdefault(f"{nome_mod}.u", []).append((p, slice(H, 2 * H)))
                vistos.add(id(p))
            for p in (mod.candidate.weight, mod.candidate.bias):
                g.setdefault(f"{nome_mod}.c", []).append((p, None))
                vistos.add(id(p))
    for nome, p in model.named_parameters():
        if id(p) not in vistos:
            g.setdefault(nome.split(".")[0], []).append((p, None))
    return g


def _sq(t: torch.Tensor, s: slice | None) -> torch.Tensor:
    return (t[..., s] if s is not None else t).double().square().sum()


def _hash_lote(batch) -> str:
    h = hashlib.sha1()
    for k in ("x", "x_mask", "y_raw", "y_mask"):
        h.update(batch[k].numpy().tobytes())
    return h.hexdigest()[:16]


QUANTIS = {"mediana": 0.5, "p90": 0.9, "p99": 0.99}
RESUMO = ("perda", "norma_grad_antes_clip", "norma_grad_depois_clip", "norma_atualizacao", "razao_atualizacao_parametros",
          "norma_atualizacao_encoder.0", "passo_efetivo_adam_mediana", "frac_sqrt_vhat_menor_eps",
          "max_abs_previsao", "media_abs_previsao_alvos")


class Instrumentacao:
    def __init__(self, model: nn.Module, out: Path, start_epoch: int):
        self.dir = Path(out) / "instrumentacao"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.params = list(model.parameters())
        self.grupos = grupos(model)
        self.enc0 = [p for nome, p in model.named_parameters() if nome.startswith("encoder.0.")]
        # colunas medidas depois do passo (ordem fixa: as linhas de todas as épocas são anexadas ao mesmo CSV)
        self.pos = ["norma_parametros", "norma_parametros_encoder.0", "norma_atualizacao", "norma_atualizacao_encoder.0",
                    "razao_atualizacao_parametros", "passo_efetivo_adam_mediana", "frac_sqrt_vhat_menor_eps",
                    *[f"u2_{k}" for k in self.grupos]]
        for f in ("passos.csv", "epocas.csv"):          # retomada: descarta épocas que serão refeitas
            p = self.dir / f
            if p.exists():
                d = pd.read_csv(p)
                d[d.epoca < start_epoch].to_csv(p, index=False)

    def _grava(self, linhas: list[dict], epoca: int) -> pd.DataFrame:
        R = pd.DataFrame(linhas)
        p = self.dir / "passos.csv"
        R.to_csv(p, mode="a", header=not p.exists(), index=False)
        pn = R.norma_parametros.dropna()
        res = {"epoca": epoca, "passos": len(R), "frac_clip": float(R.clip_atuou.mean()),
               "nao_finitos": int(R.nao_finito.sum()),
               "norma_parametros_inicio": float(pn.iloc[0]) if len(pn) else np.nan,
               "norma_parametros_fim": float(pn.iloc[-1]) if len(pn) else np.nan,
               "norma_parametros_encoder.0_fim": float(R["norma_parametros_encoder.0"].dropna().iloc[-1]) if len(pn) else np.nan,
               "lr": float(R.lr.iloc[-1]),
               "tempo_medicoes_s": float(R.tempo_medicoes_s.sum())}
        for c in RESUMO:
            v = R[c].astype(float)
            for nome, q in QUANTIS.items():
                res[f"{c}_{nome}"] = float(v.quantile(q))
            res[f"{c}_max"] = float(v.max())
        g2 = R[[c for c in R.columns if c.startswith("g2_")]]
        tot = g2.sum(axis=1).replace(0, np.nan)
        alto = R.norma_grad_antes_clip >= R.norma_grad_antes_clip.quantile(0.99)
        for c in g2.columns:                     # participação de cada grupo no gradiente² (antes do clipping)
            res[f"part_{c[3:]}_mediana"] = float((g2[c] / tot).median())
            res[f"part_{c[3:]}_top1pct"] = float(g2[c][alto].sum() / tot[alto].sum())
        E = pd.DataFrame([res])
        p = self.dir / "epocas.csv"
        E.to_csv(p, mode="a", header=not p.exists(), index=False)
        return E

    def epoca(self, epoca, model, loader, optimizer, device, clip, batches_seen=0, target_scale=None):
        """Mesmo laço de `federation.train_local_epoch`, com medições. Devolve (MAE de treino, batches_seen)."""
        guardado = {}

        def prever(m, b, d, teacher=False, batches_seen=0):
            guardado["p"] = forward(m, b, d, teacher, batches_seen)
            return guardado["p"]

        objective = masked_mae_objective(device, prever, target_scale)
        sync = torch.cuda.synchronize if torch.device(device).type == "cuda" else (lambda: None)
        mu, sd = (target_scale if target_scale is not None else (0.0, 1.0))
        group = optimizer.param_groups[0]
        eps, b2 = group["eps"], group["betas"][1]
        model.train()
        total, count, linhas, passo = 0.0, 0, [], 0
        for batch in loader:
            passo += 1
            optimizer.zero_grad()
            result = objective(model, batch, batches_seen)
            if result is None:
                continue
            loss = result.loss
            loss.backward()
            sync(); t = time.perf_counter()
            with torch.no_grad():
                g2 = {f"g2_{k}": float(sum(_sq(p.grad, s) for p, s in v if p.grad is not None))
                      for k, v in self.grupos.items()}
                antes = [p.detach().clone() for p in self.params]
            sync(); dt = time.perf_counter() - t
            g = float(nn.utils.clip_grad_norm_(model.parameters(), clip))
            sync(); t = time.perf_counter()
            with torch.no_grad():
                g_depois = float(torch.sqrt(sum(p.grad.double().square().sum() for p in self.params if p.grad is not None)))
                pred = guardado["p"].detach() * float(sd) + float(mu)
                m = batch["y_mask"].to(device)
                nf = {"perda": not bool(torch.isfinite(loss)), "gradiente": not np.isfinite(g),
                      "previsao": not bool(torch.isfinite(pred).all())}
                linha = {"epoca": epoca, "passo_epoca": passo, "passo_global": batches_seen + 1, "lote": _hash_lote(batch),
                         "alvos": result.count, "perda": float(loss), "norma_grad_antes_clip": g,
                         "norma_grad_depois_clip": g_depois, "clip_atuou": g > clip, "lr": float(group["lr"]),
                         "max_abs_previsao": float(pred.abs().max()), "media_abs_previsao_alvos": float(pred[m].abs().mean()),
                         "frac_previsao_negativa_alvos": float((pred[m] < 0).double().mean()), **g2,
                         **dict.fromkeys(self.pos, np.nan), "nao_finito": False, "tempo_medicoes_s": np.nan}
            sync(); dt += time.perf_counter() - t
            if nf["perda"] or nf["gradiente"] or nf["previsao"]:
                self._parar(epoca, linhas, linha, nf, dt, model, optimizer, batches_seen)
            optimizer.step()
            sync(); t = time.perf_counter()
            with torch.no_grad():
                pn = float(torch.sqrt(sum(p.double().square().sum() for p in self.params)))
                ant = {id(p): a for p, a in zip(self.params, antes)}
                du2 = {k: float(sum(_sq(p - ant[id(p)], s) for p, s in v)) for k, v in self.grupos.items()}
                du = float(np.sqrt(sum(du2.values())))
                pn0 = float(torch.sqrt(sum(p.double().square().sum() for p in self.enc0)))
                du0 = float(torch.sqrt(sum((p - ant[id(p)]).double().square().sum() for p in self.enc0)))
                vhat = torch.cat([(optimizer.state[p]["exp_avg_sq"] / (1 - b2 ** float(optimizer.state[p]["step"]))).sqrt()
                                  .flatten() for p in self.params])
                nf["parametros"] = not np.isfinite(pn)
                linha.update({"norma_parametros": pn, "norma_parametros_encoder.0": pn0, "norma_atualizacao": du,
                              "norma_atualizacao_encoder.0": du0, "razao_atualizacao_parametros": du / pn,
                              "passo_efetivo_adam_mediana": float((group["lr"] / (vhat + eps)).median()),
                              "frac_sqrt_vhat_menor_eps": float((vhat < eps).double().mean()),
                              **{f"u2_{k}": v for k, v in du2.items()}, "nao_finito": any(nf.values())})
            sync(); dt += time.perf_counter() - t
            linha["tempo_medicoes_s"] = dt
            linhas.append(linha)
            if nf["parametros"]:
                self._parar(epoca, linhas[:-1], linha, nf, dt, model, optimizer, batches_seen)
            total += result.metric_sum
            count += result.count
            batches_seen += 1
        if count == 0:
            raise ValueError("época local sem alvos observados; verifique o objetivo e os dados do cliente")
        self._grava(linhas, epoca)
        return total / count, batches_seen

    def _parar(self, epoca, linhas, linha, nf, dt, model, optimizer, batches_seen):
        linha.update({"nao_finito": True, "tempo_medicoes_s": dt})
        self._grava(linhas + [linha], epoca)
        (self.dir / "nao_finito.json").write_text(json.dumps({"epoca": epoca, "passo_global": batches_seen + 1, **nf}))
        torch.save({"epoca": epoca, "passo_global": batches_seen + 1, "nao_finito": nf,
                    "model_state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
                    "optimizer_state_dict": optimizer.state_dict()}, self.dir / "nao_finito.pt")
        raise NaoFinito(f"valor não finito na época {epoca}, passo global {batches_seen}: {nf}")
