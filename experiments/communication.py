"""Auditoria de comunicação: recalcula cada componente a partir das dimensões e confronta com o gravado.

Não treina nada e não precisa de GPU — lê `metrics.json`, as partições e o cabeçalho dos arquivos de
dados. Implementa o contrato descrito no README (secção "Contrato de comunicação do halo"): uma
verificação por linha, valor gravado contra valor recalculado, e o motivo de cada divergência.

O que é e o que não é:
  - São **payloads estimados** das dimensões do problema. NÃO são tráfego medido na rede, e não
    incluem cabeçalhos, serialização, identificadores, retransmissões nem mecanismos de proteção.
  - A coluna de exposição descreve o que o destinatário consegue observar. **Não é auditoria de
    privacidade**: documenta exposição potencial.

Verificações:
  modelo      2 × model_state_bytes × num_clients × rounds_completed
  halo 1hop   T × n_b × F × 4 × 2            (valores + máscara, float32, série inteira)
  halo static     n_b × F × 4 × 2            (uma cópia; o agregado é constante no tempo)
  exposição   nós virtuais com um único vizinho externo, no total e por par (cliente-fonte, alvo)

A verificação do modelo passa em todos os artefatos conhecidos e por isso NÃO detecta o erro
histórico do `halo_bytes`; é a verificação do halo que o apanha.

    python -m experiments.communication --config config/fedgnnhet_leconv.yaml
    python -m experiments.communication --runs results/fedgnnhet_leconv results/fedavg_leconv
"""
from __future__ import annotations

import argparse
import csv
import json
import zipfile
from pathlib import Path

import numpy as np

from .train import ROOT, load_config

ELEM = 4          # float32 nos agregados do halo
VAL_E_MASCARA = 2  # cada agregado viaja com a sua máscara


# ---------------------------------------------------------------- dimensões

def timesteps(dataset: str) -> int:
    """T do dataset, lido do cabeçalho — sem carregar a série."""
    from data_loader import DATA_DIR

    if dataset == "sunt":
        z = DATA_DIR / "sunt" / "processed" / "sunt_passenger_flows.npz"
        with zipfile.ZipFile(z) as zf, zf.open("features.npy") as fh:
            np.lib.format.read_magic(fh)
            shape, _, _ = np.lib.format.read_array_header_1_0(fh)
        return int(shape[0])
    import pyarrow.parquet as pq

    return pq.ParquetFile(DATA_DIR / dataset / "processed" / f"{dataset.replace('-', '_')}.parquet").metadata.num_rows


def exposicao(dataset: str, requested_clients: int, partition_seed: int) -> dict:
    """Nós virtuais com um único vizinho externo: no total e por par (cliente-fonte, alvo).

    O campo `halo_single_neighbour_pct` dos artefatos conta só a primeira forma. A segunda é a que
    vale se os agregados forem formados por somas parciais por cliente de origem.
    """
    from .federated import partition_dir
    from .train import load_experiment

    pdir = partition_dir(dataset, requested_clients, partition_seed)
    meta = json.loads((pdir / "metadata.json").read_text())
    assign = np.load(pdir / "client_assignments.npy")
    data = load_experiment(dataset, 1, 1)          # só precisamos de `adj`; janela mínima
    A, N = data.adj.astype(np.float64), data.nodes

    alvos = total_um = pares = pares_um = fontes = 0
    conjunto_fontes = set()
    for c in range(int(meta["num_clients"])):
        nodes = np.flatnonzero(assign == c)
        local = np.zeros(N, bool)
        local[nodes] = True
        ext = A[:, nodes].copy()
        ext[local, :] = 0.0
        b = np.flatnonzero(ext.sum(0) > 0)
        alvos += len(b)
        grau = (ext[:, b] > 0).sum(0)
        total_um += int((grau == 1).sum())
        conjunto_fontes.update(np.flatnonzero(ext[:, b].sum(1) > 0).tolist())
        for cf in range(int(meta["num_clients"])):
            if cf == c:
                continue
            m = np.zeros(N, bool)
            m[np.flatnonzero(assign == cf)] = True
            g = (ext[m][:, b] > 0).sum(0)
            pares += int((g > 0).sum())
            pares_um += int((g == 1).sum())
    fontes = len(conjunto_fontes)
    return {"alvos": alvos, "alvos_um_vizinho": total_um, "fontes_distintas": fontes,
            "parciais": pares, "parciais_um_vizinho": pares_um}


# ---------------------------------------------------------------- verificações

def linhas_de_uma_execucao(f: Path) -> list[dict]:
    """Uma linha por componente de comunicação, com gravado, recalculado e motivo."""
    d = json.loads(f.read_text())
    ds, halo = d["dataset"], d.get("halo")
    nc, rounds = d.get("num_clients"), d.get("rounds_completed")
    linhas = []

    if nc and rounds and "model_communication_bytes" in d:
        esperado = 2 * d["model_state_bytes"] * nc * rounds
        # O FED-PUB não usa a fórmula densa: grava contabilização real (download denso + upload
        # esparso + embedding) e guarda o equivalente FedAvg em `dense_communication_bytes`.
        fedpub = d.get("method", "").startswith("fedpub")
        gravado = d["dense_communication_bytes"] if fedpub else d["model_communication_bytes"]
        linhas.append({
            "execucao": str(f.parent.relative_to(ROOT / "results")),
            "componente": "modelo (equivalente denso)" if fedpub else "modelo",
            "remetente": "servidor ↔ clientes", "destinatario": "ambos", "conteudo": "estado denso",
            "frequencia": "por rodada", "bytes_gravado": gravado, "bytes_recalculado": esperado,
            "divergencia": gravado - esperado,
            "motivo": "" if gravado == esperado else "fórmula do modelo não confere",
            "hipotese": "participação integral; envio individual; sem compressão",
            "exposicao": "parâmetros do modelo",
        })
        if fedpub:
            # o embedding é a representação (dimensão `hidden`), não a previsão (dimensão `horizon`)
            H, hid = d["horizon"], int(d.get("hidden_size") or 64)
            dim = hid if d.get("similarity_source", "representation").startswith("representation") else H
            linhas.append({
                "execucao": str(f.parent.relative_to(ROOT / "results")),
                "componente": "FED-PUB: embedding funcional",
                "remetente": "clientes", "destinatario": "servidor",
                "conteudo": f"vetor de dimensão {dim} (float64)", "frequencia": "por rodada e cliente",
                "bytes_gravado": H * 8, "bytes_recalculado": dim * 8,
                "divergencia": H * 8 - dim * 8,
                "motivo": "" if dim == H else (
                    f"conta `horizon × 8` = {H*8} B, mas com similarity_source=representation o vetor "
                    f"tem dimensão `hidden` = {hid}, logo {dim*8} B (fedpub.py, embedding_bytes)"),
                "hipotese": "float64; índices do upload esparso NÃO contabilizados",
                "exposicao": "média da representação espacial do cliente",
            })

    if halo and halo != "none":
        nb, F = d["halo_nodes_total"], len(d["features"])
        T = timesteps(ds)
        por_passo = nb * F * ELEM * VAL_E_MASCARA
        esperado = por_passo if halo == "static" else T * por_passo
        gravado = d.get("halo_bytes")
        motivo = ""
        if gravado != esperado:
            motivo = ("grava a expansão temporal (T × cópia) em vez de uma cópia única — artefato "
                      "anterior ao fix 95f4ccf" if halo == "static" and gravado == T * por_passo
                      else "recálculo não confere com o gravado")
        linhas.append({
            "execucao": str(f.parent.relative_to(ROOT / "results")), "componente": f"halo {halo}, entrega",
            "remetente": "quem formou o agregado", "destinatario": "cliente consumidor",
            "conteudo": f"{nb} nós virtuais × {F} feature(s), valores + máscara"
                        + ("" if halo == "static" else f" × {T} passos"),
            "frequencia": "uma vez, antes do treino", "bytes_gravado": gravado,
            "bytes_recalculado": esperado, "divergencia": (gravado - esperado) if gravado is not None else None,
            "motivo": motivo,
            "hipotese": "protocolo P1 (envio único); pesos fixos; cache reutilizado",
            "exposicao": "agregado ponderado dos vizinhos externos; com UM vizinho, a série "
                         "padronizada e a máscara desse vizinho, idênticas",
        })
        linhas.append({
            "execucao": str(f.parent.relative_to(ROOT / "results")), "componente": f"halo {halo}, formação",
            "remetente": "não definido", "destinatario": "não definido",
            "conteudo": "lido dos dados globais em memória", "frequencia": "—",
            "bytes_gravado": 0, "bytes_recalculado": None, "divergencia": None,
            "motivo": "NÃO CONTABILIZADO — preparação offline confiável, não é protocolo distribuído",
            "hipotese": "simulação de benchmark",
            "exposicao": "todos os históricos, ao processo que monta os agregados",
        })
    return linhas


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", help="lê `output` do YAML")
    p.add_argument("--runs", nargs="*", help="diretórios de resultados a auditar")
    p.add_argument("--exposicao", action="store_true", help="também mede exposição de vizinho único (usa as partições)")
    p.add_argument("--out", help="prefixo de saída (grava .csv e .md); por omissão, só imprime")
    a = p.parse_args()

    bases = [ROOT / load_config(a.config)["output"]] if a.config else [
        (Path(r) if Path(r).is_absolute() else ROOT / r) for r in (a.runs or ["results"])]
    linhas = []
    for base in bases:
        for f in sorted(base.glob("**/metrics.json")):
            linhas.extend(linhas_de_uma_execucao(f))
    if not linhas:
        raise SystemExit("nenhum metrics.json encontrado")

    div = [l for l in linhas if l["divergencia"] not in (None, 0)]
    naoc = [l for l in linhas if "NÃO CONTABILIZADO" in l["motivo"]]
    print(f"# auditoria de comunicação — {len(linhas)} linhas, {len(div)} divergências, "
          f"{len(naoc)} componentes não contabilizados\n")
    print("Payloads estimados das dimensões; NÃO é tráfego medido. A coluna de exposição documenta")
    print("exposição potencial e NÃO constitui auditoria de privacidade.\n")
    if div:
        print("## divergências entre gravado e recalculado\n")
        vistos = set()
        for l in div:
            chave = (l["componente"], l["bytes_gravado"], l["bytes_recalculado"])
            if chave in vistos:
                continue
            vistos.add(chave)
            print(f"- `{l['execucao']}` — {l['componente']}: gravado {l['bytes_gravado']:,}, "
                  f"recalculado {l['bytes_recalculado']:,}. {l['motivo']}")
        print()

    if a.exposicao:
        print("## exposição de vizinho único\n")
        print("| dataset | k | nós virtuais | 1 vizinho no total | parciais | parciais com 1 vizinho |")
        print("|---|---|---|---|---|---|")
        vistos = set()
        for base in bases:
            for f in sorted(base.glob("**/metrics.json")):
                d = json.loads(f.read_text())
                if not d.get("halo") or d["halo"] == "none":
                    continue
                chave = (d["dataset"], d["requested_clients"])
                if chave in vistos:
                    continue
                vistos.add(chave)
                e = exposicao(d["dataset"], d["requested_clients"], d["partition_seed"])
                print(f"| {d['dataset']} | {d['requested_clients']} | {e['alvos']} | "
                      f"{e['alvos_um_vizinho']} ({100*e['alvos_um_vizinho']/e['alvos']:.1f} %) | "
                      f"{e['parciais']} | {e['parciais_um_vizinho']} "
                      f"({100*e['parciais_um_vizinho']/e['parciais']:.1f} %) |")
        print()

    if a.out:
        out = Path(a.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out.with_suffix(".csv"), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(linhas[0]))
            w.writeheader()
            w.writerows(linhas)
        print(f"gravado: {out.with_suffix('.csv')}")


if __name__ == "__main__":
    main()
