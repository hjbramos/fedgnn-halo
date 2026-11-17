"""SUNT (Salvador Urban Network Transportation) — alinhado ao projeto sunt-fedstgnn.

Fonte: HF labiaufba/PublicTransportationSunt, tabela OD diária (od-YYYY-MM-DD.parquet).
  data/sunt/raw/OD/od-2024-03-01.parquet ... od-2024-05-30.parquet   (91 dias)
  data/sunt/processed/
      sunt_passenger_flows.npz   features (T, N, 3) [n_boardings, n_alightings, loading], mask, timestamps, stop_ids
                                 (embarques/desembarques = soma na hora; loading = média na hora)
      sunt_n_boardings.parquet   série (T x N) de embarques (alvo)
      sunt_transit_graph.npz     adj (N x N), transition_counts, stop_ids

Contrato (config/datasets/sunt/temporal.yaml do sunt-fedstgnn):
  período   2024-03-01 00:00 -> 2024-05-31 00:00 (exclusivo), bin 1h  => T = 2184, N = 2903
  treino    < 2024-05-03 16:00 ; validação < 2024-05-12 18:00 ; teste = resto
  grafo     "transit": paradas consecutivas (pt_sequence) na mesma viagem, só no período de treino,
            não direcionado, peso = log1p(contagem) / max.
"""
from __future__ import annotations

import gc
import os
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from .common import DATA_DIR, align_to_graph, split_by_dates

RAW_DIR = DATA_DIR / "sunt" / "raw"
OD_DIR = RAW_DIR / "OD"
PROCESSED_DIR = DATA_DIR / "sunt" / "processed"

HF_REPO = "labiaufba/PublicTransportationSunt"
_HF_API = f"https://huggingface.co/api/datasets/{HF_REPO}/tree/{{revision}}/{{folder}}"
_HF_RESOLVE = f"https://huggingface.co/datasets/{HF_REPO}/resolve/{{revision}}/{{folder}}/{{fname}}"
_FOLDER = {"boarding": "Boarding", "od": "OD", "alighting": "Alighting"}

# --- contrato temporal (== sunt-fedstgnn/config/datasets/sunt/temporal.yaml)
START = pd.Timestamp("2024-03-01 00:00:00")
END_EXCLUSIVE = pd.Timestamp("2024-05-31 00:00:00")
TRAIN_END_EXCLUSIVE = pd.Timestamp("2024-05-03 16:00:00")
VALIDATION_END_EXCLUSIVE = pd.Timestamp("2024-05-12 18:00:00")
BIN = "1h"
FEATURES = ("n-boardings", "n-alighting", "loading")
FEATURE_NAMES = ("n_boardings", "n_alightings", "loading")
AGGREGATION = ("sum", "sum", "mean")  # loading é ocupação instantânea: média por hora (notebook 01_Build_Loading_Time_Series)

# --- previsão (== config/datasets/sunt/forecasting.yaml; oficial usa 168)
INPUT_STEPS = 72
HORIZON = 24


# --------------------------------------------------------------------------- download
@lru_cache(maxsize=8)
def available_dates(kind: str) -> set[str]:
    r = requests.get(_HF_API.format(folder=_FOLDER[kind], revision=json.loads(os.environ.get("FEDGNN_HF_REVISIONS", "{}")).get(HF_REPO, "main")), timeout=15)
    r.raise_for_status()
    return {e["path"].split("/")[-1].split("-", 1)[-1].removesuffix(".parquet") for e in r.json()}


def download_day(kind: str, date: str, force: bool = False) -> Path:
    """Baixa um parquet diário para data/sunt/raw/<Folder>/ (se ainda não existir)."""
    folder = _FOLDER[kind]
    fname = f"{kind}-{date}.parquet"
    dest = RAW_DIR / folder / fname
    if dest.exists() and not force:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"baixando {folder}/{fname}")
    r = requests.get(_HF_RESOLVE.format(folder=folder, fname=fname, revision=json.loads(os.environ.get("FEDGNN_HF_REVISIONS", "{}")).get(HF_REPO, "main")), timeout=120)
    r.raise_for_status()
    dest.write_bytes(r.content)
    return dest


def od_dates() -> list[str]:
    """Os 91 dias de OD do período (2024-03-01 .. 2024-05-30)."""
    return [d.strftime("%Y-%m-%d") for d in pd.date_range(START, END_EXCLUSIVE - pd.Timedelta(days=1))]


def download_raw() -> list[Path]:
    avail = available_dates("od")
    files = [download_day("od", d) for d in od_dates() if d in avail]
    assert len(files) == 91, f"esperados 91 arquivos OD; encontrados {len(files)}"
    return files


# --------------------------------------------------------------------------- série
def _discover_stop_ids(files: list[Path]) -> np.ndarray:
    ids: set[int] = set()
    for p in files:
        col = pd.read_parquet(p, columns=["stop_id"])["stop_id"]
        ids.update(int(v) for v in pd.to_numeric(col, errors="coerce").dropna().unique())
    return np.array(sorted(ids), dtype=np.int64)


def build_passenger_flows(force: bool = False) -> dict[str, np.ndarray]:
    """features (T, N, 3), observation_mask (T, N, 3), timestamps (T,), stop_ids (N,)."""
    cache = PROCESSED_DIR / "sunt_passenger_flows.npz"
    if cache.exists() and not force:
        return dict(np.load(cache))

    files = download_raw()
    stop_ids = _discover_stop_ids(files)
    node_index = pd.Series(np.arange(len(stop_ids)), index=stop_ids)
    timestamps = pd.date_range(START, END_EXCLUSIVE - pd.Timedelta(BIN), freq=BIN)
    T, N = len(timestamps), len(stop_ids)

    values = np.zeros((T, N, len(FEATURES)), dtype=np.float64)
    counts = np.zeros((T, N, len(FEATURES)), dtype=np.int64)
    for p in files:
        df = pd.read_parquet(p, columns=["stop_time", "stop_id", *FEATURES])
        df["stop_time"] = pd.to_datetime(df["stop_time"], errors="coerce")
        df["stop_id"] = pd.to_numeric(df["stop_id"], errors="coerce")
        df = df.dropna(subset=["stop_time", "stop_id"])
        df = df[(df["stop_time"] >= START) & (df["stop_time"] < END_EXCLUSIVE)]
        if df.empty:
            continue
        hour = ((df["stop_time"].dt.floor(BIN) - START) / pd.Timedelta(BIN)).astype(int).to_numpy()
        node = node_index.loc[df["stop_id"].astype("int64")].to_numpy()
        for k, col in enumerate(FEATURES):
            v = pd.to_numeric(df[col], errors="coerce")
            ok = v.notna().to_numpy() & np.isfinite(v.to_numpy(dtype=float)) & (v.fillna(0) >= 0).to_numpy()
            np.add.at(values[:, :, k], (hour[ok], node[ok]), v.to_numpy(dtype=float)[ok])
            np.add.at(counts[:, :, k], (hour[ok], node[ok]), 1)
        del df
        gc.collect()

    mask = counts > 0
    for k, agg in enumerate(AGGREGATION):
        if agg == "mean":
            values[:, :, k] = np.divide(values[:, :, k], counts[:, :, k], out=np.zeros_like(values[:, :, k]), where=mask[:, :, k])

    out = dict(
        features=values.astype(np.float32),
        observation_mask=mask,
        timestamps=timestamps.to_numpy(dtype="datetime64[ns]"),
        stop_ids=stop_ids,
    )
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, **out)
    return out


def load_sunt(force: bool = False) -> pd.DataFrame:
    """Série (T x N) de embarques por parada (alvo), T = 2184 h, N = 2903 paradas."""
    cache = PROCESSED_DIR / "sunt_n_boardings.parquet"
    if cache.exists() and not force:
        return pd.read_parquet(cache)
    flows = build_passenger_flows(force)
    df = pd.DataFrame(
        flows["features"][:, :, 0],
        index=pd.DatetimeIndex(flows["timestamps"], name="timestamp"),
        columns=[str(s) for s in flows["stop_ids"]],
    )
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache, engine="pyarrow", index=True)
    return df


def load_sunt_features(force: bool = False) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex, list[str]]:
    """(features (T,N,3), mask (T,N,3), timestamps, stop_ids) — embarques, desembarques e carga."""
    f = build_passenger_flows(force)
    return f["features"], f["observation_mask"], pd.DatetimeIndex(f["timestamps"]), [str(s) for s in f["stop_ids"]]


# --------------------------------------------------------------------------- grafo
def _directed_transition_counts() -> tuple[np.ndarray, np.ndarray]:
    """(contagens u->v (N x N, uint64), stop_ids): transições entre posições consecutivas das viagens do treino."""
    files = download_raw()
    # Nós e arestas usam só o período de treino (nada de validação/teste).
    ids: set[int] = set()
    for p in files:
        df = pd.read_parquet(p, columns=["stop_id", "stop_time"])
        df["stop_time"] = pd.to_datetime(df["stop_time"], errors="coerce")
        df["stop_id"] = pd.to_numeric(df["stop_id"], errors="coerce")
        df = df.dropna()
        df = df[(df["stop_time"] >= START) & (df["stop_time"] < TRAIN_END_EXCLUSIVE)]
        ids.update(df["stop_id"].astype("int64").unique().tolist())
    stop_ids = np.array(sorted(ids), dtype=np.int64)
    node_index = pd.Series(np.arange(len(stop_ids)), index=stop_ids)
    N = len(stop_ids)

    directed = np.zeros((N, N), dtype=np.uint64)
    for p in files:
        df = pd.read_parquet(p, columns=["trip_id", "pt_sequence", "stop_id", "stop_time"])
        df["stop_time"] = pd.to_datetime(df["stop_time"], errors="coerce")
        df["pt_sequence"] = pd.to_numeric(df["pt_sequence"], errors="coerce")
        df["stop_id"] = pd.to_numeric(df["stop_id"], errors="coerce")
        df = df.dropna(subset=["trip_id", "pt_sequence", "stop_id", "stop_time"])
        df = df[(df["stop_time"] >= START) & (df["stop_time"] < TRAIN_END_EXCLUSIVE)].copy()
        if df.empty:
            continue
        df["stop_id"] = df["stop_id"].astype("int64")

        # Descarta a viagem inteira se uma sequência aponta para paradas diferentes.
        seq_stops = df.groupby(["trip_id", "pt_sequence"], sort=False)["stop_id"].nunique()
        ambiguous = set(seq_stops[seq_stops > 1].index.get_level_values(0))
        df = df[~df["trip_id"].isin(ambiguous)]

        df = df.sort_values(["trip_id", "pt_sequence", "stop_time"], kind="stable")
        df = df.drop_duplicates(["trip_id", "pt_sequence"], keep="first")
        g = df.groupby("trip_id", sort=False)
        df["next_stop_id"] = g["stop_id"].shift(-1)
        df["next_sequence"] = g["pt_sequence"].shift(-1)
        seg = df.dropna(subset=["next_stop_id", "next_sequence"])
        seg = seg[(seg["next_sequence"] - seg["pt_sequence"]).eq(1) & seg["stop_id"].ne(seg["next_stop_id"])]

        counts = seg.groupby(["stop_id", "next_stop_id"], sort=False).size()
        src = node_index.loc[counts.index.get_level_values(0).astype("int64")].to_numpy()
        dst = node_index.loc[counts.index.get_level_values(1).astype("int64")].to_numpy()
        np.add.at(directed, (src, dst), counts.to_numpy(dtype=np.uint64))
        del df, seg, counts
        gc.collect()
    return directed, stop_ids


def build_transit_graph(force: bool = False) -> dict[str, np.ndarray]:
    """Grafo 'transit' do sunt-fedstgnn: adj (N x N) float32, transition_counts (N x N), stop_ids."""
    cache = PROCESSED_DIR / "sunt_transit_graph.npz"
    if cache.exists() and not force:
        return dict(np.load(cache))
    directed, stop_ids = _directed_transition_counts()

    # Não direcionado: soma os dois sentidos; peso log1p normalizado.
    transition_counts = directed + directed.T
    np.fill_diagonal(transition_counts, 0)
    transition_counts = transition_counts.astype(np.uint32)
    adj = np.log1p(transition_counts.astype(np.float64))
    adj /= adj.max()
    adj = adj.astype(np.float32)
    np.fill_diagonal(adj, 0.0)

    out = dict(adj=adj, transition_counts=transition_counts, stop_ids=stop_ids)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, **out)
    return out


def build_transit_graph_directed(force: bool = False) -> dict[str, np.ndarray]:
    """Grafo 'transit_directed': a mesma contagem do 'transit', por sentido (adj[u, v] = u -> v), sem somar.

    Peso = log1p(contagem do sentido) / máximo, diagonal zero, contagens só do treino (a mesma regra do
    simétrico, sem a soma dos sentidos). Cache próprio, para nunca carregar o simétrico por engano.
    """
    cache = PROCESSED_DIR / "sunt_transit_graph_directed.npz"
    if cache.exists() and not force:
        return dict(np.load(cache))
    directed, stop_ids = _directed_transition_counts()
    np.fill_diagonal(directed, 0)
    transition_counts = directed.astype(np.uint32)
    adj = np.log1p(transition_counts.astype(np.float64))
    adj /= adj.max()
    adj = adj.astype(np.float32)
    np.fill_diagonal(adj, 0.0)
    out = dict(adj=adj, transition_counts=transition_counts, stop_ids=stop_ids)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, **out)
    return out


def pair_affinity_orientations(adj_sym: np.ndarray, directed_counts: np.ndarray) -> dict[str, np.ndarray]:
    """Piloto de direção: a MESMA afinidade por par do grafo simétrico, só nos sentidos observados.

    observado[u, v] = adj_sym[u, v] se houve transição u -> v no treino, senão 0; invertido = observado.T
    (a afinidade acompanha a aresta invertida). Mesmo universo de nós e mesmo suporte de pares do simétrico;
    o peso NÃO é a contagem de cada sentido (isso é o `transit_directed`, do estudo de carga).
    """
    adj_sym = np.asarray(adj_sym, dtype=np.float32)
    seen = np.asarray(directed_counts) > 0
    np.fill_diagonal(seen, False)
    if not np.array_equal(adj_sym > 0, seen | seen.T):
        raise ValueError("o suporte do simétrico não é a simetrização das transições observadas")
    if not np.array_equal(adj_sym, adj_sym.T):
        raise ValueError("o grafo simétrico não é simétrico")
    observed = np.where(seen, adj_sym, np.float32(0)).astype(np.float32)
    return {"observed": observed, "reversed": np.ascontiguousarray(observed.T)}


def _pair_affinity_graph(which: str) -> dict[str, np.ndarray]:
    sym, directed = build_transit_graph(), build_transit_graph_directed()
    if not np.array_equal(sym["stop_ids"], directed["stop_ids"]):
        raise ValueError("caches do simétrico e do direcionado com universos de nós diferentes")
    adj = pair_affinity_orientations(sym["adj"], directed["transition_counts"])[which]
    return dict(adj=adj, transition_counts=directed["transition_counts"], stop_ids=sym["stop_ids"])


def build_transit_pair_observed(force: bool = False) -> dict[str, np.ndarray]:
    """Piloto de direção, braço 2: afinidade do par simétrico nos sentidos observados (adj[u, v] = u -> v)."""
    return _pair_affinity_graph("observed")


def build_transit_pair_reversed(force: bool = False) -> dict[str, np.ndarray]:
    """Piloto de direção, braço 3: o braço 2 com origem e destino trocados (transposta), mesma afinidade."""
    return _pair_affinity_graph("reversed")


GRAPHS = {"transit": build_transit_graph, "transit_directed": build_transit_graph_directed,
          "transit_pair_observed": build_transit_pair_observed, "transit_pair_reversed": build_transit_pair_reversed}


GRAPH_FILE_PREFIX = "arquivo:"


GRAPH_FILE_WEIGHTS = ("adj", "contagem")


def load_graph_file(spec: str | Path) -> dict[str, np.ndarray]:
    """Grafo gravado por um experimento (`adj` N x N, adj[u, v] = u -> v; `stop_ids`), caminho relativo à raiz do
    repositório. O universo e a ordem dos nós têm de ser os do contrato (os do `transit`), conferidos aqui.
    "<npz>?peso=contagem" usa `transition_counts` bruto (nº de transições u -> v no treino, sem log nem escala) no
    lugar de `adj`; exige a mesma topologia de `adj`. Padrão: peso=adj (o peso gravado)."""
    path, _, q = str(spec).partition("?")
    opts = dict(kv.split("=", 1) for kv in q.split("&") if kv)
    peso = opts.pop("peso", "adj")
    if opts or peso not in GRAPH_FILE_WEIGHTS:
        raise ValueError(f"opções inválidas no arquivo de grafo: {opts or peso}; peso em {GRAPH_FILE_WEIGHTS}")
    p = Path(path)
    p = p if p.is_absolute() else DATA_DIR.parent / p
    g = dict(np.load(p))
    ref = build_transit_graph()["stop_ids"]
    if not np.array_equal(np.asarray(g["stop_ids"]).astype(np.int64), np.asarray(ref).astype(np.int64)):
        raise ValueError(f"{p}: universo ou ordem de paradas diferente do contrato")
    adj = np.asarray(g["adj"], dtype=np.float32)
    if adj.shape != (len(ref), len(ref)) or not np.isfinite(adj).all() or (adj < 0).any() or np.diag(adj).any():
        raise ValueError(f"{p}: adjacência inválida (forma, não finitos, negativos ou diagonal)")
    if peso == "contagem":
        c = np.asarray(g["transition_counts"])
        if c.shape != adj.shape or not np.issubdtype(c.dtype, np.integer) or (c < 0).any():
            raise ValueError(f"{p}: transition_counts inválido (forma, tipo não inteiro ou negativos)")
        if not np.array_equal(c > 0, adj > 0):
            raise ValueError(f"{p}: transition_counts e adj com topologias diferentes")
        if c.max() >= 2 ** 24 or c.sum(0).max() >= 2 ** 24 or c.sum(1).max() >= 2 ** 24:
            raise ValueError(f"{p}: contagens ou graus acima da precisão inteira exata do float32")
        adj = c.astype(np.float32)
    return dict(adj=adj, stop_ids=g["stop_ids"], peso=peso)


ITINERARY_PREFIX = "itinerarios:"
ITINERARY_WEIGHTS = ("afinidade_exp", "distancia", "binario")


def itinerary_weights(distancia_m: np.ndarray, peso: str) -> np.ndarray:
    """Peso computacional a partir da distância viária preservada (results/sunt_montevideo/CRITERIOS.md §7).
    afinidade_exp: exp(-d/λ), λ = mediana de d das arestas do arquivo; distancia: d bruto (como o Montevideo fornece:
    sob D_O^-1 W, saída mais longe ⇒ mais influência); binario: 1."""
    d = np.asarray(distancia_m, dtype=np.float64)
    if peso == "afinidade_exp":
        w = np.exp(-d / np.median(d))
    elif peso == "distancia":
        w = d.copy()
    elif peso == "binario":
        w = np.ones_like(d)
    else:
        raise ValueError(f"peso desconhecido: {peso}; use {ITINERARY_WEIGHTS}")
    w = w.astype(np.float32)
    if not (np.isfinite(w).all() and (w > 0).all()):
        raise ValueError("peso nulo ou não finito em aresta (underflow?)")
    return w


def load_itinerary_graph(spec: str) -> dict[str, np.ndarray]:
    """Grafo SUNT/Montevideo: "<npz>?peso=<modo>" (results/sunt_montevideo/grafo/<candidato>/grafo.npz).
    Os nós são um subconjunto do contrato, na ordem do contrato; as arestas são físicas (A→B), com distancia_m. O
    peso é preparado aqui e a adjacência devolvida tem adj[u, v] = peso de u→v (o reverso fica a cargo do modelo)."""
    path, _, q = spec.partition("?")
    opts = dict(kv.split("=", 1) for kv in q.split("&") if kv)
    peso = opts.pop("peso", "afinidade_exp")
    if opts:
        raise ValueError(f"opções desconhecidas no grafo de itinerários: {sorted(opts)}")
    p = Path(path)
    p = p if p.is_absolute() else DATA_DIR.parent / p
    g = dict(np.load(p))
    ids = np.asarray(g["stop_ids"]).astype(np.int64)
    ref = np.asarray(build_transit_graph()["stop_ids"]).astype(np.int64)
    pos = {s: i for i, s in enumerate(ref)}
    if len(set(ids)) != len(ids) or any(s not in pos for s in ids):
        raise ValueError(f"{p}: paradas duplicadas ou fora do contrato")
    idx = np.array([pos[s] for s in ids])
    if not (np.diff(idx) > 0).all() or not np.array_equal(idx, g["indice_contrato"]):
        raise ValueError(f"{p}: ordem diferente da do contrato")
    o, t, d = g["origem"].astype(np.int64), g["destino"].astype(np.int64), g["distancia_m"].astype(np.float64)
    n = len(ids)
    if not (len(o) == len(t) == len(d)) or len(o) == 0:
        raise ValueError(f"{p}: arestas desalinhadas ou vazias")
    if (o < 0).any() or (t < 0).any() or (o >= n).any() or (t >= n).any() or (o == t).any():
        raise ValueError(f"{p}: aresta com nó inexistente ou laço")
    if len(set(zip(o.tolist(), t.tolist()))) != len(o):
        raise ValueError(f"{p}: aresta dirigida duplicada")
    if not (np.isfinite(d).all() and (d > 0).all()):
        raise ValueError(f"{p}: distância inválida")
    adj = np.zeros((n, n), np.float32)
    adj[o, t] = itinerary_weights(d, peso)
    return dict(adj=adj, stop_ids=ids, distancia_m=d, origem=o, destino=t, peso=peso)


def load_adjacency(df: pd.DataFrame, graph: str | None = None) -> tuple[np.ndarray, list[str], pd.DataFrame]:
    """Adjacência alinhada às colunas da série (mesma assinatura dos loaders DCRNN). graph: transit (padrão,
    simétrico) | transit_directed | ... | "arquivo:<npz>" (grafo de um experimento, ver load_graph_file) |
    "itinerarios:<npz>?peso=<modo>" (recorte SUNT/Montevideo: subconjunto de paradas, ver load_itinerary_graph)."""
    graph = graph or "transit"
    if graph.startswith(ITINERARY_PREFIX):
        g = load_itinerary_graph(graph[len(ITINERARY_PREFIX):])
        stop_ids = [str(s) for s in g["stop_ids"]]
        return g["adj"], stop_ids, align_to_graph(df[stop_ids], stop_ids)
    if graph.startswith(GRAPH_FILE_PREFIX):
        g = load_graph_file(graph[len(GRAPH_FILE_PREFIX):])
        stop_ids = [str(s) for s in g["stop_ids"]]
        return g["adj"], stop_ids, align_to_graph(df, stop_ids)
    if graph not in GRAPHS:
        raise ValueError(f"grafo do SUNT desconhecido: {graph}; use {sorted(GRAPHS)}")
    g = GRAPHS[graph]()
    stop_ids = [str(s) for s in g["stop_ids"]]
    return g["adj"], stop_ids, align_to_graph(df, stop_ids)


# --------------------------------------------------------------------------- splits
def split_indices(index: pd.DatetimeIndex, seq_len: int = INPUT_STEPS, horizon: int = HORIZON):
    """(train_idx, val_idx, test_idx) para janelas de make_windows(seq_len, horizon)."""
    return split_by_dates(index, TRAIN_END_EXCLUSIVE, VALIDATION_END_EXCLUSIVE, seq_len, horizon)


# --------------------------------------------------------------------------- variante simples (sunt_examples)
def load_boarding_series(
    start_date: str = "2024-04-01", n_days: int = 60, freq: str = "1h", top_n: int = 30, force: bool = False
) -> pd.DataFrame:
    """Variante do sunt_examples: contagem de eventos de Boarding, top_n paradas. Não é o contrato oficial."""
    cache = PROCESSED_DIR / f"sunt_boarding_{start_date}_{n_days}d_{freq}_top{top_n}.parquet"
    if cache.exists() and not force:
        return pd.read_parquet(cache)
    avail = available_dates("boarding")
    dates = [d.strftime("%Y-%m-%d") for d in pd.date_range(start_date, periods=n_days) if d.strftime("%Y-%m-%d") in avail]
    daily = []
    for d in dates:
        df = pd.read_parquet(download_day("boarding", d), columns=["register_time", "stop_id"])
        df = df[df["stop_id"].notna() & (df["stop_id"].astype(str) != "nan")]
        df["register_time"] = pd.to_datetime(df["register_time"])
        t0 = pd.Timestamp(d)
        df = df[(df["register_time"] >= t0) & (df["register_time"] < t0 + pd.Timedelta(days=1))]
        if not df.empty:
            daily.append(df.set_index("register_time").groupby([pd.Grouper(freq=freq), "stop_id"]).size().unstack(fill_value=0))
    ts = pd.concat(daily).fillna(0).sort_index()
    ts = ts.groupby(ts.index).sum()
    ts = ts[ts.sum().nlargest(top_n).index]
    ts = ts.reindex(pd.date_range(ts.index.min(), ts.index.max(), freq=freq), fill_value=0).astype(np.float32)
    ts.columns = [str(c) for c in ts.columns]
    ts.index.name = "timestamp"
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    ts.to_parquet(cache, engine="pyarrow", index=True)
    return ts
