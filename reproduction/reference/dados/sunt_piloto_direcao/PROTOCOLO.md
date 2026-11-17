# Piloto exploratório: orientação do grafo do SUNT no centralizado de embarques

Definido antes dos treinos (28/10/2026). **Piloto exploratório**: só validação, sem teste, sem alterar o baseline
oficial nem os resultados existentes; não se misturar com o estudo de carga + grafo direcionado.

## Pergunta

Preservar o sentido observado das transições entre paradas consecutivas muda o erro de validação do LEConv
centralizado na previsão de embarques, frente ao grafo simétrico vigente? O braço invertido serve de diagnóstico da
orientação: tem a mesma esparsidade e os mesmos pares do observado, com os sentidos trocados.

## Braços (5 sementes pareadas: 7, 42, 101, 123, 314)

| braço | grafo (`windows.sunt.graph`) | arestas |
|---|---|---|
| simétrico | `transit` (vigente) | os dois sentidos de cada par |
| observado | `transit_pair_observed` | só os sentidos com transição observada no treino |
| invertido | `transit_pair_reversed` | os do observado com origem e destino trocados |

- **Construção:** contagens de transições entre posições consecutivas das viagens, só no período de treino
  (`data_loader/sunt.py`, `_directed_transition_counts`). Mesmo universo de nós (2 903) e mesmo suporte de pares
  (4 666) nos três braços; o simétrico é a simetrização exata do observado.
- **Pesos:** uma única afinidade por par não direcionado, pela regra vigente do simétrico:
  `log1p(contagem do par u↔v)/máx`. A mesma afinidade é usada em cada sentido disponível; no invertido, ela acompanha
  a aresta invertida. A contagem específica de cada sentido **não** é usada (é o `transit_directed` do estudo de carga).
- **Normalização:** por destino (soma dos pesos de entrada), como no vigente; **os denominadores mudam com a
  topologia** (o grau de entrada mediano cai de 3 para 1). Nó sem entrada usa só o termo próprio da LEConv, sem
  arestas artificiais (`model.allow_nodes_without_incoming: true` nos três braços; sem efeito no simétrico).
- **Iguais nos três braços:** dados, máscaras, entradas `[n_boardings, n_alightings]`, alvo `n_boardings`, W168/H24,
  cortes, escala ajustada no treino, LEConv h=64, lr 0,001, lote 32, clip 5, **600 épocas sem parada antecipada**,
  melhor checkpoint pelo menor MAE de validação. Mesma semente ⇒ mesma inicialização (hash do estado inicial gravado)
  e mesma ordem dos dados (gerador do carregador com a semente).
- **Determinismo estrito:** `use_deterministic_algorithms(True)` sem `warn_only`, cuDNN determinístico, TF32
  desligado, `CUBLAS_WORKSPACE_CONFIG=:4096:8` e `PYTHONHASHSEED=0` antes do processo; opções efetivas gravadas.
- **Teste:** `evaluate_test: false` — não é carregado nem avaliado.

## Auditoria (antes dos treinos; `experiments/sunt_direcao_auditoria.py`)

Universo, pares, simetrização, mesma afinidade, transposição, diagonal, arestas, reciprocidade, graus, componentes e
nós sem entrada por braço (`auditoria/auditoria_topologias.json`), e os **estratos**, definidos só pela topologia,
antes de examinar erros (`auditoria/estratos_por_no.csv`), por braço direcionado, relativos ao grau de entrada no
simétrico: **sem entrada** (grau 0); **grande redução** (0 < grau < 0,5 × grau no simétrico); **demais**.

## Execução

Terumo, worktree própria, sessão tmux `piloto_direcao`; uma janela por semente, presa a uma GPU (7 → L40S; 42, 101,
123, 314 → A16), com os três braços em sequência (mesmo hardware dentro da semente). Comando:
`SESSION=piloto_direcao bash scripts/sunt_piloto_direcao.sh`. Logs e códigos de saída em `logs/`. Antes, smoke de 2
épocas por braço (semente 7) em `results/_smoke/sunt_piloto_direcao`.

## Relatório (`experiments/sunt_direcao_relatorio.py`)

MAE e RMSE de validação na escala original, por semente, agregados, por horizonte e por estrato; diferenças pareadas
**observado − simétrico**, **invertido − simétrico** e **observado − invertido** (média, dp e os cinco valores);
melhor época e tempo de treino. Descritivo, sem testes de hipótese.

## Leitura

Observado × invertido é diagnóstico da orientação; resultados semelhantes **não** demonstram que a esparsidade
explique o efeito. Com cinco sementes e um único conjunto de dados, nada se generaliza; o baseline oficial não muda.
