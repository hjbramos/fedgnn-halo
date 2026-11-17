# Protocolo — análise estratificada do SUNT (v1)

Fixado em 05/10/2026, **antes** de calcular qualquer MAE por estrato. O script
`experiments/sunt_stratified_analysis.py` registra o SHA-256 deste arquivo no manifesto e recusa
executar se o arquivo tiver mudado depois de congelado (constante `PROTOCOL_SHA256`).

## Natureza da análise

**Exploratória e motivada por resultados já conhecidos.** Os MAEs agregados de teste
(`results/sunt_report/metricas.md`) já eram conhecidos antes deste protocolo: o halo `1hop` fica
próximo do FedAvg nas três partições (3,965 vs 4,008 em k=6; 4,068 vs 4,074 em k=20; 4,329 vs 4,335
em k=166), e todos os federados ficam acima do Centralizado (3,942). A pergunta é **onde** a partição
prejudica e **onde**, se em algum lugar, o halo recupera. Nenhum resultado desta análise seleciona
método, hiperparâmetro, checkpoint ou faixa; não há correção para comparações múltiplas e os
intervalos são descritivos.

## Insumos (somente estes)

- Alvo `n_boardings`; entradas `[n_boardings, n_alightings]` com máscaras; W=168, H=24; grafo
  `transit` simétrico (peso `log1p(contagem)/máx`, contagens do treino).
- Sementes `7, 42, 101, 123, 314`; partições Leiden `partition_seed_042` realizadas com k=6, 20, 166.
- Registros (`metrics.json`) — enumerados explicitamente, sem busca genérica:
  - Centralizado: `results/sunt_centralized_boardings/sunt/seed_{s}/`
  - FedAvg: `results/sunt_fedavg_final/sunt/k{k}/seed_{s}/` (k=6, 20, 166)
  - halo `1hop` (FedGNNHet): `results/sunt_fedgnnhet_final/sunt/k{k}/halo_1hop/seed_{s}/` (k=6, 20, 166)
  - Secundários: FedProx μ=0,01 `results/sunt_fedprox_final/sunt/k{k}/mu0p01/seed_{s}/` e Ditto λ por
    cliente `results/sunt_ditto_final/sunt/k{k}/lam_per_client/seed_{s}/` (k=6, 20). Para o Ditto, usa-se
    o campo `test` (modelos pessoais), não `test_global`.
- Excluídos: sondagens, alvo carga/`loading`, grafo direcionado, FED-PUB, `sunt-fedstgnn`.
- Fonte das métricas por nó: `test.per_node` (principal) e `validation.per_node` (robustez). Sem
  reavaliação: se faltar `per_node`, a lacuna é registrada e a célula fica fora.

## Período de treino para descritores

Treino = horas com timestamp `< 2024-05-03 16:00` (`TRAIN_END_EXCLUSIVE`), o mesmo corte das janelas de
treino (todos os alvos antes dele), do escalonador e do grafo. Inclui as primeiras 168 h, que só
aparecem como entrada. O script confere que o índice do corte coincide com a 1.ª origem de validação e
que as horas são consecutivas e únicas (uma célula = um par nó × hora, sem repetição entre janelas).

## Estratos (só estrutura da partição e dados de treino)

Seja `A` a adjacência do grafo `transit` com a diagonal zerada; aresta de entrada `u→v` se `A[u,v]>0`.
Externa se `u` e `v` estão em clientes diferentes.

| Dimensão | Definição | Faixas |
|---|---|---|
| Fronteira | `fronteira` = ≥1 vizinho de entrada externo; `interior` = nenhum; `sem_entrada` = grau de entrada 0 | categórica |
| Perda de vizinhança | nº de arestas de entrada externas ÷ grau de entrada | `0`, `(0;0,25]`, `(0,25;0,50]`, `(0,50;1]`; `sem_entrada` à parte |
| Perda de peso | Σ pesos de entrada externos ÷ Σ pesos de entrada (sem laços) | mesmas faixas; `sem_entrada` à parte |
| Volume | média de `n_boardings` nas células observadas do treino | tercis; `sem_obs_treino` à parte |
| Cobertura | células observadas do alvo no treino ÷ horas do treino | tercis; `sem_obs_treino` à parte |

- Verificação: o grafo deve ser simétrico; logo, nenhum nó pode ter aresta de saída externa sem ter
  aresta de entrada externa (`só_saída` = 0). Se falhar, a análise para.
- Tercis: limites `q1/3` e `q2/3` (quantil linear do numpy) calculados **uma única vez** sobre os nós
  com ≥1 observação de treino, comuns a métodos, sementes e partições. Faixas fechadas à direita:
  `baixo = ≤ q1`, `médio = (q1; q2]`, `alto = > q2`. Empates caem inteiros na mesma faixa; se `q1 == q2`
  ou se uma faixa ficar vazia, ela é fundida/omitida e o número de faixas efetivo é relatado. Não se
  acrescenta ruído.
- Ausência de observação nunca vira embarque zero: média e cobertura usam só a máscara.
- Nós virtuais do halo não entram: `per_node` tem exatamente os 2903 nós reais (verificado).

## Métrica

Para cada (método, k, semente, split, estrato):
`MAE = Σ erro absoluto ÷ Σ alvos observados` sobre os nós do estrato (somas do `per_node`).
O Centralizado é estratificado por cada partição. Registram-se: nº de nós; nº de nós com alvos
avaliáveis (`count>0`); nº de ocorrências de alvos observados (`Σ count`, **ocorrências em janelas,
com repetição temporal até H=24 vezes por célula**, não células únicas); participação do estrato no
denominador global (`Σ count` do estrato ÷ total) e no numerador (erro absoluto) de cada método.
Secundário: MAE médio entre nós avaliáveis (média simples dos `err/count` por nó).

Verificações numéricas (tolerância relativa 1e-9 para recomposições; 1e-6 frente ao MAE gravado):
erros finitos e ≥0; contagens inteiras ≥0 e idênticas entre métodos (mesmos alvos); `Σ err ÷ Σ count`
reproduz `MAE` gravado e `Σ count` reproduz `observations`; a média ponderada dos MAEs dos estratos
(peso = participação no denominador) reproduz o MAE global; nós por cliente iguais a `client_nodes`;
ordem dos nós da partição igual à da série; hashes dos `metrics.json` iguais aos do
`generated-traffic/provenance.json`.

## Contrastes (pareados por semente)

- `FedAvg − Centralizado` (perda associada ao procedimento federado);
- `halo − FedAvg` (negativo favorece o halo);
- `halo − Centralizado` (distância residual);
- secundários (k=6, 20): `FedProx − FedAvg`, `Ditto − FedAvg`, `FedProx − Centralizado`, `Ditto − Centralizado`.
- Concentração na fronteira: `(halo − FedAvg)_fronteira − (halo − FedAvg)_interior`, por semente.
  Análogo para `FedAvg − Centralizado`.

Por contraste: os 5 valores, média, desvio-padrão (ddof=1), nº de sementes com valor < 0 e IC95% `t`
(4 g.l.) — **exploratório, não ajustado, dependente de normalidade**. Unidade de réplica = semente;
nós, clientes, janelas e horizontes não são réplicas. As tendências por faixa de perda de peso são
descritas (sinal e monotonicidade da média), sem teste de tendência.

## Perguntas a responder no relatório

1. O halo ajuda especificamente os nós que perderam mensagens de entrada?
2. Há melhora localizada ocultada pelo MAE global?
3. A perda frente ao Centralizado cresce com a fração ponderada de entrada externa?
4. O padrão persiste entre sementes e entre k?
5. Volume e cobertura ajudam a interpretar a heterogeneidade?
6. Qual hipótese merece o próximo experimento: valor do grafo, otimização federada ou personalização?

Fora do escopo desta etapa: análise por horizonte, horário de pico, novos controles e novos métodos.
