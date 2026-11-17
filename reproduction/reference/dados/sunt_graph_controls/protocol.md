# Protocolo — controles centralizados de grafo no SUNT (v1)

Fixado em 05/10/2026, antes de qualquer smoke ou treino desta etapa e antes de qualquer resultado de
teste novo. Continuação de `results/sunt_stratified_analysis/v1/` (análise exploratória). Autorização:
dez treinamentos finais (2 controles × 5 sementes) e os smokes necessários; nenhum outro método, nenhum
corte em k=6 ou k=166.

## Pergunta

Quanto do erro do Centralizado depende das mensagens entre paradas e, em particular, das conexões
cortadas pela partição k=20? Os controles não identificam sozinhos a causa da perda do FedAvg (ver
"Contrastes").

## Braços

| braço | definição | config de treino (cego) | config final (teste) |
|---|---|---|---|
| Centralizado completo (referência existente) | grafo `transit` simétrico inteiro | `config/sunt_probe_boardings.yaml` | `config/sunt_centralized_boardings.yaml` |
| **sem_arestas** | nenhuma aresta entre nós distintos | `config/sunt_control_noedges_probe.yaml` | `config/sunt_control_noedges.yaml` |
| **cortes_k20** | só as arestas entre nós do mesmo cliente da partição Leiden k=20, semente 42 | `config/sunt_control_cut20_probe.yaml` | `config/sunt_control_cut20.yaml` |
| FedAvg k=20 (existente) | subgrafos dos 20 clientes, treino federado | — | `results/sunt_fedavg_final/sunt/k020/` |

## Protocolo comum (idêntico ao Centralizado final; conferido por código, não por comentário)

Alvo `n_boardings`; entradas `[n_boardings, n_alightings]` com máscaras; W=168, H=24; mesmos dados,
cortes temporais (treino < 2024-05-03 16:00, validação < 2024-05-12 18:00), ordem dos nós e padronização
ajustada no treino; LEConv, `hidden_size` 64, Adam lr 0,001, lote 32, clipping 5,0; 600 épocas sem parada
antecipada (`patience` 999); checkpoint = menor MAE de validação; sementes 7, 42, 101, 123, 314.

Fluxo igual ao da referência: (1) treino cego com `evaluate_test: false` e `keep_last_checkpoint: true`
(`run_grid --backend local`); (2) depois das 600 épocas das cinco sementes de um braço,
`experiments.evaluate_frozen` carrega cada `best_model.pt`, exige que a validação recalculada reproduza a
gravada e só então avalia o teste. `execution.cpus_per_run` (3, contra 4 na referência) só limita threads
de CPU num servidor compartilhado.

`experiments/sunt_graph_controls_check.py` confere a igualdade das configs com as da referência, salvo
`model` (opção do controle), `output` e `execution`.

## Tratamento das arestas (decidido pelo comportamento efetivo do código, não pelo desempenho)

LEConv do PyG 2.4 (versão nas três máquinas): `x_i' = lin3(x_i) + Σ_j e_ji · (lin1(x_j) − lin2(x_i))`,
com `e_ji` normalizado por destino em `models.normalize_incoming_edge_weights` (Σ_j e_ji = 1 por nó).

- **FedAvg (verificado):** cada cliente recebe as arestas internas gravadas na partição
  (`client_XXX/edge_index.npy`, `edge_weight.npy` = submatriz da adjacência, pesos brutos
  `log1p(contagem)/máx`) e o `build_model` **renormaliza** os pesos de entrada de cada nó sobre as arestas
  internas restantes. Não há pesos preservados da normalização do grafo completo.
- **cortes_k20:** mesma regra. O controle remove as 338 arestas dirigidas entre clientes (169 não
  dirigidas, `metadata.json`), mantém as 8 994 internas com os pesos brutos e renormaliza por destino.
  Nenhum nó fica sem aresta de entrada em k=20 (perda máxima 2/3), então não há exceção para nós sem
  entrada. Opção `model.cut_partition` (`experiments/task.py::cut_partition_edges`).
- **sem_arestas:** ablação existente `model.self_loops_only: true` (a mesma dos controles sem arestas de
  METR-LA/PEMS-BAY). Cada nó recebe um auto-laço de peso 1, logo `x_i' = lin3(x_i) + lin1(x_i) − lin2(x_i)`:
  o termo próprio da LEConv permanece, a função de cada nó continua afim na sua própria janela, e não
  há mensagem de nenhum outro nó. Mesma contagem de parâmetros (130 712). Alternativa não adotada:
  remover as arestas sem auto-laço (só `lin3`), mesma classe de funções com outra parametrização; não foi
  usada para não criar uma variante nova. O grafo completo não tem auto-laços (diagonal nula).

## Verificações antes da campanha (sem treino)

`python -m experiments.sunt_graph_controls_check` (local e em cada máquina usada), resultado em
`prelaunch_<host>.json`:

- dados: amostras por split, scaler, features, alvo, W/H iguais aos `metrics.json` da referência;
  contagem de alvos observados por nó em validação e teste igual ao `per_node.count` da referência (só
  denominadores); hashes de série, máscara, origens, nós e adjacência registrados e comparados entre máquinas;
- configs efetivas (acima); arestas e pesos (acima), contra os arquivos da partição e os modelos-cliente
  do FedAvg;
- equivalência: com os mesmos parâmetros e entradas (janelas de treino), a saída do modelo cortes_k20 é a
  reunião das saídas dos modelos-cliente do FedAvg; perturbações: em cortes_k20 nada se propaga entre
  clientes, em sem_arestas nada se propaga entre nós, e no grafo completo a mesma perturbação alcança
  outros clientes (sanidade do teste).

Resultado local (05/10): 42 verificações, 0 falhas; diferença máxima na equivalência 0.

## Smokes (não são resultado)

`config/sunt_control_{noedges,cut20}_smoke.yaml`, saída `results/_smoke/sunt_control_*`, 2 épocas, sem
teste, uma semente por braço no Terumo. Retomada: 1 época, depois `--max-epochs 2` sobre o mesmo
diretório (deve retomar da época 2 pelo `last.pt`). Depois, `sunt_graph_controls_check --smoke`: o
`best_model.pt` carrega estritamente no modelo da config final e reproduz a validação gravada.

## Execução

Terumo, cada execução em `tmux`, sessão e log registrados em `runs.md`. Plano: cada semente roda os dois
controles na mesma GPU e no mesmo tipo de GPU da referência dessa semente (referência: s7 em L40S;
s42, s101, s123, s314 em A16): GPU 0 (L40S) s7; GPUs 1–4 (A16) s42, s101, s123, s314; dois processos por
GPU. Nenhum hardware fica com um único controle. Isso não elimina diferenças de ambiente com a
referência (outra data, outra carga do servidor, não determinismo do CUDA). Tempos não são comparados
entre ambientes. O desktop (RTX 2060) está disponível e só seria usado se o Terumo deixasse de estar livre.

## Análise depois dos treinos (definições já fixadas)

Script novo, saída em `results/sunt_graph_controls/v1/`. Insumos: os 10 `metrics.json` novos, o
Centralizado final e o FedAvg k=20 final (5 sementes cada), enumerados explicitamente.

- Auditoria por execução: config efetiva, commit, semente, hardware, épocas concluídas (600), melhor
  época, hashes dos `metrics.json`, dos checkpoints e dos insumos estruturais, `per_node` em validação e
  teste. Cinco sementes completas por braço; falha ou execução incompleta é registrada, sem reduzir o
  conjunto em silêncio.
- Estratos: exatamente os de `results/sunt_stratified_analysis/v1/` para k=20 (fronteira/interior,
  faixas de perda de vizinhança e de peso, tercis de volume e cobertura com os limites já fixados), lidos
  de `node_descriptors.csv` (hash registrado). Para todos os braços, inclusive sem_arestas, vale a
  classificação do grafo original e da partição original.
- MAE = Σ erro absoluto ÷ Σ alvos observados por estrato; mesmos nós e alvos em todos os braços
  (verificado); recomposição dos estratos = MAE global; MAE médio entre nós como secundário.
- Contrastes principais, pareados por semente (a semente s do controle com a semente s da referência):
  1. **sem_arestas − Centralizado completo:** benefício líquido do grafo sob o protocolo;
  2. **cortes_k20 − Centralizado completo:** sensibilidade às conexões entre clientes;
  3. **FedAvg k=20 − cortes_k20:** diferença associada aos procedimentos de treinamento com
     disponibilidade de arestas correspondente. Não isola uma causa: centralizado e federado ainda diferem
     em atualizações, lotes, agregação e dinâmica de otimização.
  Os contrastes não formam decomposição causal aditiva. Contexto descritivo: FedAvg k=20 − Centralizado
  completo (já relatado).
- Por contraste e estrato: 5 valores, média, dp, sementes < 0, IC95% `t` pareado (4 g.l.), exploratório,
  sem ajuste para multiplicidade. A semente é a réplica; nós e janelas não são. O teste não seleciona
  hiperparâmetros, braços nem quais resultados relatar; todos os contrastes acima são relatados.
- Validação relatada como robustez, lembrando que escolheu os checkpoints.

Fora do escopo: cortes em k=6/k=166, FedRep e outros métodos, horizonte, horário de pico; nenhuma
atualização de `generated-traffic/`, dissertação, slides ou artigo nesta etapa.
