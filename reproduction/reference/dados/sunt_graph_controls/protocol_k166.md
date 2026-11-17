# Protocolo complementar — controle com cortes da partição k=166 (desktop)

Fixado em 05/10/2026, antes de qualquer smoke ou treino deste braço e antes de qualquer resultado de
teste dos controles. Amplia `protocol.md` (mantido sem alteração como histórico): a autorização passa a
incluir o **Centralizado com os cortes da partição k=166** (Leiden, semente 42; 163 pedidos → 166
realizados, a mesma do FedAvg final), sementes 7, 42, 101, 123, 314, executado no desktop. Os dez
treinamentos do Terumo continuam sem alteração. Nenhum outro método, nenhum corte em k=6.

## Braço novo

| braço | definição | treino cego | avaliação final |
|---|---|---|---|
| **cortes_k166** | só as arestas entre nós do mesmo cliente da partição k=166 | `config/sunt_control_cut166_probe.yaml` | `config/sunt_control_cut166.yaml` |

Protocolo comum idêntico ao de `protocol.md` (alvo, entradas e máscaras, W168/H24, dados e scaler, LEConv
64, Adam 0,001, lote 32, clipping 5, 600 épocas sem parada antecipada, checkpoint pela validação, teste só
via `evaluate_frozen` depois das cinco sementes). Mesma regra de arestas do `cortes_k20`: pesos brutos
das arestas internas, renormalização por destino sobre as arestas que restam, como os clientes do FedAvg.

## Nós sem vizinhos internos

Verificado antes do smoke: em k=166 **nenhum nó fica sem aresta de entrada interna** (7 998 arestas
dirigidas mantidas, 1 334 removidas; os arquivos de aresta dos 166 clientes são iguais à submatriz da
adjacência). O FedAvg k=166 rodou sem `allow_nodes_without_incoming`, o que só é possível nessa condição.
Portanto o comportamento efetivo do FedAvg (nó sem entrada → só o termo próprio `lin3`, sem aresta
artificial) não chega a ocorrer, e o controle não precisa de exceção nem de auto-laço: o termo próprio da
LEConv está presente em todos os nós, como no FedAvg. A verificação fica registrada em
`prelaunch_*.json` ("nós sem aresta de entrada interna": 0).

Equivalência numérica (local, 05/10): com os mesmos parâmetros e entradas, a saída do controle é a reunião
das saídas dos 166 modelos-cliente do FedAvg, diferença máxima 2,4e-7 (escala 16; arredondamento de
float32); perturbar o cliente 0 não altera nenhum nó de outro cliente.

## Desktop: verificações, smoke e concorrência

- Dados e ambiente: `sunt_graph_controls_check` no desktop (env `fedstgnn_env`, mesmas versões do
  Terumo); hashes de série, máscara, origens, nós, adjacência e scaler comparados com o Terumo.
- Condição do hardware (verificada em 05/10 antes do lançamento): RTX 2060 6 GB ociosa a 42 °C (lentidão
  a 90 °C, desligamento a 93 °C), CPU i9-9900K a 41 °C (alto 86 °C), 27 GB livres, disco com 126 GB
  livres; nenhum erro de GPU (Xid), OOM ou travamento nos boots registrados, inclusive na campanha de
  ~12,7 h a carga ~15/16 de 04–05/10.
- Smoke sem teste em `results/_smoke/sunt_control_cut166` com registro de temperatura e memória;
  retomada (1 época, depois 2) e checkpoint carregado na config final reproduzindo a validação.
- Concorrência definida pela memória de GPU e RAM medidas no smoke e pela temperatura sob carga, sem
  lançar as cinco sementes ao mesmo tempo sem essa medição. Durante a campanha, um vigia registra
  temperaturas; se a GPU passar de 85 °C ou o pacote da CPU de 90 °C de forma sustentada, os processos de
  treino são pausados (SIGSTOP) até esfriar (SIGCONT); a retomada não altera o treino.

## Hardware frente às referências

Centralizado completo: s7 na L40S, demais na A16 (Terumo). FedAvg k=166: s7 na L40S, demais na A16
(Terumo). Este braço: RTX 2060 (desktop) nas cinco sementes. A diferença de hardware entra nos contrastes
pareados (não determinismo do CUDA, sem TF32 nas três GPUs); é registrada, não corrigida. Tempos não são
comparados entre máquinas.

## Análise (acréscimo)

Mesmos insumos fixos de `protocol.md`, mais o FedAvg k=166 final (`results/sunt_fedavg_final/sunt/k166/`).
Estratos da partição k=166 de `results/sunt_stratified_analysis/v1/node_descriptors.csv` (fronteira/
interior, faixas de perda de vizinhança e de peso; volume e cobertura do treino, limites já fixados).
Contrastes pareados por semente, com os mesmos resumos e limitações:

1. **cortes_k166 − Centralizado completo:** sensibilidade às conexões entre clientes em k=166;
2. **FedAvg k=166 − cortes_k166:** diferença associada aos procedimentos de treinamento com
   disponibilidade de arestas correspondente; não isola uma causa (atualizações, lotes, agregação,
   1 000 rodadas no FedAvg contra 600 épocas no Centralizado, hardware).

Não formam decomposição causal aditiva. Contexto: FedAvg k=166 − Centralizado completo (já relatado).
