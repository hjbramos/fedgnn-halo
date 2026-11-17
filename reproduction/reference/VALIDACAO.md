# Relatório de validação do pacote `generated-traffic`

- Resultado: **VÁLIDO**
- Exportador 1.29.0 (experiments/export_traffic.py), commit `c31798b`
- Gerado em 2026-11-16T16:30:37-03:00

- 360 registros selecionados em 72 células; 65 comparações pareadas; 1240 macros.

- Ampliação de arestas internas: 35 registros em sete células adicionais e 14 contrastes em uma família complementar de Holm, exportados separadamente em dados/controles_arestas_internas/.

## O que o exportador verifica (todas as conferências passaram)

1. Cada célula não opcional tem exatamente as sementes 7, 42, 101, 123 e 314, sem duplicatas.
2. Todo registro selecionado tem MAE de teste finito.
3. Metadados gravados coerentes com o papel da célula: alvo, entradas, grafo, janela e horizonte.
4. Nós, janela e horizonte únicos por dataset e alvo entre as células selecionadas.
5. SUNT principal: 600 épocas no Centralizado e 600 / 600 / 1 000 rodadas em k = 6 / 20 / 166. Nas avaliações de checkpoint de sondagem (Centralizado e FedAvg), a sondagem de origem existe, tem a mesma semente, melhor rodada e total de rodadas, não tem teste, e o `best_model.pt` associado a ela tem hash no manifesto.
6. Consistência numérica, com tolerância relativa 1e-09: MAE, RMSE e R² de cada registro conferem com o metrics.json de origem, cujo hash confere; a partir dos CSVs exportados, os resumos (média e dp de MAE, RMSE e R²) e as diferenças pareadas (média, IC95%, p do t e p do Wilcoxon) foram recalculados e conferem; tabelas e macros re-renderizadas a partir desses valores são idênticas às do pacote.
7. Estatísticas estruturais (particoes.csv e macros de corte): cada execução federada aponta para a partição usada nas estatísticas e tem os mesmos nós por cliente (`client_nodes`) que `bincount(client_assignments.npy)`; os hashes das atribuições, de `node_ids.npy`, `metadata.json`, `partitions.csv` e do conteúdo da adjacência estão em `insumos_estruturais` no provenance e foram reconferidos contra o disco. O metrics.json não guarda hash da partição: o vínculo é por caminho e tamanhos.
8. Duas gerações independentes, cada uma recarregando as fontes do disco, produziram arquivos idênticos (todos, inclusive provenance.json e este relatório, exceto o carimbo de data).

## O que não é verificado

- Arquitetura, normalização e configuração efetiva de cada treino (a partição é conferida como no item 7). Por registro, há o hash da config na revisão do commit que versionou o resultado, que aproxima a versão usada, sem garanti-la.
- Que o checkpoint produziu a métrica: o manifesto identifica o `best_model.pt` ASSOCIADO a cada resultado (rastreabilidade), mas sem vínculo gravado na avaliação original nem reprodução da avaliação ele não demonstra, sozinho, que aquele arquivo produziu o número (reprodução). O exportador não carrega checkpoints.
- Comparabilidade entre células: hashes atestam integridade, não comparabilidade experimental.
- Multiplicidade: os p ajustados (Holm) seguem famílias definidas retrospectivamente, depois de conhecer os resultados; os IC95% não são ajustados (ver `estatistica` no provenance).

## Observações (não invalidam)

- pemsbay_halonone_k5: 5 checkpoint(s) sem hash no manifesto

## Análises adicionais do SUNT

Estratificação e controles centralizados de grafo: hashes das métricas, scripts e protocolos conferidos; MAE e denominadores de cada estrato reconstruídos a partir de per_node; diferenças das cinco sementes, médias, desvios e IC95% recalculados. Auditorias de origem sem falhas. São análises exploratórias com IC95% não ajustados, fora das famílias Holm originais. Procedência em analises_sunt no provenance.json; tabelas, macros e figuras próprias no pacote.

## Piloto de orientação do grafo do SUNT (só validação)

Insumos versionados e com hash; auditoria das topologias sem falhas; 15 execuções com rc=0, 600 épocas, determinismo estrito, teste não avaliado, adjacência igual à auditada e mesma inicialização entre os braços de cada semente. MAE e RMSE reconstituídos de validation_detail.npz e conferidos com o metrics.json; diferenças pareadas e por estrato recalculadas e conferidas com o relatório do piloto. Descritivo: sem IC, sem p e fora das famílias Holm. Procedência em piloto_direcao_sunt no provenance.json.

## Evolução do MAE de validação no SUNT (orçamento limitado)

80 execuções selecionadas nas 16 células do SUNT (Centralizado  FedAvg  FedProx  Ditto  FedGNN-Halo e Local em k = 6  20 e 166; sementes 7  42  101  123 e 314): todos os `history.csv`  `metrics.json` e `config/*.yaml` existem  estão versionados no git e conferem em orçamento (600 épocas no Centralizado; 600  600 e 1 000 rodadas em k = 6  20 e 166)  número de observações de validação por etapa (10 867 874) e seleção de checkpoint (`argmin(validation_MAE) == melhor_iteracao`, coincidente com `metrics.json`). Nas 20 execuções avaliadas a partir de sondagens cegas (`sunt_cent` e `sunt_fedavg_k*`), o `history.csv` final tem o mesmo SHA-256 do `history.csv` da sonda de origem. No Ditto, utiliza-se explicitamente a validação dos modelos personalizados (`validation_MAE`, distinta de `validation_global_MAE`). Contagens recalculadas automaticamente dos históricos: em k = 166, 19/20 execuções federadas selecionaram o checkpoint nos últimos 10% do orçamento (rodadas >= 901) e 18/20 a partir da rodada 983 (inclusive); no Local, 15/15 execuções terminaram com MAE de validação acima do checkpoint selecionado. O exportador garante a **reprodução determinística das figuras e estatísticas** a partir dos históricos e métricas versionados, sem executar nem reproduzir treinamentos. Procedência e hashes dos 100 arquivos `history.csv` auditados em `evolucao_validacao_sunt` no `provenance.json`.
