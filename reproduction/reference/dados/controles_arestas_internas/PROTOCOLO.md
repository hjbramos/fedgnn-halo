# Ampliação da cobertura dos controles — 14/11/2026

Solicitada durante a revisão da metodologia, após conhecimento dos resultados anteriores. Não é um protocolo retrospectivamente apresentado como pré-registrado. As configurações desta ampliação são fixadas antes dos novos resultados.

## Escopo

- Controle sem halo: SUNT k=6,20,166 e PEMS-BAY k=5,10,20.
- Centralizado com arestas internas: SUNT k=6 e METR-LA/PEMS-BAY k=5,10,20.
- Cinco sementes por célula: 7,42,101,123,314. Total: 13 células e 65 execuções.
- Não repetir células existentes. Não criar controle de parada federada para SUNT, cujo protocolo utiliza orçamento fixo.

## Protocolo preservado

As configurações derivam dos arquivos identificados em cells.json. Mesmos dados, máscaras, escalonamento, partições Leiden/seed42, arquitetura LEConv, inicializações por semente e critérios de seleção dos comparadores existentes. A seleção utiliza exclusivamente validação. Teste somente ao final do treinamento, no checkpoint selecionado, sem orientar ajustes ou orçamento.

Sem halo usa o executor FedGNNHet com halo=none. SUNT mantém 600/600/1000 rodadas para k=6/20/166; PEMS-BAY até 400 rodadas, paciência30, min_delta0.005. Centralizado com cortes mantém 600 épocas sem parada antecipada no SUNT; rodovias até100 épocas, paciência10, min_delta0.005. Arestas internas mantêm pesos brutos e são normalizadas por destino, sem adicionar conexões.

Smoke de uma época/rodada por célula, semente7, sem teste e com saída separada, é obrigatório antes de executar a campanha completa. Verificar ordem dos nós, número de grupos, ausência de nós sem entrada e igualdade dos dados/preparação entre cópia local e servidor. Se houver falha, interromper e investigar, sem alterar silenciosamente o protocolo.

## Execução e rastreabilidade

Utilizar cópia isolada do código vigente, com hashes e commit de origem, dados existentes via link e saídas próprias. Registrar ambiente, GPUs, logs e checkpoints. No máximo dois processos de treinamento simultâneos, três threads de CPU por processo. Não sobrescrever resultados anteriores. Resultados entre hardwares diferentes não sustentam comparação de tempo e a variação numérica de GPU permanece uma limitação.

## Análise após conclusão

Contrastes: halo versus none nas novas células; centralizado com cortes versus centralizado completo e FedAvg na mesma partição. Preservar todas as sementes e apresentar diferenças pareadas, IC95% t não ajustado e Wilcoxon exato como sensibilidade. As novas famílias de comparações serão identificadas como complementares: seis contrastes halo-none e quatorze contrastes de cortes, com Holm em cada família. Não reescrever os ajustes antigos como se as novas comparações já estivessem presentes. Não afirmar equivalência com base em ausência de diferença detectável.

A exportação, a validação do pacote e a atualização da dissertação ocorrerão após auditoria dos novos registros. Até lá, a tabela de cobertura continua descrevendo somente experimentos já concluídos.

## Emenda de execução — paralelismo, 14/11/2026

Por solicitação do autor, o limite operacional passa de dois para cinco treinos simultâneos, no máximo um por GPU. Esta emenda substitui apenas o limite de concorrência acima. Os dois treinos existentes são preservados com seus PIDs e progresso. Um despachante único assume as tarefas pendentes, priorizando o SUNT k166, sem mudar configurações científicas, sementes, critérios de seleção ou orçamentos.

As novas execuções usam três threads de CPU e prioridade nice5. Novos lançamentos aguardam carga média de um minuto inferior a42 (60 processadores lógicos), pelo menos24 GiB de RAM disponível, pelo menos4 GiB livres na GPU e ausência de outro processo de computação nessa GPU. Esses limites controlam admissões, sem interromper treinos em andamento. Se houver falha, novas admissões são bloqueadas. A variação de hardware e concorrência permanece registrada e não sustenta comparações de tempo entre métodos.

## Segunda emenda de execução — até dois treinos por GPU, 14/11/2026

Ampliação solicitada pelo autor após verificar a folga do servidor. O novo teto é de dez treinos simultâneos, no máximo dois por GPU, com uma nova admissão por ciclo de30 segundos. Processos científicos existentes são adotados sem reinício. As configurações científicas, dados, sementes e critérios permanecem intactos.

Novas admissões requerem: carga de1 minuto inferior a36, CPU global abaixo de65%, pelo menos32 GiB de RAM disponível, pelo menos8 GiB livres na GPU, utilização da GPU abaixo de70% e ausência de processos externos à campanha nessa GPU. Novos processos usam nice10 e três threads de CPU. Os limites só regulam novos lançamentos: não há suspensão automática de treinamentos ou garantia de ausência total de interferência em um servidor compartilhado. Se houver contenção, reduzir o teto em dual_limits.json e aguardar os processos terminarem, sem alterar os experimentos.
