# generated-traffic

Pacote de resultados da dissertação, defesa e artigo, gerado por `python -m experiments.export_traffic` no
repositório `fed-stgnn-traffic`. **Não editar à mão**: regenere. Validação em `VALIDACAO.md`; procedência
(fontes, hashes, configs, commit, hardware por registro) em `provenance.json`.

## Conteúdo

- `dados/sunt_stratified_analysis/` e `dados/sunt_graph_controls/`: análises adicionais auditadas do SUNT,
  com protocolos, manifestos, auditorias, estratos e contrastes. As tabelas `sunt_estratos_fronteira.tex`,
  `sunt_controles_grafo.tex` e `macros_analises_sunt.tex` são próprias dessa etapa. IC95% exploratórios
  não ajustados, sem correção de multiplicidade; não alteram as famílias Holm originais. Figuras de
  prévia de layout não são exportadas. Não há novo treinamento no exportador.
- `dados/sunt_piloto_direcao/`, `tabelas/piloto_direcao.tex` e as macros `\TrafficSUNTDir...` (desde o 1.18.0, em
  `macros_analises_sunt.tex`): piloto de orientação do grafo do SUNT no centralizado com alvo de embarques: versões
  simétrica (vigente), direcionada (sentidos observados) e invertida, com a mesma afinidade por par. **Só validação**
  (o teste não foi avaliado; a validação também escolheu os checkpoints); não se compara com as métricas de teste.
  Descritivo: diferenças pareadas por semente sem IC, sem p e fora das famílias Holm.
- `dados/resultados_por_semente.csv`: um registro por célula e semente (métricas de teste, validação,
  iterações, tipo de registro, checkpoint de origem, hardware e hash da fonte). Ausente fica vazio.
- `dados/resumo_por_celula.csv`: média e desvio padrão amostral por célula.
- `figuras/panorama_relativo_centralizado.{pdf,png,svg}` e `dados/panorama_relativo_centralizado.csv` (1.28.0):
  três painéis com todos os métodos, incluindo Local; 100 × (MAE médio do método − MAE médio centralizado) /
  MAE médio centralizado. Sem barras de dispersão ou classificação estatística; escalas verticais distintas
  entre rodovias e SUNT, posições de k igualmente espaçadas e linhas apenas como guia visual.
- `dados/diferencas_pareadas.csv`: A − B por semente, IC95% (t pareado), p do t e Wilcoxon exato (sensibilidade).
- `dados/treinos_de_sondagem.csv`: treinos das sondagens cegas do SUNT (sem teste), cujos checkpoints foram
  avaliados nas células finais. Não são repetições adicionais.
- `dados/historico_fora_da_evidencia.csv`: registros históricos e exploratórios, fora da evidência.
- `dados/particoes.csv`: estrutura das partições Leiden (arestas dirigidas e pares cortados, nós de fronteira em
  qualquer sentido e nós com vizinho de entrada externo; definições no docstring de `partition_stats`).
  Desde o 1.26.1, `nos_sem_conexoes` conta os nós sem arestas com outros nós em nenhum dos sentidos,
  desconsiderando autolaços, e fornece as macros `...IsolatedNodes` dos três conjuntos.
- `dados/sunt_grafo_gtfs.csv` e as macros `\TrafficSUNTGTFS...` (desde o 1.26.0): cobertura do grafo do estudo no SUNT em
  relação ao grafo derivado do GTFS distribuído com o conjunto (`graph_gtfs_fev_2024.parquet`, SHA-256 e commit de origem no
  provenance): paradas em comum, paradas com séries ausentes do GTFS e o inverso, e pares não direcionados em comum ou
  exclusivos de cada grafo (autolaços do GTFS à parte). Descritivo; o grafo do GTFS não entra no treino.
- `dados/halo.csv`, `dados/comunicacao_por_execucao.csv`, `dados/datasets.csv`: nós virtuais, vizinho único,
  exatidão do halo e payload; comunicação de modelos por execução (contabilização do código, não tráfego medido);
  nós, janela, horizonte e número de janelas por split.
- `dados/custo_computacional_por_execucao.csv` e `dados/custo_computacional_por_celula.csv` (desde o 1.16.0): tempo de
  treino gravado pelo próprio treino (`training_seconds`), em horas e por iteração, com GPU e ambiente. **Descritivos**:
  GPU, concorrência com outros treinos e período variam entre registros, mesmo no mesmo servidor; não servem para
  ordenar métodos. A diferença estrutural de computação local (por exemplo, o Ditto com λ por cliente treina o modelo
  global e um modelo pessoal por candidato a cada rodada) está descrita nos métodos, não medida aqui.
- `tabelas/sunt_custo_k166.tex` e as macros `...TrainHours` e `...SecPerRound` (desde o 1.17.0): SUNT com $k=166$,
  comunicação estimada de modelos, payload do halo e tempo de treino (média ± dp entre sementes) e segundos por rodada,
  para FedAvg, FedProx, Ditto e FedGNN-Halo. Tempos indicativos, medidos no servidor compartilhado (GPUs A16 e L40S).
- `tabelas/*.tex`: fragmentos `tabular` (booktabs), sem legenda, números em `\num{}` (siunitx: o separador
  decimal segue cada documento), com rótulos em português (a versão em inglês
  para o artigo ainda não existe); `tabelas/macros_resultados.tex`: macros. As macros `...HaloK...RecoveryPct`
  existem só no METR-LA e são **descritivas** em todos os k: (FedAvg − halo) / (FedAvg − Centralizado) sobre as
  médias, não redução percentual de MAE e não evidência inferencial (em k=5 e k=10 a diferença halo − FedAvg não é
  detectável). No SUNT não há macro de recuperação.
- Diferenças pareadas: p nominais preservados e p com Holm dentro de famílias definidas **retrospectivamente**
  (principal = 9 contrastes halo × FedAvg, ampliada de 7 no 1.9.0 com o lote 1; secundária = halo × none no METR-LA; caracterização e comparadores à
  parte, exploratórios); os IC95% não são ajustados. A coluna `classificacao` separa, do ponto de vista de A,
  melhora ou piora observada, com evidência nominal ou com evidência após ajuste.
- Checkpoints: `checkpoint_sha256` identifica o arquivo associado a cada resultado (rastreabilidade), não prova que
  ele produziu a métrica (reprodução).
- `figuras/*.{png,svg}` e `figuras/sunt_evolucao_validacao_{federados,local}.{pdf,png,svg}` (mais `*_pb.png`, desde o 1.21.0):
  MAE por k (SUNT e METR-LA), diferenças pareadas no SUNT e duas figuras separadas da evolução do MAE de validação no SUNT
  sob orçamento limitado, dimensionadas em 16,0 cm (Figura A: Centralizado e métodos federados com linha de trajetórias
  completas e linha de ampliações do trecho final; Figura B: Local sob escala comum), acompanhadas de
  `dados/sunt_evolucao_validacao_por_semente.csv`, `dados/sunt_evolucao_validacao_resumo.csv` e
  `dados/sunt_evolucao_validacao_contagens.json`.

## Figuras e RMSE adicionais (1.29.0)

As figuras `halo_fracao_cortes` e `contrastes_controles_centralizados` usam os contrastes pareados já auditados, com IC95% sem ajuste. A escala relativa divide a diferença e os limites pela média do MAE do FedAvg; não estima contribuições causais. Os CSVs correspondentes preservam a precisão numérica. As tabelas `metrla_rmse` e `pemsbay_rmse` apresentam médias e desvios existentes com três casas, incluindo Local. O eixo do Local usa épocas por modelo.

## Grupos

`principal` (comparação principal), `controle` (halo none, 1hop_min2, Ditto com λ fixo, Centralizado sem
arestas), `referência` (Local = Ditto λ=0) e `complementar` (carga com grafo direcionado; FED-PUB adaptado).
As tabelas principais `metrla_mae.tex`, `pemsbay_mae.tex`, `sunt_mae.tex` e `sunt_rmse.tex` têm o Centralizado numa
coluna própria, "Sem partição", e FedAvg, FedProx, Ditto e FedGNN-Halo nas colunas por k (desde o 1.14.0;
controles retirados no 1.11.0, Local no 1.12.0). Desde o 1.19.0, a menor média de cada coluna de k fica em negrito,
com as casas impressas (empates em negrito nos dois); o Centralizado, sozinho na sua coluna, não. Em `carga_direcionado.tex`,
o negrito marca a melhor média de cada métrica. É descritivo, e não indica diferença detectável. Local e controles ficam em `dados/resumo_por_celula.csv`, nas macros
e nas diferenças pareadas; `sunt_principal.tex` não muda. A macro `...RMSEStd` (desvio do RMSE) existe desde o 1.13.0.
O Ditto fica visível nas tabelas do METR-LA e do PEMS-BAY, com k=5, 10 e 20 nos dois (no PEMS-BAY, k=10 desde o
1.15.0; o Ditto k=20 foi treinado na estação de trabalho, e a referência FedAvg, no servidor). No SUNT, FedProx e
Ditto têm k=166 desde o 1.16.0 (lote 5), e a exploratória do SUNT passa de 4 para 6 comparações. Desde o 1.20.0
(lote 6), o SUNT tem o Local (`sunt_local_k6/20/166`, caminho `ditto.local_only`: só os modelos pessoais
com λ=0, sem treinar o modelo global, com a mesma trajetória pessoal do Ditto λ=0; `results/lote6_sunt_local/`), fora
das tabelas por k, como nas rodovias; a exploratória do SUNT passa de 6 para 12 comparações (Local − FedAvg e
Local − halo). O tempo do Local do SUNT não é comparável ao do Local das rodovias, que ainda treinava o modelo global.

## Ambientes

Servidor do laboratório e estação de trabalho; hardware por registro na procedência. Não comparar tempos
entre ambientes; dentro do mesmo servidor, os tempos também dependem da GPU e da concorrência de cada período
(`dados/custo_computacional_*.csv`). Registros antigos sem GPU gravada constam como "não registrado".
