# Reproduzir a dissertação

Este guia distingue a recomposição das análises registradas da repetição dos treinamentos. A referência é o pacote 1.29.0, produzido pelo código experimental na revisão `c31798b5347c42bf041032dcdf113c2346c1d91c`. Nenhum dos comandos de verificação ou planejamento abaixo inicia treinamento.

## Ambiente

Na raiz de um clone autorizado do repositório privado:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev,notebooks,reproduction]'
python -m fedgnn_halo.reproduce verify
python -m pytest tests -q
```

Use Python 3.10–3.13. Para CUDA, instale a distribuição do PyTorch apropriada à sua GPU antes do projeto. Os testes de instalação usam CPU; isso não implica que os treinamentos completos, especialmente o SUNT, tenham custo viável em CPU. A licença MIT do código não substitui as licenças dos dados.

## Análises sem treino ou download dos dados brutos

```bash
python -m fedgnn_halo.reproduce export-reference --output results/reference-recomputed
```

O diretório de saída precisa ser novo. O comando confere os hashes das fontes, os 360 registros por semente, recalcula as 72 células e os 65 contrastes das famílias originais, exporta os controles complementares e as curvas de validação, e compara as 26 tabelas/macros com a referência. Também refaz a sensibilidade do Holm, sem alterar suas famílias ou incluir os 14 contrastes posteriores nessa sensibilidade. O relatório fica em `reproduction-report.json`.

Os exportadores usam as fórmulas originais. Como a distribuição não contém toda a história Git da pesquisa, a exigência de fontes presentes naquele histórico é substituída por um arquivo de evidências com manifesto SHA-256. As verificações de configurações, checkpoints dos controles, históricos, somas por nó e cálculos são mantidas. A procedência histórica aparece como `reference-provenance.json`, não como uma execução nova.

Os descritores estruturais dos conjuntos e as informações históricas do pacote são preservados, não recalculados a partir das bases por esse comando. A verificação independente dos dados e dos grafos está no fluxo seguinte. As imagens são renderizadas novamente; diferenças de fonte ou versão do Matplotlib podem modificar seus bytes sem alterar os pontos. O arquivo de evidências inclui os checkpoints necessários à auditoria dos controles, mas não todos os modelos históricos. Portanto, este fluxo reproduz as análises registradas, não reavalia todos os modelos antigos no teste.

## Dados e partições

```bash
python -m fedgnn_halo.reproduce prepare-data --dataset metr-la --data-dir data
python -m fedgnn_halo.reproduce prepare-data --dataset pems-bay --data-dir data
python -m fedgnn_halo.reproduce prepare-data --dataset sunt --data-dir data
```

Os carregadores obtêm as bases das fontes identificadas em [dados e experimentos](experiments.md). Os dados brutos não são redistribuídos. O SUNT usa 91 arquivos OD, de março a maio de 2024. Reserve espaço para os downloads, caches e checkpoints; somente os insumos brutos de referência somam aproximadamente 4,4 GB.

`data-manifest.json` registra os hashes dos 103 arquivos brutos usados. `prepared-manifest.json` verifica séries, máscaras, adjacências, timestamps e ordem dos nós: 207 no METR-LA, 325 no PEMS-BAY e 2.903 no SUNT. Se a fonte externa mudar ou faltar um arquivo, a preparação falha antes do treino; não substitua silenciosamente a versão dos dados. Os downloads de reprodução usam as revisões Hugging Face fixadas em `source-revisions.json`, além do registro Zenodo dos dados rodoviários. Os hashes conferem o conteúdo efetivo obtido nessas fontes.

```bash
python -m fedgnn_halo.reproduce verify-data --dataset all --data-dir data
python -m fedgnn_halo.reproduce install-partitions --data-dir data
```

As nove partições completas estão incluídas em `reproduction/partitions/`, com as atribuições, índices locais e metadados originais. A instalação recusa sobrescrever uma partição diferente. Não é necessário repetir a busca Leiden, cujos resultados podem depender da versão da biblioteca. A semente das partições é 42. Os pedidos 5, 20 e 163 do SUNT correspondem a 6, 20 e 166 clientes efetivos; nas rodovias, são 5, 10 e 20. O índice histórico do SUNT também menciona uma partição com dez clientes, que não integra as nove configurações distribuídas.

## Configurações e comandos de treinamento

`reproduction/catalog.json` reúne 86 receitas: as 72 células do pacote, os controles de arestas internas, o piloto de orientação e o Ditto com coeficiente fixo no SUNT, avaliado somente na validação. Cada receita aponta para seu YAML resolvido e sua configuração de origem. O planejamento inclui as sementes 7, 42, 101, 123 e 314.

```bash
python -m fedgnn_halo.reproduce list
python -m fedgnn_halo.reproduce plan --group principal
python -m fedgnn_halo.reproduce plan
python -m fedgnn_halo.reproduce train --experiment metrla_halo_k20 --seed 42
python -m fedgnn_halo.reproduce train --experiment sunt_fedavg_k166 --seed 42
```

**Os comandos acima só imprimem o plano.** Para uma execução futura, acrescentar `--execute --device cuda` ao comando `train` inicia aquele treinamento, após conferir dados e partições. `--data-dir` e `--output` permitem usar outros diretórios. Execute cada semente separadamente; o comando não dispara uma grade em paralelo.

| Condição | Limite e seleção |
| --- | --- |
| Rodovias, centralizado | 100 épocas, paciência 10 |
| Rodovias, métodos com clientes | 400 rodadas, paciência 30 |
| Rodovias, controle de parada ampliada | 400 épocas, paciência 30 |
| SUNT, centralizado principal | 600 épocas |
| SUNT, métodos com 6 ou 20 clientes | 600 rodadas |
| SUNT, métodos com 166 clientes | 1.000 rodadas |

A seleção continua pelo menor MAE de validação. Os YAMLs preservam janelas, atributos, alvo, arquitetura, lote, regularização e demais escolhas. Os limites efetivos são passados explicitamente ao treinador; o campo histórico `rounds_by_clients` não é tratado como se o treinador o aplicasse automaticamente. O arquivo resolvido por execução fica na saída.

No SUNT, 20 registros publicados resultam da avaliação congelada de checkpoints de sondagem (centralizado e FedAvg). Para uma repetição do zero, as receitas treinam com o mesmo orçamento e avaliam o modelo selecionado; não tentam recriar a história de transferência daqueles checkpoints. `experiments.evaluate_frozen` também está disponível para quem tiver os checkpoints de sondagem e quiser seguir o fluxo histórico. Não há segundo treino implícito nessa avaliação.

No Local do SUNT, a avaliação de teste ocorre em um segundo comando sobre o checkpoint selecionado, após o treino. O controle Ditto fixo do SUNT e os pilotos direcionados permanecem sem avaliação de teste. Nas rodovias, o Local conserva o caminho histórico com coeficiente zero e um modelo global não utilizado para prever; não deve ser substituído pelo exemplo simplificado `local_only` ao comparar custos históricos.

O diagnóstico complementar do Adam tem comando separado:

```bash
python -m experiments.adam_diagnostic
```

Por padrão ele só descreve o plano: três sementes (7, 42 e 101), dois braços, 100 épocas, sem acesso ao teste. O início futuro exige `--execute`, CUDA e dados verificados. Para um diretório externo, defina `FEDGNN_DATA_DIR` antes do Python e informe o mesmo caminho em `--data-dir`. O script preserva a comparação pareada da inicialização e da sequência de lotes, sem substituir qualquer receita principal.

Depois de executar as receitas escolhidas, compare os novos registros com a referência:

```bash
python -m fedgnn_halo.reproduce compare-results --results results/reproduced --output results/comparison
```

O relatório mostra as execuções ausentes e a diferença das novas métricas para as publicadas. Ele não preenche lacunas com resultados antigos. Se os 360 registros das 72 células estiverem presentes, as diferenças pareadas e o Holm são recalculados pelas funções originais; com registros incompletos, produz apenas a comparação descritiva, sem reduzir o tamanho das famílias. Os controles adicionais mantêm seus arquivos próprios e não são inseridos nessas famílias.

## Alcance da reprodução

Os testes de instalação, verificação das evidências e análise estatística não demonstram que novos treinamentos produzirão números idênticos. O ambiente histórico não foi registrado integralmente em todas as execuções, e operações em GPU podem variar entre placas e versões. Os registros mantêm os ambientes conhecidos; informações ausentes não foram preenchidas por inferência. A repetição dos treinos completos e a comparação de seus resultados continuam sendo uma etapa posterior.

O pacote de reprodução está disponível no clone, não dentro do wheel básico dos módulos. Para usar uma instalação não editável com os artefatos de um clone, defina `FEDGNN_REPRODUCTION_DIR` com o caminho absoluto para `reproduction/`.
