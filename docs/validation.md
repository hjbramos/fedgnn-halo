# Validação da distribuição

Teste realizado em 16 de novembro de 2026, em Linux x86_64, Python 3.13.11 e PyTorch 2.15.0+cpu, com uma thread de CPU por treinamento.

## Instalação isolada

O código versionado foi transferido por Git bundle e clonado em uma pasta nova. Esse transporte foi usado porque o repositório estava privado e a máquina de teste não tinha credenciais do GitHub. O ambiente foi criado com `python -m venv`, sem acesso aos pacotes de outros ambientes. As dependências foram instaladas pela configuração do projeto, com PyTorch obtido do índice de CPU. Nenhum dado ou resultado de pesquisa existente foi reutilizado.

- `pip check`: nenhuma dependência quebrada.
- Suíte: **70 testes aprovados**, incluindo os seis métodos, agrupamentos contíguos/aleatórios, determinismo das partições, GRU, LSTM, DCRNN e propriedades do halo.
- Os dois notebooks executaram todas as células.
- O pacote wheel foi construído e instalado; o exemplo do halo executou fora da pasta do código, importando de `site-packages`.
- As versões instaladas estão em `requirements/desktop-cpu-py313.txt`.

O ambiente emitiu avisos de depreciação de PyTorch/PyTorch Geometric e um aviso do kernel Jupyter sobre transporte TCP local. Não houve falha nos testes concluídos. Windows e GPU não foram validados nesta rodada.

## Dados reais

O METR-LA foi baixado anonimamente das fontes públicas, preparado e carregado com 34.272 instantes e 207 sensores. O exemplo de `examples/metr-la.yaml` foi executado com centralizado, FedAvg e Halo, uma época/rodada por método e resolução Leiden 1,0, que produziu nove clientes.

| Método | MAE de teste (mph) | Tempo de treinamento aproximado |
| --- | ---: | ---: |
| Centralizado | 3,957581 | 7,6 s |
| FedAvg | 4,246937 | 19,4 s |
| FedGNN-Halo | 4,177493 | 28,5 s |

A verificação das mensagens do halo, sob parâmetros comuns, registrou diferença máxima de aproximadamente 2,38 × 10⁻⁷ em relação ao centralizado. Os tempos são registros de execução, não uma comparação de eficiência.

**Estes são testes de funcionamento, não resultados adicionais da dissertação.** O orçamento curto, a arquitetura reduzida e a partição não correspondem ao protocolo principal. Esta primeira rodada não reproduziu as tabelas da dissertação nem os experimentos completos com SUNT ou PEMS-BAY. O acesso público por clone e a implantação do GitHub Pages serão conferidos após a autorização para tornar o repositório público.

## Verificação do pacote de reprodução

Na segunda rodada, o mesmo ambiente isolado do desktop recebeu os artefatos versionados por Git bundle, sem acesso ao repositório original da pesquisa. A instalação foi atualizada com o extra `reproduction`.

- **160 testes passaram**, incluindo a interpretação das 86 receitas pelos runners, sem chamar as funções de treinamento, e a recusa de partições alteradas.
- A recomposição conferiu os 360 registros, recalculou as 72 células e os 65 contrastes, executou as auditorias dos controles complementares e das curvas e reproduziu exatamente as 26 tabelas/macros da referência. A sensibilidade de Holm também coincidiu.
- METR-LA: dados brutos, séries preparadas, máscaras, grafo, timestamps e ordem dos nós conferidos no desktop.
- PEMS-BAY: download das fontes públicas nas revisões fixadas e preparação do zero concluídos no desktop; todos os hashes brutos e preparados coincidiram.
- SUNT: os 91 arquivos OD foram baixados da revisão pública fixada e as séries e o grafo foram reconstruídos do zero no desktop. Todos os hashes brutos e preparados coincidiram, incluindo máscaras, timestamps e ordem dos 2.903 nós. A consulta independente aos metadados LFS também confirmou os hashes dos arquivos brutos.
- As nove partições foram instaladas e verificadas. Os comandos SUNT com 166 clientes e do diagnóstico do Adam permaneceram em modo de planejamento. O pequeno autoteste do Adam confirmou a equivalência entre limpar seu estado e reconstruir o otimizador; o diagnóstico com os dados do SUNT não foi treinado.

O relatório da recomposição separa a procedência histórica da nova exportação. Os descritores estruturais são preservados da referência e têm verificação independente pelo comando de dados. O ambiente desta rodada está em `requirements/reproduction-desktop-cpu-py313.txt`.

Nenhum treinamento completo foi iniciado. Repetir os treinamentos e comparar suas métricas permanece uma etapa posterior; os testes acima verificam a distribuição, os insumos e o recálculo da evidência existente. O repositório permaneceu privado e o GitHub Pages não foi publicado.

A automação do GitHub também concluiu com sucesso na revisão `6a68b70` ([execução 43987527386](https://github.com/hjbramos/fedgnn-halo/actions/runs/43987527386)), incluindo os 160 testes, os dois notebooks e a recomposição das análises. O comando de comparação de novas execuções foi conferido com os 360 registros originais, reproduzindo os 65 contrastes; um teste adicional verifica que registros incompletos não provocam um ajuste de Holm sobre famílias menores.
