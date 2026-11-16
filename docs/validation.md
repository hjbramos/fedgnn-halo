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

**Estes são testes de funcionamento, não resultados adicionais da dissertação.** O orçamento curto, a arquitetura reduzida e a partição não correspondem ao protocolo principal. Esta validação não reproduziu as tabelas da dissertação nem os experimentos completos com SUNT ou PEMS-BAY. O acesso público por clone e a implantação do GitHub Pages serão conferidos após a autorização para tornar o repositório público.
