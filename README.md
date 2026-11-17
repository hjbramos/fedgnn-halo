# FedGNN-Halo

Código para estudar previsão espaço-temporal federada em grafos particionados. O FedGNN-Halo representa mensagens de vizinhos externos por nós virtuais de fronteira. O projeto reúne modelos, treinamento local, agregação federada, agrupamentos e carregamento dos conjuntos SUNT, METR-LA e PEMS-BAY.

**Esta distribuição está em preparação.** O exemplo sintético permite começar sem dados externos. Executar o exemplo não equivale a reproduzir os resultados da dissertação. O repositório permanece privado durante a preparação para publicação.

## Começar do zero

Use Python **3.10 a 3.13**; recomendamos 3.11 ou 3.12. No terminal, na pasta em que deseja guardar o projeto:

```bash
git clone https://github.com/hjbramos/fedgnn-halo.git
cd fedgnn-halo
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev,notebooks,reproduction]"
fedgnn-halo --config examples/synthetic.yaml
```

Enquanto o repositório estiver privado, o clone exige acesso autorizado. No Windows, use `.venv\Scripts\activate` no lugar de `source`. O primeiro exemplo usa CPU, 12 nós, três clientes e três rodadas. Não requer GPU nem download de bases. Se o diretório de saída já existir e contiver arquivos, escolha outro com `--output`.

```bash
fedgnn-halo --config examples/synthetic.yaml --method fedavg --output results/example-fedavg
fedgnn-halo --config examples/synthetic.yaml --method local --output results/example-local
fedgnn-halo --config examples/synthetic-gru.yaml
python -m pytest tests -q
jupyter lab notebooks/01_quickstart.ipynb
```

O [segundo notebook](notebooks/02_methods_and_partitions.ipynb) compara seis métodos e dois agrupamentos. Ambos têm saídas limpas e podem ser executados desde a primeira célula.

## Exemplos curtos com dados reais

Com o ambiente ativado, execute estes comandos na raiz do repositório para comparar centralizado, FedAvg e FedGNN-Halo no METR-LA:

```bash
fedgnn-halo --config examples/metr-la.yaml --method centralized --output results/metr-centralized
fedgnn-halo --config examples/metr-la.yaml --method fedavg --output results/metr-fedavg
fedgnn-halo --config examples/metr-la.yaml --method halo --output results/metr-halo
```

O primeiro comando baixa automaticamente os dados das fontes públicas e os prepara em `data/`; os seguintes reutilizam esses arquivos. É necessário acesso à internet no primeiro uso, mas não copiar dados de outro projeto. O CSV principal tem aproximadamente 73 MB, além dos arquivos do grafo e dos dados preparados.

Os três exemplos usam CPU, uma arquitetura reduzida e apenas **uma época no centralizado ou uma rodada nos métodos federados**. Eles verificam o funcionamento com dados reais; não reproduzem o orçamento nem os resultados da dissertação. O número de clientes é determinado pelo Leiden com resolução 1,0, sem usar as partições congeladas do protocolo principal.

Cada diretório de saída contém `summary.json` com as métricas, as configurações e os arquivos da execução. Para repetir um exemplo, escolha outro diretório com `--output`, pois saídas existentes não são sobrescritas.

## Alterar o experimento

O YAML define os pontos de configuração:

| Campo | Opções e finalidade |
| --- | --- |
| `method` | `centralized`, `local`, `fedavg`, `fedprox`, `ditto`, `halo` |
| `model` | LEConv, GRU, LSTM ou DCRNN e seus parâmetros |
| `partition` | `contiguous`, `random`, `leiden` com resolução, ou `file` com rótulos `.npy` |
| `training` | Épocas, rodadas, épocas locais, taxa de aprendizado, paciência e regularização |
| `dataset` | `synthetic` ou um dos três conjuntos reais, janela e horizonte |
| `seed`, `device`, `threads` | Semente, CPU/CUDA e limite de threads |

O halo implementado exige **uma camada LEConv**. Trocar a arquitetura não estende automaticamente sua propriedade de recuperação das mensagens. Leiden determina o número efetivo de grupos pela resolução; `clients` define o número nos agrupamentos contíguo e aleatório. Uma partição externa deve seguir exatamente a ordem dos nós do conjunto.

Cada execução grava configuração solicitada e resolvida, versões do ambiente, partição, histórico, modelo selecionado pela validação e métricas. Os diretórios de saída não são sobrescritos pela entrada `fedgnn-halo`.

## Organização

```text
fedgnn_halo/   entrada de execução, exemplo sintético e agrupamentos intercambiáveis
models/        arquiteturas e fábrica build_model(ModelSpec, ...)
federation/    agregação de parâmetros, rodadas, objetivo local e agregados do halo
data_loader/   preparação e carregamento dos conjuntos
experiments/   treinamento, validação, checkpoints e métricas
examples/      configurações pequenas para começar
config/        configurações de referência importadas do código experimental
reproduction/  receitas resolvidas, partições, evidências e manifestos da dissertação
notebooks/     exemplos de uso da API
tests/         contratos dos módulos, regressões e integração
docs/          página do projeto e guias
```

`models` não depende dos carregadores. `federation` recebe funções de construção do modelo, otimizador e treino local. Para acrescentar uma arquitetura, implemente-a em `models`, registre-a em `build_model` e adapte `experiments/task.py` se a assinatura de previsão mudar. Para um novo agrupamento, estenda `fedgnn_halo/partition.py` ou forneça os rótulos com `method: file`. Para um novo objetivo local, use `ObjectiveResult` e `train_local_epoch`, mantendo a agregação separada.

## Dados reais e reprodução

O [guia de reprodução](docs/reproduction.md) reúne as configurações resolvidas, as nove partições verificadas e os comandos por experimento e semente. É possível recalcular as análises registradas sem iniciar treinamentos:

```bash
python -m fedgnn_halo.reproduce verify
python -m fedgnn_halo.reproduce export-reference --output results/reference-recomputed
python -m fedgnn_halo.reproduce list
python -m fedgnn_halo.reproduce train --experiment metrla_halo_k20 --seed 42
```

O último comando só mostra o plano; treinar exige `--execute`. Os dados brutos são obtidos das fontes originais e conferidos por hash antes dos treinos. Consulte [dados e experimentos](docs/experiments.md) para as fontes. O nome interno `fedgnnhet` corresponde ao FedGNN-Halo e foi mantido nos runners para compatibilidade.

A simulação é sequencial por padrão. As estimativas de comunicação não medem tráfego de rede; os agregados do halo são preparados com acesso centralizado aos dados. Eles não oferecem, por si, proteção das séries. O método Local usa o ramo independente do treinador de modelos pessoais, com coeficiente zero e sem troca de parâmetros.

## Procedência e licença

`SOURCE.json` registra a revisão e os hashes do código experimental usado como base. Os módulos de distribuição e a injeção de dados/partições nos treinadores foram acrescentados sem mudar as fórmulas de agregação. As referências às arquiteturas e aos métodos foram preservadas. Veja [atribuições](docs/attributions.md).

O código próprio está disponível sob a [licença MIT](LICENSE). As licenças dos conjuntos e das dependências continuam independentes. Não há autorização de redistribuição dos dados implícita neste repositório.
