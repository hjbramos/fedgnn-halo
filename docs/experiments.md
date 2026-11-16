# Dados e experimentos

## Primeiro teste

Comece por `examples/synthetic.yaml`. O comando executa os mesmos treinadores usados pelos módulos experimentais, com dados e partição fornecidos explicitamente. O exemplo usa cortes temporais sem sobreposição de alvos entre treino, validação e teste; a padronização usa apenas o treino. Os protocolos dos conjuntos reais mantêm suas próprias divisões.

## Bases reais

- **SUNT:** carregador em `data_loader/sunt.py`, fonte `labiaufba/PublicTransportationSunt` no Hugging Face. Constrói séries horárias de embarques/desembarques e o grafo a partir das transições no período de treino. O conjunto original e o grafo construído são objetos distintos.
- **METR-LA e PEMS-BAY:** carregadores `data_loader/metr_la.py` e `data_loader/pems_bay.py`, arquivos de referência no Zenodo (registro 5146275). Usam velocidades e grafos alinhados à ordem dos sensores.

Não é necessário copiar uma pasta de outro projeto. Os carregadores baixam os arquivos faltantes e preparam os dados em `data/`, ou no caminho definido por `FEDGNN_DATA_DIR`. Conserve espaço livre para arquivos brutos e processados. Leia os termos das fontes antes de redistribuir dados.

```bash
python -m data_loader --dataset pems-bay
fedgnn-halo --config examples/pems-bay.yaml
```

Esse exemplo faz apenas duas rodadas. É uma verificação de execução com dados reais, não um orçamento adequado para comparar desempenho. O download inicial pode ser grande; não é realizado nos testes rápidos.

## Protocolo experimental original

Os runners também podem ser usados diretamente:

```bash
python -m experiments.partition --dataset pems-bay --clients 5 10 20 --seed 42
python -m experiments.run_grid --config config/fedavg_leconv.yaml --backend local --datasets pems-bay --seeds 7 --clients 5 --max-epochs 2
```

Essa interface preserva a busca de resolução Leiden do protocolo original, que escolhe a partição mais próxima do número solicitado. Verifique o número efetivo de clientes no arquivo de metadados. A entrada simplificada `fedgnn-halo` usa resolução fixa quando `partition.method: leiden`; portanto, esses agrupamentos não são intercambiáveis para reproduzir uma tabela.

Antes de reproduzir os resultados completos, é necessário fixar a versão dos dados, a partição, as cinco sementes, os limites efetivos de épocas/rodadas, o critério de parada e as opções de cada método. Os YAMLs históricos incluem campanhas diferentes e não devem ser tratados como comandos finais sem conferir esses parâmetros. GPU e bibliotecas diferentes também podem introduzir variação numérica.

## Arquiteturas e métodos

LEConv aceita `hidden_size`. GRU/LSTM aceitam também `num_layers`, `cl_decay_steps` e `use_curriculum_learning`. DCRNN exige `hidden_size`, `num_layers`, `max_diffusion_step` e `cl_decay_steps`; consulte `config/centralized_dcrnn.yaml`. Arquiteturas recorrentes não são compatíveis com o halo disponível.

No FedProx, use `training.proximal_mu`. No Ditto, use `training.lambda: per_client` e `lambda_candidates`, ou um valor fixo. A seleção por cliente usa a validação, não o teste. O treino local independente fixa o coeficiente em zero e desabilita o modelo global no ramo `local_only`.

## Teste de instalação isolada

Clone em um diretório novo, crie um venv sem `--system-site-packages` e siga os comandos do README. Execute a suíte, os dois notebooks e os métodos sintéticos. Não reutilize `data/`, `results/` ou pacotes de outro projeto. O relatório em `docs/validation.md`, quando disponível, registra o alcance efetivamente verificado; não implica reprodução integral da dissertação.

A variante exploratória `leconv_edge` depende de atributos operacionais externos preparados e não integra os exemplos autocontidos. Ela exige `model.edge_attributes_root` ou `FEDGNN_EDGE_ATTRIBUTES_DIR`, sem procurar outros repositórios no computador.

O exemplo sintético habilita `model.allow_nodes_without_incoming: true`: em uma partição aleatória, um nó pode perder todos os vizinhos locais. Nesse caso, a mensagem dos vizinhos é zero e a LEConv mantém o termo do próprio nó, sem criar novas arestas.

## Versões verificadas

`requirements/desktop-cpu-py313.txt` registra o ambiente Linux/CPU com Python 3.13 usado na validação. Nesse ambiente, pode-se instalar as versões registradas com `python -m pip install -r requirements/desktop-cpu-py313.txt` e depois `python -m pip install --no-deps -e .`. Para macOS, Windows ou outra versão de Python, use o procedimento do README, pois as distribuições binárias podem ser diferentes.

O teste com dados reais pode ser repetido com `fedgnn-halo --config examples/metr-la.yaml --method centralized --output results/metr-centralized` e, em diretórios distintos, com `--method fedavg` e `--method halo`. Esse exemplo usa uma época/rodada, arquitetura reduzida e resolução Leiden fixa, conforme o relatório de validação.
