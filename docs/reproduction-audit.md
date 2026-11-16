# Auditoria de reprodução da dissertação

Conferência em 16 de novembro de 2026. Nenhum treinamento completo foi iniciado. O repositório permanece privado.

## Parecer

O repositório executa exemplos e os treinadores principais, mas **ainda não permite reproduzir integralmente os resultados da dissertação de maneira autocontida**. A validação de instalação não deve ser confundida com a reprodução do protocolo experimental.

A referência auditada é o pacote `generated-traffic` 1.29.0 da dissertação, procedente da revisão `c31798b5347c42bf041032dcdf113c2346c1d91c`. Seu CSV de resultados por semente contém 360 registros em 72 células: 195 principais, 45 do Local, 100 de controles e 20 complementares. Existem ainda análises adicionais em arquivos próprios. As 360 fontes de métricas foram encontradas no repositório experimental e seus hashes coincidiram com a procedência. Portanto, a ausência no repositório novo não representa perda dessas evidências.

A matriz de 48 células principais e de referência está em [main-experiments.csv](reproduction/main-experiments.csv). O [inventário](reproduction/inventory.json) lista as configurações ausentes e os hashes das nove partições de referência.

## O que falta

| Parte | Situação atual | Necessário para reprodução |
| --- | --- | --- |
| Dados | Carregadores dos três conjuntos presentes; METR-LA baixado e testado do zero | Fixar revisões das fontes e conferir hashes dos arquivos brutos, séries preparadas e grafos; verificar do zero SUNT e PEMS-BAY |
| Partições | Implementação original de busca Leiden presente; exemplos usam uma interface simplificada | Distribuir as nove atribuições de referência e a ordem dos nós, ou gerar e exigir igualdade com seus hashes |
| Configurações | 8 dos 23 caminhos referidos pela procedência estão presentes | Incorporar os 15 caminhos ausentes e consolidar o protocolo efetivo, incluindo opções que vieram da linha de comando |
| Comandos | Exemplos curtos e runners disponíveis | Receitas separadas por conjunto, método, partição e semente, com limites explícitos e sem executar variantes extras por engano |
| Avaliação de checkpoints | Avaliação adiada do Ditto existe; `evaluate_frozen.py` não foi incluído | Portar avaliação congelada do centralizado e FedAvg no SUNT e documentar a seleção pela validação |
| Evidência existente | Métricas, históricos e checkpoints não foram distribuídos | Publicar métricas/manifestos e definir um pacote de checkpoints separado, se desejada reprodução sem retreinar |
| Estatística e figuras | Exportadores e análises complementares ausentes | Portar a cadeia que gera médias, diferenças pareadas, Holm, sensibilidade, tabelas, figuras e macros, com suas dependências e famílias originais |
| Ambiente | Instalação atual testada em CPU | Distinguir ambiente de instalação do ambiente histórico de treinamento; registrar versões conhecidas e limites da reprodução numérica entre CPU/GPU |

As configurações ausentes incluem centralizado, FedAvg, FedProx e Local finais do SUNT, sondagens e controles de parada. Copiar somente os YAMLs não basta: o próprio arquivo de procedência ressalva que a versão da configuração associada ao commit do resultado é uma aproximação da usada na execução, não um registro completo dos argumentos.

## Parâmetros que não podem ser confundidos com os exemplos

- Sementes: **7, 42, 101, 123 e 314**; semente do particionamento: **42**.
- Rodovias: partições com 5, 10 e 20 clientes; janela/horizonte 12/12; LEConv com 64 unidades. Centralizado: limite de 100 épocas e paciência 10. Condições federadas: limite de 400 rodadas e paciência 30, além dos controles com regra ampliada.
- SUNT: 2.903 nós; entradas embarques e desembarques; alvo embarques; janela/horizonte 168/24. Pedidos de **5, 20 e 163** produzem **6, 20 e 166** clientes na busca original. Não substituir os pedidos por números obtidos sem validar a partição.
- SUNT: 600 épocas no centralizado; 600 rodadas para 6/20 clientes e 1.000 para 166, com seleção do checkpoint pela validação e sem parada antecipada nessas execuções.
- `rounds_by_clients` na configuração final do FedAvg é usado na avaliação congelada; o treinador não deve ser presumido capaz de aplicar esse mapa automaticamente. Os limites por célula precisam ser explícitos.
- Local nas rodovias foi registrado pelo caminho Ditto com coeficiente zero, ainda treinando um modelo global não usado na previsão local. No SUNT foi usado `local_only`. O exemplo novo de Local usa `local_only`; não se deve equiparar seus custos aos registros históricos das rodovias.
- A interface de exemplo usa Leiden com resolução fixa. A reprodução requer a busca original e conferência das atribuições, inclusive do tratamento dos sensores isolados.

## Dados e arquivos de referência

A preparação esperada resulta em 207 nós no METR-LA, 325 no PEMS-BAY e 2.903 no SUNT. As contagens de janelas de treino/validação/teste registradas são 23.974/3.424/6.851, 36.465/5.209/10.419 e 1.337/195/415, respectivamente. Esses valores são verificações úteis antes de treinar, mas não substituem hashes e a conferência da ordem dos nós.

O SUNT exige os 91 arquivos OD de março a maio de 2024; o carregador verifica a quantidade, mas acessa a revisão `main` da fonte. Os grafos rodoviários também são obtidos de fontes externas sem revisão Hugging Face fixada. A distribuição precisa tornar a versão dos insumos verificável, sem obrigatoriamente redistribuir os dados brutos.

As nove partições existem no repositório de pesquisa. Dos 360 caminhos de checkpoint registrados no CSV principal, 135 foram encontrados nesta cópia local; isso não significa que os demais tenham sido perdidos, pois há manifestos e armazenamento em outras máquinas. A recuperação e a conferência dos demais precisam ser feitas antes de anunciar reavaliação completa sem treino. Vinte registros do CSV são avaliações de checkpoints de sondagem, e não novos treinamentos.

## Sequência de comandos a documentar

**Plano de execução futura; não executado nesta auditoria.** Os comandos abaixo ilustram os runners existentes. As receitas finais precisam ser consolidadas e verificadas antes dos treinos completos.

1. Instalar o ambiente e obter os conjuntos:

```bash
python -m pip install -e '.[dev,notebooks]'
python -m data_loader --dataset all
```

2. Gerar as partições com a busca original e comparar os hashes antes de prosseguir:

```bash
python -m experiments.partition --dataset metr-la --clients 5 10 20 --seed 42
python -m experiments.partition --dataset pems-bay --clients 5 10 20 --seed 42
python -m experiments.partition --dataset sunt --clients 5 20 163 --seed 42
```

3. Nas rodovias, especificar as cinco sementes e a variante principal; por exemplo:

```bash
python -m experiments.run_grid --backend local --config config/fedgnnhet_leconv.yaml --datasets metr-la pems-bay --seeds 7 42 101 123 314 --clients 5 10 20 --halo 1hop
python -m experiments.run_grid --backend local --config config/ditto_leconv.yaml --datasets metr-la pems-bay --seeds 7 42 101 123 314 --clients 5 10 20 --lam per_client
```

Sem `--halo 1hop` ou `--lam per_client`, os YAMLs incluem outras variantes. Centralizado, FedAvg e FedProx precisam de receitas próprias; Local usa coeficiente zero e o caminho histórico apropriado.

4. Depois de incorporar as configurações SUNT ausentes, separar os pedidos 5/20 e 163 e aplicar `--max-epochs 600` e `--max-epochs 1000`, respectivamente, no `run_grid`. Esse argumento controla rodadas nos métodos federados. O centralizado usa 600 épocas. Preservar os fluxos de teste adiado quando a receita for baseada nos checkpoints existentes.

5. Para reavaliar os checkpoints de sondagem do SUNT, será necessário portar `experiments.evaluate_frozen` e suas configurações. O Local tem `--evaluate-test-from-checkpoint` no runner Ditto. Em ambos os casos, conferir o checkpoint selecionado e reproduzir a validação antes de avaliar o teste.

6. Portar o exportador e seus insumos, mantendo as famílias e o tratamento estatístico da dissertação. Apenas depois será possível regenerar `generated-traffic` no repositório novo. O exportador atual exige fontes versionadas e usa a história Git para parte da procedência; no repositório novo, esse vínculo deve ser preservado por manifesto, sem atribuir os commits novos aos experimentos antigos.

## Falha no GitHub

O envio ao GitHub funcionou. O workflow falhou em uma referência numérica de teste: diferença absoluta de 1,8249 × 10⁻⁹ em um parâmetro `float32`, maior que a tolerância absoluta de 10⁻⁹. O teste passou a considerar a precisão relativa do tipo (duas vezes epsilon), preservando os valores de referência e os testes de agregação ponderada. Nenhum cálculo de treinamento, resultado ou hiperparâmetro foi alterado. A execução corrigida do GitHub Actions (`43981998843`) concluiu com sucesso, incluindo a suíte e os dois notebooks.

## Ordem recomendada

Primeiro, preparar partições, configurações resolvidas e uma receita por célula. Depois, portar métricas e exportadores para reproduzir tabelas sem treinar. Por fim, realizar reproduções de treinamento selecionadas e comparar resultados com tolerâncias justificadas. Os experimentos completos devem esperar essas etapas, para evitar gastar processamento com um protocolo diferente do publicado.
