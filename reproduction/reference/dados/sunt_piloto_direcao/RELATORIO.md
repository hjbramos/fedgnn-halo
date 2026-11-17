# Piloto exploratório: orientação do grafo do SUNT — relatório (só validação)

Protocolo fixado antes dos treinos em `PROTOCOLO.md`. **Exploratório**: cinco sementes, um conjunto de dados, só
validação (o teste não foi carregado nem avaliado); não altera o baseline oficial.

## Execução e conferências

- Terumo, worktree `fed-stgnn-traffic-direcao`, commit `48e774e` sem alterações locais; lançado em 28/10/2026
  15:43 UTC, sessão tmux `piloto_direcao`; 15 execuções com `rc=0` (última às 03:58 UTC de 29/10).
- torch 2.5.1+cu124, cuDNN 9.1, PyG 2.4.0; semente 7 na L40S, demais em A16 (três braços da semente na mesma GPU).
- `experiments/sunt_direcao_relatorio.py` sem problemas: teste não avaliado, determinismo estrito efetivo,
  adjacência igual à auditada, 600 épocas em todas, **mesma inicialização entre os três braços em cada semente**.
- Cópia para o MacBook conferida por `MANIFEST_terumo.sha256` (85 arquivos). O manifesto lista também os
  checkpoints (`best_model.pt`) e os logs de treino (`*.log`), que não são versionados (`.gitignore`); ficam no
  Terumo e na cópia local.

## MAE e RMSE de validação (escala original, embarques)

| semente | GPU | simétrico MAE / RMSE | observado MAE / RMSE | invertido MAE / RMSE | melhor época (s/o/i) | treino h (s/o/i) |
|---|---|---|---|---|---|---|
| 7 | L40S | 4,8562 / 21,732 | 4,9336 / 22,737 | 4,9622 / 23,194 | 551 / 507 / 577 | 2,93 / 3,12 / 3,27 |
| 42 | A16 | 4,9927 / 24,131 | 4,9498 / 23,447 | 4,9425 / 23,258 | 291 / 574 / 489 | 4,40 / 4,19 / 3,65 |
| 101 | A16 | 4,9189 / 23,385 | 4,9715 / 23,744 | 4,9729 / 23,650 | 558 / 345 / 563 | 4,46 / 4,19 / 3,49 |
| 123 | A16 | 4,9704 / 22,814 | 5,0040 / 23,516 | 4,9677 / 23,356 | 597 / 474 / 596 | 4,36 / 4,20 / 3,50 |
| 314 | A16 | 4,8799 / 22,824 | 4,9357 / 23,838 | 4,9523 / 23,170 | 554 / 600 / 581 | 4,37 / 4,19 / 3,51 |
| **média (dp)** | | **4,9236 (0,058) / 22,977** | **4,9589 (0,029) / 23,456** | **4,9595 (0,012) / 23,326** | | |

O tempo de treino é indicativo (servidor compartilhado). Várias melhores épocas ficam perto de 600, como no
protocolo vigente.

## Diferenças pareadas (por semente: 7, 42, 101, 123, 314)

| contraste | MAE média (dp) | % do MAE de referência | MAE por semente | RMSE média (dp) | % do RMSE de referência | RMSE por semente |
|---|---|---|---|---|---|---|
| observado − simétrico | +0,0353 (0,046) | +0,72 % | +0,077, −0,043, +0,053, +0,034, +0,056 | +0,48 (0,70) | +2,1 % | +1,00, −0,68, +0,36, +0,70, +1,01 |
| invertido − simétrico | +0,0359 (0,062) | +0,73 % | +0,106, −0,050, +0,054, −0,003, +0,072 | +0,35 (0,83) | +1,5 % | +1,46, −0,87, +0,26, +0,54, +0,35 |
| observado − invertido | −0,0006 (0,025) | −0,01 % | −0,029, +0,007, −0,001, +0,036, −0,017 | +0,13 (0,40) | +0,6 % | −0,46, +0,19, +0,09, +0,16, +0,67 |

**Por horizonte** (MAE, média das diferenças; detalhe em `relatorio_validacao.json`): nos horizontes 1–2 os dois
braços direcionados ficam ligeiramente abaixo do simétrico (−0,02 a −0,04; 3 de 5 sementes). Nos horizontes mais
longos ficam acima na maior parte dos passos, com picos de ~+0,08 (observado, h18) e ~+0,11 (invertido, h14). O
invertido fica abaixo também em h3 e h8–h10. Observado − invertido oscila em torno de zero, com sinal variável por
horizonte.

## Estratos (fixados pela topologia antes de examinar erros)

Em cada linha, os dois braços do contraste são avaliados sobre o **mesmo conjunto de nós** (o estrato indicado),
semente a semente.

| contraste | estrato | nós | MAE média (dp) | sementes com diferença < 0 |
|---|---|---|---|---|
| observado − simétrico | grande redução (do observado) | 669 | +0,071 (0,057) | 1/5 |
| observado − simétrico | demais (do observado) | 2 233 | +0,025 (0,044) | 1/5 |
| invertido − simétrico | grande redução (do invertido) | 658 | +0,072 (0,105) | 1/5 |
| invertido − simétrico | demais (do invertido) | 2 245 | +0,025 (0,049) | 2/5 |
| observado − invertido | grande redução do **observado** | 669 | +0,032 (0,029) | 0/5 |
| observado − invertido | grande redução do **invertido** | 658 | −0,044 (0,048) | 4/5 |
| observado − simétrico | sem entrada (do observado) | 1 | −24,7 (2,5) | 5/5 |

O nó sem entrada (índice 2365, stop_id 127813152) tem só 264 observações de validação (0,002 % do total). Seu MAE
cai de ~52 (simétrico e invertido) para ~28 no observado, onde ele usa só o termo próprio. Isso indica que, para
certas paradas, receber mensagens dos vizinhos pode prejudicar a previsão; mas é um único nó, sem peso no agregado,
e não sustenta conclusão geral.

## Leitura (descritiva)

1. **Simétrico × direcionados.** O simétrico teve os menores MAE e RMSE médios. Pelas diferenças pareadas, o
   observado fica +0,035 de MAE acima do simétrico (+0,72 %; RMSE +2,1 %), com diferença positiva em 4 das 5
   sementes (dp das diferenças 0,046). O invertido fica +0,036 acima (+0,73 %; RMSE +1,5 %), positivo em 3 de 5,
   sendo uma praticamente nula (−0,003). As diferenças são pequenas e variam entre sementes; a semente 42 vai no
   sentido oposto nos dois contrastes. Com cinco pares, isso descreve uma tendência, sem estabelecer sua magnitude.
2. **Observado × invertido.** As médias foram praticamente iguais (−0,0006 de MAE; diferenças com sinais mistos, 3 de
   5 negativas). Isso **não demonstra equivalência**: só não houve vantagem consistente de uma orientação sobre a outra.
3. **Estratos.** Em cada braço, o erro sobe mais nos nós que perdem entradas *naquele* braço: os estratos de grande
   redução trocam de sinal entre observado e invertido. Isso é **compatível** com uma perda ligada à redução das
   entradas, mas não a identifica: a mudança de topologia altera ao mesmo tempo a quantidade de vizinhos de entrada,
   a identidade desses vizinhos e os denominadores da normalização, e o piloto não separa esses fatores.
4. **Hipótese não testada.** O grafo simétrico deixa cada parada receber informação das duas vizinhas na rota, o que
   pode ser útil para prever embarques mesmo sem representar a circulação literalmente. É uma explicação possível,
   não demonstrada aqui.

**Conclusão.** Neste piloto centralizado do SUNT, restringir as mensagens aos sentidos observados não melhorou a
validação. O grafo simétrico apresentou os menores erros médios, e não houve vantagem consistente da orientação
observada sobre a invertida. Os resultados não identificam isoladamente o mecanismo dessa diferença. Com cinco
sementes, um único conjunto de dados e sem teste, nada se generaliza: o baseline oficial (grafo simétrico) é mantido,
e o piloto fica registrado como ablação informativa da direção do grafo.

## Reprodução

```bash
# no Terumo, raiz da worktree, commit 48e774e
python -m experiments.sunt_direcao_auditoria --saida results/sunt_piloto_direcao/auditoria
SESSION=piloto_direcao bash scripts/sunt_piloto_direcao.sh
python -m experiments.sunt_direcao_relatorio --base results/sunt_piloto_direcao
```

Configurações: `config/sunt_piloto_direcao_{simetrico,observado,invertido}.yaml`. Logs e códigos de saída: `logs/`.
