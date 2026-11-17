"""Audita e exporta a ampliação de arestas internas, sem alterar famílias anteriores."""
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from scipy import stats

from .train import ROOT

BASE = ROOT / 'results/completar_controles_20261114'
CFG = ROOT / 'config/completar_controles_20261114'
SEEDS = [7, 42, 101, 123, 314]


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def require(ok, message):
    if not ok:
        raise ValueError('Controles com arestas internas: ' + message)


def export(dest):
    proc = BASE / 'procedencia'
    remote = json.loads((proc / 'remote_hashes.json').read_text())
    for path, expected in remote.items():
        require(digest(ROOT / path) == expected, 'hash remoto divergente: ' + path)
    audit = json.loads((proc / 'audit.json').read_text())
    require(audit == json.loads((CFG / 'audit_local.json').read_text()), 'dados do preflight divergentes')
    source = json.loads((proc / 'source_manifest.json').read_text())
    for path in ['experiments/train.py', 'experiments/task.py', 'experiments/data.py', 'models/leconv.py']:
        require(digest(ROOT / path) == source['files'][path], 'código de treinamento divergente: ' + path)
    principal = pd.read_csv(dest / 'dados/resultados_por_semente.csv')
    cells = [c for c in json.loads((CFG / 'cells.json').read_text()) if c['kind'] == 'cut']
    rows, sources = [], {}
    for cell in cells:
        ds, k, name = cell['dataset'], cell['k'], cell['name']
        config_path = ROOT / cell['config']
        require(digest(config_path) == digest(proc / 'config' / config_path.name), 'configuração remota diferente')
        cfg = yaml.safe_load(config_path.read_text())
        tr, window = cfg['training'], cfg['windows'][ds]
        files = sorted((BASE / name).glob('*/seed_*/metrics.json'))
        require(len(files) == 5, name + ': número de registros')
        seeds = []
        for p in files:
            m = json.loads(p.read_text()); seed = m['seed']; seeds.append(seed)
            hist = pd.read_csv(p.with_name('history.csv'))
            require(m['dataset'] == ds and m['method'] == 'centralized' and m['model'] == 'leconv', name + ': método')
            require(m['model_options'] == cfg['model'], name + ': opções do modelo')
            require(m['batch_size'] == cfg['data']['batch_size'] and m['input_steps'] == window['input_steps']
                    and m['horizon'] == window['horizon'], name + ': lote/janela')
            require(m['model_edges'] == audit[ds]['partitions'][str(k)]['edges'], name + ': arestas efetivas')
            assign = np.load(ROOT / cfg['model']['cut_partition'] / 'client_assignments.npy')
            require(hashlib.sha256(assign.tobytes()).hexdigest() == audit[ds]['partitions'][str(k)]['assignments'], name + ': partição')
            require(m['origins_sha256'] == {split: audit[ds]['fingerprints']['origins_' + split]
                                          for split in ['train', 'validation', 'test']}, name + ': divisões temporais')
            require(m['epochs_completed'] == len(hist) and hist.epoch.tolist() == list(range(1, len(hist)+1)), name + ': histórico')
            require(np.isfinite(hist.validation_MAE).all() and np.allclose(hist.lr, tr['learning_rate']), name + ': histórico inválido')
            best = hist.loc[hist.validation_MAE.idxmin()]
            require(int(best.epoch) == m['best_epoch'] and np.isclose(best.validation_MAE, m['validation']['MAE'], rtol=1e-9), name + ': seleção')
            ck = torch.load(p.with_name('best_model.pt'), map_location='cpu', weights_only=True)
            require(ck['best_epoch'] == m['best_epoch'] and np.isclose(ck['validation_MAE'], m['validation']['MAE'], rtol=1e-9), name + ': checkpoint selecionado')
            require(all(torch.isfinite(v).all().item() for v in ck['model_state_dict'].values()), name + ': parâmetros não finitos')
            ref, stale = float('inf'), 0
            for idx, val in enumerate(hist.validation_MAE):
                if val < ref - tr['early_stopping_min_delta']:
                    ref, stale = val, 0
                else:
                    stale += 1
                require(stale < tr['patience'] or idx == len(hist)-1, name + ': treino após parada')
            require(len(hist) <= tr['max_epochs'] and (len(hist) == tr['max_epochs'] or stale >= tr['patience']), name + ': encerramento')
            tag = {'sunt':'sunt', 'metr-la':'metrla', 'pems-bay':'pemsbay'}[ds]
            central_row = principal[(principal.celula == tag+'_cent') & (principal.semente == seed)].iloc[0]
            central = json.loads((ROOT / central_row.fonte).read_text())
            for key in ['nodes', 'input_steps', 'horizon', 'samples']:
                require(m.get(key) == central.get(key), name + ': comparabilidade ' + key)
            expected_features = central.get('features', ['speed'] if ds != 'sunt' else None)
            expected_target = central.get('target', 'speed' if ds != 'sunt' else None)
            require(m['target'] == expected_target and m['features'] == expected_features, name + ': atributos/alvo')
            for key in ['mean', 'std']:
                require(np.allclose(m['scaler'][key], central['scaler'][key], rtol=1e-7, atol=1e-8), name + ': padronização')
            for split in ['validation', 'test']:
                v = m[split]; pn = v['per_node']; count = np.asarray(pn['count']); error = np.asarray(pn['abs_error'])
                require(len(count) == m['nodes'] and np.isfinite(error).all() and (error >= 0).all()
                        and (count >= 0).all() and (count == np.floor(count)).all(), name + ': somas por nó')
                require(count.sum() == v['observations'] and np.isclose(error.sum()/count.sum(), v['MAE'], rtol=1e-6), name + ': MAE por nó')
                require(np.isclose(v['RMSE']**2, v['MSE'], rtol=1e-9), name + ': RMSE')
            require(m['test_evaluated'] and all(np.isfinite(m['test'][q]) for q in ['MAE','RMSE','R2']), name + ': teste')
            rel = str(p.relative_to(ROOT))
            rows.append(dict(celula=name,dataset=ds,k=k,semente=seed,MAE=m['test']['MAE'],RMSE=m['test']['RMSE'],
                             melhor_epoca=m['best_epoch'],epocas=m['epochs_completed'],gpu=m['gpu'],fonte=rel,sha256=digest(p),
                             checkpoint_sha256=digest(p.with_name('best_model.pt'))))
            sources[rel] = digest(p)
        require(sorted(seeds) == SEEDS, name + ': sementes')
    runs = pd.DataFrame(rows)
    contrasts = []
    for cell in cells:
        ds,k,name=cell['dataset'],cell['k'],cell['name']; tag={'sunt':'sunt','metr-la':'metrla','pems-bay':'pemsbay'}[ds]
        a=runs[runs.celula==name].set_index('semente').loc[SEEDS].MAE.to_numpy()
        for comparator,cid in [('Centralizado',tag+'_cent'),('FedAvg',tag+f'_fedavg_k{k}')]:
            b=principal[principal.celula==cid].set_index('semente').loc[SEEDS].test_MAE.to_numpy()
            d=a-b; mean=float(d.mean()); half=float(stats.t.ppf(.975,4)*d.std(ddof=1)/np.sqrt(5))
            contrasts.append(dict(dataset=ds,k=k,comparador=comparator,diferenca=mean,IC95_inf=mean-half,IC95_sup=mean+half,
                                  p_t=float(stats.ttest_rel(a,b).pvalue),p_wilcoxon=float(stats.wilcoxon(d,method='exact').pvalue),
                                  **{f'dif_semente_{s}':float(v) for s,v in zip(SEEDS,d)}))
    ct=pd.DataFrame(contrasts); order=np.argsort(ct.p_t.to_numpy()); adjusted=np.minimum(1,np.maximum.accumulate(ct.p_t.to_numpy()[order]*np.arange(14,0,-1)))
    ct['p_holm']=0.;ct.loc[order,'p_holm']=adjusted
    out=dest/'dados/controles_arestas_internas';out.mkdir()
    runs.to_csv(out/'resultados_por_semente.csv',index=False,float_format='%.17g')
    ct.to_csv(out/'contrastes.csv',index=False,float_format='%.17g')
    summary=runs.groupby(['dataset','k'],sort=False).agg(MAE=('MAE','mean'),dp=('MAE','std'),RMSE=('RMSE','mean'),RMSE_dp=('RMSE','std')).reset_index()
    summary.to_csv(out/'resumo.csv',index=False,float_format='%.17g')
    def fmt(v,n=4):return f'{v:.{n}f}'.replace('.',',')
    names={'sunt':'SUNT','metr-la':'METR-LA','pems-bay':'PEMS-BAY'}
    lines=[r'\begin{tabular}{lcrr}',r'\toprule',r'Conjunto & $k$ & MAE & RMSE \\',r'\midrule']
    for r in summary.itertuples():lines.append(f'{names[r.dataset]} & {r.k} & ${fmt(r.MAE,3)} \\pm {fmt(r.dp,3)}$ & ${fmt(r.RMSE,3)} \\pm {fmt(r.RMSE_dp,3)}$ '+r'\\')
    lines += [r'\bottomrule',r'\end{tabular}']
    (dest/'tabelas/controles_arestas_internas.tex').write_text('\n'.join(lines)+'\n')
    lines=[r'\begin{tabular}{lclrrr}',r'\toprule',r'Conjunto & $k$ & Comparador & Diferença & IC95\% & $p$ (Holm) \\',r'\midrule']
    for r in ct.itertuples():lines.append(f'{names[r.dataset]} & {r.k} & {r.comparador} & {fmt(r.diferenca)} & [{fmt(r.IC95_inf)}; {fmt(r.IC95_sup)}] & {fmt(r.p_holm)} '+r'\\')
    lines += [r'\bottomrule',r'\end{tabular}']
    (dest/'tabelas/contrastes_arestas_internas.tex').write_text('\n'.join(lines)+'\n')
    # Reutiliza os contrastes auditados, sem alterar a família de 14 testes.
    # A tabela do corpo usa FedAvg - controle; a fonte usa controle - FedAvg.
    table = dest / 'tabelas/sunt_controles_grafo.tex'
    text = table.read_text()
    additions = []
    for comparator, key, label, anchor in [
        ('Centralizado', 'CutSix', 'Cortes $k=6$ $-$ completo', 'Cortes $k=20$'),
        ('FedAvg', 'GapSix', 'FedAvg $k=6$ $-$ cortes', 'FedAvg $k=20$')]:
        selected = ct[(ct.dataset == 'sunt') & (ct.k == 6) & (ct.comparador == comparator)]
        require(len(selected) == 1, 'contraste SUNT k=6 ausente ou duplicado')
        row = selected.iloc[0]
        mean, low, high = row.diferenca, row.IC95_inf, row.IC95_sup
        if comparator == 'FedAvg':
            mean, low, high = -mean, -high, -low
        num = lambda value: r'\num{' + f'{value:.3f}' + '}'
        line = label + ' & ' + num(mean) + ' & [' + num(low) + '; ' + num(high) + r']\\'
        require(text.count(anchor) == 1, 'posição da linha do controle não é única')
        text = text.replace(anchor, line + '\n' + anchor)
        for suffix, value in [('Diff', mean), ('CILow', low), ('CIHigh', high)]:
            additions.append(r'\newcommand{\TrafficControl' + key + suffix + '}{' + f'{value:.3f}' + '}')
    # Só reorganiza as sete linhas já formatadas; mantém os números e sua ordem.
    values = [line.split(' & ', 1)[1] for line in text.splitlines()
              if ' & ' in line and r'\num{' in line]
    require(len(values) == 7, 'número de linhas da tabela de controles')
    labels = [
        r'Centralizado sem arestas entre nós distintos $-$ centralizado com grafo completo',
        r'Centralizado com arestas internas $-$ centralizado com grafo completo',
        r'FedAvg $-$ centralizado com arestas internas',
    ]
    lines = ['% Gerado dos contrastes auditados; não editar.',
             r'\begin{tabular}{@{}p{0.49\linewidth}crr@{}}\toprule',
             r'Comparação & $k$ & $\Delta$MAE & IC95\%\\\midrule',
             labels[0] + r' & --- & ' + values[0], r'\midrule']
    for start, label in [(1, labels[1]), (4, labels[2])]:
        for offset, k in enumerate([6, 20, 166]):
            heading = r'\multirow{3}{=}{' + label + '}' if offset == 0 else ''
            lines.append(heading + f' & {k} & ' + values[start + offset])
        if start == 1:
            lines.append(r'\midrule')
    lines.append(r'\bottomrule\end{tabular}')
    table.write_text('\n'.join(lines) + '\n')
    with (dest / 'tabelas/macros_analises_sunt.tex').open('a') as handle:
        handle.write('\n'.join(additions) + '\n')
    for f in ['PROTOCOLO.md','cells.json']:shutil.copy2(CFG/f,out/f)
    manifest=dict(registros=35,celulas=7,contrastes=14,fontes=sources,hashes_remotos=remote,
                  sha256_script=digest(__file__),sha256_protocolo=digest(CFG/'PROTOCOLO.md'),
                  verificacoes='Hashes remotos; preflight; código; configurações; sementes; dados e partições; históricos; seleção; parada; checkpoint; somas por nó.',
                  metadados_legados='Nos centralizados antigos das rodovias, atributos e alvo ausentes usam o padrão speed do carregador e da configuração das rodovias; escaladores escalares versus vetores float32 são comparados com rtol=1e-7 e atol=1e-8.',
                  limite='Não houve reavaliação dos checkpoints; a auditoria confere os registros gravados, sem demonstrar reprodução bit a bit em GPU.')
    (out/'auditoria.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n')
    return manifest
