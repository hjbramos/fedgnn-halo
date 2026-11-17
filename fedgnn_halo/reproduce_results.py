"""Compara novas execuções com a referência sem misturar as duas evidências."""
from pathlib import Path
import json
import math
from .reproduce import REPRO, catalog, verify, sha


def compare_results(results, output):
    import pandas as pd
    from experiments.export_traffic import summarize, comparisons, _cells
    verify()
    output=Path(output).resolve()
    if output.exists():
        raise ValueError('A saída já existe; use um diretório novo')
    reference=pd.read_csv(REPRO/'reference/dados/resultados_por_semente.csv',float_precision='round_trip')
    recipes={x['id']:x for x in catalog()}
    rows=[]; missing=[]; differences=[]
    for old in reference.to_dict('records'):
        cid,seed=old['celula'],int(old['semente'])
        paths=list((Path(results)/cid).rglob(f'seed_{seed:03d}/metrics.json'))
        if len(paths)>1:
            raise ValueError(f'Mais de um registro para {cid}, semente {seed}')
        if not paths:
            missing.append({'cell':cid,'seed':seed});continue
        path=paths[0];m=json.loads(path.read_text());entry=recipes[cid]
        if m.get('dataset')!=entry['dataset'] or m.get('seed')!=seed:
            raise ValueError('Identidade incompatível: '+str(path))
        if not m.get('test_evaluated', 'test' in m) or 'test' not in m:
            missing.append({'cell':cid,'seed':seed,'reason':'teste ainda não avaliado'});continue
        if entry['clients'] is not None and m.get('num_clients') != entry['clients']:
            raise ValueError('Partição incompatível: '+str(path))
        iterations=m.get('epochs_completed',m.get('rounds_completed'))
        best=m.get('best_epoch',m.get('best_round'))
        if iterations is None or best is None or iterations>entry['limit']:
            raise ValueError('Orçamento/seleção incompatível: '+str(path))
        if entry['dataset']=='sunt' and entry['group'] in ('principal','referência') and iterations!=entry['limit']:
            raise ValueError('Execução SUNT incompleta: '+str(path))
        row={k:old[k] for k in ('celula','grupo','metodo','dataset','alvo','k','semente')}
        row.update(iteracoes=iterations,melhor_iteracao=best,fonte=str(path.resolve()),sha256=sha(path))
        for key in ('MAE','RMSE','R2'):
            value=float(m['test'][key])
            if not math.isfinite(value):raise ValueError('Métrica não finita: '+str(path))
            row['test_'+key]=value
            differences.append({'celula':cid,'semente':seed,'metrica':key,'referencia':old['test_'+key],
                                'nova_execucao':value,'nova_menos_referencia':value-old['test_'+key]})
        rows.append(row)
    output.mkdir(parents=True)
    if rows:
        runs=pd.DataFrame(rows)
        runs.to_csv(output/'novas_execucoes.csv',index=False,float_format='%.17g')
        summarize(runs).to_csv(output/'resumo_novas_execucoes.csv',index=False,float_format='%.17g')
        pd.DataFrame(differences).to_csv(output/'comparacao_referencia.csv',index=False,float_format='%.17g')
        # Incomplete families must never receive a smaller Holm correction.
        if not missing:
            comparisons(runs,{c.id:c for c in _cells()}).to_csv(output/'diferencas_pareadas.csv',index=False,float_format='%.17g')
    report={'records_found':len(rows),'missing':missing,'holm_recomputed':not missing,
            'scope':'Compara métricas registradas; não certifica igualdade dos ambientes nem reavalia checkpoints. As famílias originais só são recalculadas quando todos os 360 registros estão presentes.'}
    (output/'comparison-report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(f'{len(rows)} registros encontrados; {len(missing)} ausentes. Holm recalculado: {not missing}')
