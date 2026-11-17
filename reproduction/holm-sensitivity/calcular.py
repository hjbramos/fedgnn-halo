"""Análise externa ao PDF; não modifica o pacote de resultados."""
import csv, hashlib, json, math
from pathlib import Path
from collections import defaultdict
BASE=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
SRC=BASE/'generated-traffic/dados/diferencas_pareadas.csv'
rows=list(csv.DictReader(SRC.open()))
assert len({tuple(sorted((r['A'],r['B']))) for r in rows})==len(rows)
def holm(indices):
    ordered=sorted(indices,key=lambda i:float(rows[i]['p_t_pareado']))
    out={}; running=0.
    for rank,i in enumerate(ordered):
        p=float(rows[i]['p_t_pareado']); assert math.isfinite(p) and 0<=p<=1
        running=max(running,(len(ordered)-rank)*p)
        out[i]=min(1.,running)
    return out
families=defaultdict(list)
for i,r in enumerate(rows): families[r['familia_inferencial']].append(i)
for inds in families.values():
    for i,p in holm(inds).items(): assert math.isclose(p,float(rows[i]['p_t_holm']),rel_tol=1e-12,abs_tol=1e-14)
scopes={'tarefas_principais_62':[i for i,r in enumerate(rows) if not r['familia_inferencial'].startswith('complementar:')], 'todos_exportados_65':list(range(len(rows)))}
summary={'fonte':str(SRC.relative_to(BASE)), 'sha256':hashlib.sha256(SRC.read_bytes()).hexdigest(),'familias_publicadas_reproduzidas':True,'duplicatas':0,'cenarios':{}}
for name,inds in scopes.items():
    adjusted=holm(inds); output=[]; changes=[]
    for i in inds:
        r=rows[i]; old=float(r['p_t_holm']); new=adjusted[i]
        record={**r,'p_holm_familia_unica':new,'atende_005_original':old<.05,'atende_005_unica':new<.05}
        output.append(record)
        if (old<.05)!=(new<.05): changes.append({k:record[k] for k in ['familia','A','B','dif_MAE_media','p_t_pareado','p_t_holm','p_holm_familia_unica']})
    with (OUT/(name+'.csv')).open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(output[0])); w.writeheader(); w.writerows(output)
    summary['cenarios'][name]={'n':len(inds),'atendem_original':sum(float(rows[i]['p_t_holm'])<.05 for i in inds),'atendem_unica':sum(p<.05 for p in adjusted.values()),'mudancas':changes}
(OUT/'resumo.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(summary,ensure_ascii=False,indent=2))
# Macros derivadas do pacote vigente para a síntese no apêndice.
macro=[]
for name,tag in [('tarefas_principais_62','Principais'),('todos_exportados_65','Todos')]:
    v=summary['cenarios'][name]
    for suffix,key in [('N','n'),('Antes','atendem_original'),('Depois','atendem_unica')]:
        macro.append('\\newcommand{\\HolmSens'+tag+suffix+'}{'+str(v[key])+'}')
    halo=next(x for x in v['mudancas'] if x['A']=='metrla_halo_k20')
    macro.append('\\newcommand{\\HolmSens'+tag+'HaloP}{'+format(halo['p_holm_familia_unica'],'.4f')+'}')
(OUT/'macros.tex').write_text('% Gerado por calcular.py a partir de generated-traffic; não editar.\n'+'\n'.join(macro)+'\n')
