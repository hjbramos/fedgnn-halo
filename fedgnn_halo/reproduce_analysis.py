"""Recalcula análises publicadas a partir de evidências arquivadas, sem treino."""
from pathlib import Path
from functools import lru_cache
import json
import shutil
import tempfile
import subprocess
import sys
from .reproduce import REPRO, verify, unpack, sha


def export_reference(output):
    import numpy as np
    import pandas as pd
    from experiments import export_traffic as main
    from experiments import export_sunt_analysis as sunt
    from experiments import export_sunt_direcao as direction
    from experiments import export_controles_internos as controls
    from experiments import convergencia_figuras as curves
    from experiments.export_panorama import write_panorama
    from experiments.export_contrast_figures import write_contrast_figures

    verify()
    output = Path(output).resolve()
    if output.exists():
        raise ValueError('A saída já existe; use um diretório novo')
    with tempfile.TemporaryDirectory(prefix='halo-evidence-') as temporary:
        root = Path(temporary).resolve()
        archived = unpack(root)
        # The distribution has no historical Git tree. The archive manifest
        # replaces Git membership; original source and numerical checks remain.
        for module in (main, sunt, direction, controls, curves):
            module.ROOT = root
        direction.BASE = root / 'results/sunt_piloto_direcao'
        controls.BASE = root / 'results/completar_controles_20261114'
        controls.CFG = root / 'config/completar_controles_20261114'
        def tracked(path):
            rel = str(Path(path).resolve().relative_to(root))
            return rel in archived and sha(path) == archived[rel]
        sunt.tracked = tracked
        curves._tracked_files = lru_cache(maxsize=1)(lambda: frozenset(archived))

        shutil.copytree(REPRO / 'reference', output)
        runs = pd.read_csv(output/'dados/resultados_por_semente.csv', float_precision='round_trip')
        for row in runs.itertuples():
            path = root / row.fonte
            if sha(path) != row.sha256:
                raise ValueError('Fonte divergente: '+row.fonte)
            metric = json.loads(path.read_text())
            for key in ('MAE','RMSE','R2'):
                if not np.isclose(metric['test'][key], getattr(row,'test_'+key), rtol=1e-12, atol=1e-12):
                    raise ValueError('Métrica divergente: '+row.fonte+' '+key)
        by_id = {c.id:c for c in main._cells()}
        summary, comparisons = main.summarize(runs), main.comparisons(runs,by_id)
        for name, frame, keys in [('resumo_por_celula',summary,['celula']),
                                  ('diferencas_pareadas',comparisons,['A','B'])]:
            expected=pd.read_csv(output/f'dados/{name}.csv',float_precision='round_trip').set_index(keys).sort_index()
            actual=frame.set_index(keys).sort_index()
            pd.testing.assert_index_equal(actual.index,expected.index)
            for column in actual.columns:
                if column not in expected:continue
                if pd.api.types.is_numeric_dtype(actual[column]):
                    np.testing.assert_allclose(actual[column].astype(float),expected[column].astype(float),rtol=1e-9,atol=1e-12,equal_nan=True,err_msg=name+':'+column)
                else:
                    pd.testing.assert_series_equal(actual[column].fillna(''),expected[column].fillna(''),check_dtype=False)
            frame.to_csv(output/f'dados/{name}.csv',index=False,float_format='%.17g')
        main.write_tables(output/'tabelas',summary,comparisons,by_id)
        main.write_macros(output/'tabelas/macros_resultados.tex',summary,comparisons,by_id)
        main.write_figures(output/'figuras',summary,comparisons)
        main.write_figures(output/'figuras',summary,comparisons,lang='en')
        write_panorama(output,summary,main.COLORS,main.MARKERS)
        for name in ('sunt_stratified_analysis','sunt_graph_controls','sunt_piloto_direcao','controles_arestas_internas'):
            shutil.rmtree(output/'dados'/name,ignore_errors=True)
        for module in (sunt,direction,curves,controls):
            print('Verificando e exportando:',module.__name__,flush=True)
            module.export(output)
        write_contrast_figures(output)
        sensitivity = root/'analises/holm_sensibilidade_20261115'
        shutil.copytree(REPRO/'holm-sensitivity',sensitivity)
        (root/'generated-traffic').symlink_to(output,target_is_directory=True)
        subprocess.run([sys.executable,str(sensitivity/'calcular.py')],check=True,capture_output=True,text=True)
        if (sensitivity/'macros.tex').read_bytes() != (REPRO/'holm-sensitivity/macros.tex').read_bytes():
            raise ValueError('Sensibilidade de Holm divergente')
        shutil.copytree(sensitivity,output/'holm-sensitivity')
        tables=[]
        for path in sorted((output/'tabelas').glob('*.tex')):
            reference=REPRO/'reference/tabelas'/path.name
            if path.read_bytes()!=reference.read_bytes():
                raise ValueError('Tabela/macros diferentes da referência: '+path.name)
            tables.append(path.name)
        provenance=output/'provenance.json'
        if provenance.exists():provenance.rename(output/'reference-provenance.json')
        report={'status':'verified','training_started':False,'source_package':'1.29.0',
                'records_checked':len(runs),'cells':len(summary),'contrasts':len(comparisons),
                'tables_checked':tables,
                'holm_sensitivity_verified':True,
                'archive_sha256':sha(REPRO/'analysis-sources.tar.gz'),
                'scope':'Estatísticas, tabelas e figuras recalculadas dos registros originais. Descritores estruturais e registros históricos preservados do pacote de referência; dados brutos e preparação têm verificação independente pelo comando verify-data. Não constitui nova execução dos treinamentos.'}
        (output/'reproduction-report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
        print(json.dumps(report,ensure_ascii=False,indent=2))
