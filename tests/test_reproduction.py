from pathlib import Path
from types import SimpleNamespace
import importlib
import json
import numpy as np
import pytest
import yaml
from fedgnn_halo import reproduce as r


def test_artifact_hashes():
    assert len(r.verify()['files']) > 1000


def test_partitions_preserve_nodes_and_refuse_overwrite(tmp_path):
    r.install_partitions(tmp_path)
    paths=list(tmp_path.rglob('client_assignments.npy'))
    assert len(paths)==9
    for p in paths:
        labels=np.load(p)
        assert len(labels) in (207,325,2903)
        assert len(set(labels)) in (5,6,10,20,166)
    paths[0].write_bytes(b'altered')
    with pytest.raises(ValueError,match='existente diferente'):
        r.install_partitions(tmp_path)


@pytest.mark.parametrize('entry',r.catalog(),ids=lambda e:e['id'])
def test_training_commands_parse_without_training(entry,monkeypatch,tmp_path):
    cmd=r.commands(entry,42,r.REPRO/entry['config'],tmp_path,'cpu')[0]
    module=importlib.import_module(cmd[2])
    invoked=[]
    names={'centralized':'train_centralized','fedavg':'train_fedavg','fedprox':'train_fedprox','ditto':'train_ditto','fedgnnhet':'train_fedgnnhet','fedpub':'train_fedpub'}
    monkeypatch.setattr(module,names[entry['method']],lambda *a,**kw:invoked.append((a,kw)))
    monkeypatch.setattr('sys.argv',[cmd[2]]+cmd[3:])
    module.main()
    assert len(invoked)==1
    cfg=yaml.safe_load((r.REPRO/entry['config']).read_text())
    assert cfg['seeds']==[7,42,101,123,314]
    if entry['dataset']=='sunt' and entry['group'] in ('principal','referência'):
        assert entry['limit']==(1000 if entry['clients']==166 else 600)


def test_plan_never_starts_subprocess(monkeypatch,tmp_path,capsys):
    def forbidden(*args,**kwargs):raise AssertionError('training started')
    monkeypatch.setattr(r.subprocess,'run',forbidden)
    r.train(r.catalog()[0],SimpleNamespace(seed=42,output=tmp_path,device='cpu',execute=False))
    assert 'Planejamento apenas' in capsys.readouterr().out
    assert not list(tmp_path.iterdir())


def test_partial_results_do_not_recompute_holm(tmp_path):
    from fedgnn_halo.reproduce_results import compare_results
    path=tmp_path/'runs/metrla_cent/metr-la/seed_042/metrics.json'
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({'dataset':'metr-la','seed':42,'epochs_completed':50,'best_epoch':40,
                                'test_evaluated':True,'test':{'MAE':3.8,'RMSE':7.,'R2':.5}}))
    out=tmp_path/'comparison'
    compare_results(tmp_path/'runs',out)
    report=json.loads((out/'comparison-report.json').read_text())
    assert report['records_found']==1 and len(report['missing'])==359
    assert not report['holm_recomputed']
    assert not (out/'diferencas_pareadas.csv').exists()
