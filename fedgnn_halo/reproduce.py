"""Preparação e execução explícita do protocolo da dissertação."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
REPRO = Path(os.environ.get('FEDGNN_REPRODUCTION_DIR', ROOT / 'reproduction')).resolve()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def verify():
    manifest = json.loads((REPRO / 'MANIFEST.json').read_text())
    failures = [name for name, expected in manifest['files'].items()
                if not (REPRO / name).is_file() or sha(REPRO / name) != expected]
    if failures:
        raise ValueError('Artefatos ausentes ou alterados: '+', '.join(failures[:10]))
    return manifest


def unpack(destination):
    verify()
    with tarfile.open(REPRO / 'analysis-sources.tar.gz') as archive:
        for member in archive.getmembers():
            p = Path(member.name)
            if p.is_absolute() or '..' in p.parts or not (member.isfile() or member.isdir()):
                raise ValueError('Caminho inválido no arquivo de evidências')
        archive.extractall(destination, filter='data')
    manifest = json.loads((REPRO / 'analysis-manifest.json').read_text())
    for name, expected in manifest.items():
        if sha(Path(destination) / name) != expected:
            raise ValueError('Evidência divergente: '+name)
    return manifest


def install_partitions(data_dir):
    verify()
    src = REPRO / 'partitions/data'
    dest = Path(data_dir).resolve()
    # Existing content is never silently replaced.
    for p in src.rglob('*'):
        if p.is_file():
            target = dest / p.relative_to(src)
            if target.exists() and sha(target) != sha(p):
                raise ValueError(f'Partição existente diferente: {target}')
    for p in src.rglob('*'):
        if p.is_file():
            target = dest / p.relative_to(src)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, target)
    return dest


def verify_raw(data_dir, dataset):
    files = json.loads((REPRO / 'data-manifest.json').read_text())['files']
    for item in files:
        if item['path'].split('/')[0] != dataset:
            continue
        path = Path(data_dir) / item['path']
        if not path.is_file() or sha(path) != item['sha256']:
            raise ValueError(f'Dado bruto ausente ou diferente da referência: {path}')


def prepared_fingerprint(dataset):
    import data_loader as dl
    import numpy as np
    def digest(a):
        a = np.ascontiguousarray(a)
        return hashlib.sha256(str((a.shape, str(a.dtype))).encode()+a.tobytes()).hexdigest()
    values, mask, timestamps, ids = dl.load_features(dataset)
    adj, graph_ids, _ = dl.load_adjacency(dataset, dl.load(dataset))
    if list(ids) != list(graph_ids):
        raise ValueError('Ordem dos nós do grafo difere da série')
    return {'series_sha256': digest(values.astype(np.float32)),
            'mask_sha256': digest(mask.astype(np.bool_)),
            'adjacency_sha256': digest(adj.astype(np.float64)),
            'ids_sha256': hashlib.sha256(json.dumps(ids, separators=(',', ':')).encode()).hexdigest(),
            'timestamps_sha256': digest(timestamps.to_numpy().astype('datetime64[ns]').astype(np.int64)),
            'shape': list(values.shape), 'feature_names': dl.feature_names(dataset)}


def configure_sources():
    sources = json.loads((REPRO / "source-revisions.json").read_text())
    os.environ["FEDGNN_HF_REVISIONS"] = json.dumps({v["repo"]:v["revision"] for v in sources.values()})


def verify_prepared(data_dir, dataset):
    configure_sources()
    os.environ['FEDGNN_DATA_DIR'] = str(Path(data_dir).resolve())
    verify_raw(data_dir, dataset)
    expected = json.loads((REPRO / 'prepared-manifest.json').read_text())[dataset]
    actual = prepared_fingerprint(dataset)
    if actual != expected:
        raise ValueError(f'Dados preparados divergentes em {dataset}: '+', '.join(k for k in expected if expected[k] != actual[k]))


def catalog():
    return json.loads((REPRO / 'catalog.json').read_text())


def commands(entry, seed, config, output, device):
    import yaml
    cfg = yaml.safe_load(Path(config).read_text())
    method, block = entry['method'], cfg[entry['section']]
    module = {'centralized':'train', 'fedavg':'federated', 'fedprox':'fedprox',
              'ditto':'ditto', 'fedgnnhet':'fedgnnhet', 'fedpub':'fedpub'}[method]
    cmd = [sys.executable, '-m', 'experiments.'+module, '--dataset',entry['dataset'],
           '--seed',str(seed),'--config',str(config),'--device',device,'--output',str(output)]
    if method != 'centralized':
        cmd += ['--clients',str(entry['requested_clients'])]
    option = {'fedprox':('--mu','proximal_mu'),'ditto':('--lam','lambda'),
              'fedgnnhet':('--halo','halo'),'fedpub':('--l1','lambda_l1')}.get(method)
    if option:
        cmd += [option[0], str(block[option[1]][0])]
    budget = '--max-epochs' if method == 'centralized' else '--max-rounds'
    result = [cmd + [budget,str(entry['limit'])]]
    if entry['evaluate_after_training']:
        result.append(cmd + ['--evaluate-test-from-checkpoint'])
    return result


def train(entry, args):
    verify()
    if args.seed not in entry['seeds']:
        raise ValueError('Semente fora do protocolo')
    config = REPRO / entry['config']
    output = Path(args.output).resolve() / entry['id']
    planned = commands(entry,args.seed,config,output,args.device)
    if not args.execute:
        for cmd in planned:
            print(shlex.join(cmd))
        print('Planejamento apenas. Para treinar, repita com --execute após preparar e verificar os dados.')
        return
    import yaml
    verify_prepared(args.data_dir, entry['dataset'])
    install_partitions(args.data_dir)
    cfg = yaml.safe_load(config.read_text())
    cut = cfg.get('model',{}).get('cut_partition')
    if cut:
        cfg['model']['cut_partition'] = str(Path(args.data_dir).resolve() / Path(cut).relative_to('data'))
    output.mkdir(parents=True, exist_ok=True)
    resolved = output / f'config-seed-{args.seed}.yaml'
    if resolved.exists() and yaml.safe_load(resolved.read_text()) != cfg:
        raise ValueError('Configuração existente difere; use outra saída')
    resolved.write_text(yaml.safe_dump(cfg,allow_unicode=True,sort_keys=False))
    env = dict(os.environ, FEDGNN_DATA_DIR=str(Path(args.data_dir).resolve()),
               OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', CUBLAS_WORKSPACE_CONFIG=':4096:8')
    for cmd in commands(entry,args.seed,resolved,output,args.device):
        subprocess.run(cmd,check=True,env=env,cwd=ROOT)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='action',required=True)
    sub.add_parser('verify')
    sub.add_parser('list')
    plan=sub.add_parser('plan');plan.add_argument('--group',default=None)
    part=sub.add_parser('install-partitions');part.add_argument('--data-dir',default='data')
    for action in ['prepare-data','verify-data']:
        q=sub.add_parser(action);q.add_argument('--dataset',choices=['metr-la','pems-bay','sunt','all'],required=True);q.add_argument('--data-dir',default='data')
    q=sub.add_parser('train');q.add_argument('--experiment',required=True);q.add_argument('--seed',type=int,required=True)
    q.add_argument('--execute',action='store_true');q.add_argument('--device',default='cpu');q.add_argument('--data-dir',default='data');q.add_argument('--output',default='results/reproduced')
    q=sub.add_parser('export-reference');q.add_argument('--output',required=True)
    q=sub.add_parser('compare-results');q.add_argument('--results',default='results/reproduced');q.add_argument('--output',required=True)
    args=p.parse_args()
    if args.action=='verify':print(f"{len(verify()['files'])} artefatos verificados")
    elif args.action=='install-partitions':print(install_partitions(args.data_dir))
    elif args.action=='list':
        for item in catalog():print(item['id'],item['method'],item['clients'],item['limit'],item['group'])
    elif args.action=='plan':
        for item in catalog():
            if args.group and item['group']!=args.group:continue
            for seed in item['seeds']:
                print(f"python -m fedgnn_halo.reproduce train --experiment {item['id']} --seed {seed}")
    elif args.action=='train':
        found=[x for x in catalog() if x['id']==args.experiment]
        if not found:raise ValueError('Experimento não encontrado')
        train(found[0],args)
    elif args.action in ('prepare-data','verify-data'):
        verify();configure_sources();os.environ['FEDGNN_DATA_DIR']=str(Path(args.data_dir).resolve())
        import data_loader as dl
        for dataset in dl.DATASETS if args.dataset=='all' else [args.dataset]:
            if args.action=='prepare-data':dl.download(dataset)
            verify_prepared(args.data_dir,dataset)
            print(dataset,'dados e grafo conferidos')
        install_partitions(args.data_dir)
    elif args.action=='compare-results':
        from .reproduce_results import compare_results
        compare_results(Path(args.results),Path(args.output))
    else:
        from .reproduce_analysis import export_reference
        export_reference(Path(args.output))


if __name__=='__main__':main()
