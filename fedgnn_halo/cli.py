"""Entrada única para exemplos e experimentos com configuração YAML."""
import argparse
import copy
import json
from pathlib import Path
import platform
import importlib.metadata
import numpy as np
import torch
import yaml
from .data import synthetic
from .partition import assignments, save_partition

METHODS = ('centralized', 'local', 'fedavg', 'fedprox', 'ditto', 'halo')


def run(options):
    options = copy.deepcopy(options)
    method = options.get('method', 'halo')
    if method not in METHODS:
        raise ValueError(f'Método deve ser um de {METHODS}')
    model = options.get('model', {'name': 'leconv', 'hidden_size': 16})
    if method == 'halo' and model.get('name', 'leconv') != 'leconv':
        raise ValueError('O halo disponível foi formulado para uma camada LEConv')
    output = Path(options.get('output', 'results/example')).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError(f'{output} não está vazio; escolha outra saída para preservar a execução')
    seed = int(options.get('seed', 7))
    torch.set_num_threads(int(options.get('threads', 1)))
    spec = dict(options.get('dataset', {}))
    name = spec.pop('name', 'synthetic')
    if name == 'synthetic':
        data = synthetic(seed=seed, **spec)
    else:
        from experiments.data import load_experiment
        data = load_experiment(name, **spec)
    output.mkdir(parents=True, exist_ok=True)
    labels = assignments(data.adj, options.get('partition', {}), seed)
    pdir = output / 'partition'
    k = save_partition(data, labels, pdir)
    training = {'learning_rate': .001, 'max_epochs': 3, 'max_rounds': 3,
                'patience': 10, 'early_stopping_min_delta': 0., 'gradient_clip_norm': 5.,
                'local_epochs': 1, 'partition_seed': seed, 'lambda_candidates': [.01, .1, 1.]}
    training.update(options.get('training', {}))
    if min(int(training[key]) for key in ('max_epochs', 'max_rounds', 'local_epochs')) < 1:
        raise ValueError('Épocas e rodadas devem ser positivas')
    cfg = {'windows': {name: {'input_steps': data.input_steps, 'horizon': data.horizon}},
           'data': {'batch_size': int(options.get('batch_size', 16)), 'num_workers': 0},
           'model': model, 'output': str(output / 'runs')}
    for section in ('training', 'federated', 'fedprox', 'ditto', 'fedgnnhet'):
        cfg[section] = dict(training)
    cfg['ditto']['local_only'] = method == 'local'
    device = options.get('device', 'cpu')
    kwargs = dict(data=data, partition_path=pdir, resume=False)
    (output / 'config.yaml').write_text(yaml.safe_dump(options, sort_keys=False))
    (output / 'resolved-config.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False))
    environment = {'python': platform.python_version(), 'platform': platform.platform(),
                   'packages': {p: importlib.metadata.version(p) for p in
                                ('fedgnn-halo', 'torch', 'torch-geometric', 'numpy', 'pandas')}}
    (output / 'environment.json').write_text(json.dumps(environment, indent=2)+'\n')
    if method == 'centralized':
        from experiments.train import train_centralized
        result = train_centralized(name, seed, cfg, device, **kwargs)
    elif method == 'fedavg':
        from experiments.federated import train_fedavg
        result = train_fedavg(name, k, seed, cfg, device, **kwargs)
    elif method == 'fedprox':
        from experiments.fedprox import train_fedprox
        result = train_fedprox(name, k, float(training.get('proximal_mu', .01)), seed, cfg, device, **kwargs)
    elif method in ('local', 'ditto'):
        from experiments.ditto import train_ditto
        lam = 0. if method == 'local' else training.get('lambda', 'per_client')
        result = train_ditto(name, k, lam, seed, cfg, device, **kwargs)
    else:
        from experiments.fedgnnhet import train_fedgnnhet
        result = train_fedgnnhet(name, k, '1hop', seed, cfg, device, **kwargs)
    (output / 'summary.json').write_text(json.dumps(result, indent=2, default=str)+'\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--method', choices=METHODS)
    parser.add_argument('--output')
    args = parser.parse_args()
    options = yaml.safe_load(args.config.read_text())
    if args.method:
        options['method'] = args.method
    if args.output:
        options['output'] = args.output
    result = run(options)
    print(json.dumps(result.get('test', result.get('validation', {})), indent=2))


if __name__ == '__main__':
    main()
