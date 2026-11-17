"""Validation-only paired Adam-state intervention in an isolated snapshot."""
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
import torch
import data_loader as dl
from experiments import train


def save(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


def digest(tensor):
    return hashlib.sha256(tensor.cpu().numpy().tobytes()).hexdigest()


def model_hash(model):
    h = hashlib.sha256()
    for name, value in model.state_dict().items():
        h.update(name.encode()); h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def reset_adam(optimizer):
    # No scheduler, lr changes, parameter replacement or RNG consumption.
    assert type(optimizer) is torch.optim.Adam
    optimizer.state.clear()


def selfcheck():
    torch.manual_seed(42)
    a = torch.nn.Linear(3, 1)
    b = copy.deepcopy(a)
    oa = torch.optim.Adam(a.parameters(), lr=.001)
    x = torch.arange(12, dtype=torch.float32).reshape(4, 3) / 12
    y = x.sum(1, keepdim=True)
    for epoch in range(3):
        reset_adam(oa)
        ob = torch.optim.Adam(b.parameters(), lr=.001)
        for _ in range(4):
            for m, o in ((a, oa), (b, ob)):
                o.zero_grad(); (m(x)-y).abs().mean().backward(); o.step()
        assert all(torch.equal(p, q) for p, q in zip(a.parameters(), b.parameters()))
        for p, q in zip(a.parameters(), b.parameters()):
            assert all(torch.equal(oa.state[p][k], ob.state[q][k]) for k in oa.state[p])
    return {'clear_state_equals_new_adam': True, 'epochs': 3, 'steps_per_epoch': 4}


def main():
    global ROOT
    import argparse
    from fedgnn_halo.reproduce import REPRO, verify, verify_prepared
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--execute', action='store_true')
    p.add_argument('--selfcheck', action='store_true')
    p.add_argument('--data-dir', default='data')
    p.add_argument('--output', default='results/reproduced/adam-diagnostic')
    args = p.parse_args()
    verify()
    if not args.execute and not args.selfcheck:
        print('Diagnóstico pareado: 3 sementes, 2 condições, 100 épocas, somente validação. Use --execute para treinar em CUDA.')
        return
    if args.execute:
        if Path(args.data_dir).resolve() != dl.DATA_DIR:
            raise ValueError('Defina FEDGNN_DATA_DIR antes de iniciar o Python, igual a --data-dir')
        verify_prepared(args.data_dir, 'sunt')
    ROOT = Path(args.output).resolve()
    if ROOT.exists():
        raise ValueError('A saída já existe; use outro diretório')
    ROOT.mkdir(parents=True)
    torch.set_num_threads(4); torch.set_num_interop_threads(2)
    check = selfcheck()
    save(ROOT/'selfcheck.json', check)
    if args.selfcheck:
        print(json.dumps(check), flush=True); return
    assert torch.cuda.is_available()
    original_dataset = train.WindowDataset
    class ValidationOnly(original_dataset):
        def __init__(self, data, split, *args, **kwargs):
            assert split in ('train', 'validation'), 'Test access prohibited'
            super().__init__(data, split, *args, **kwargs)
    train.WindowDataset = ValidationOnly
    cfg = train.load_config(REPRO/'configs/adam_diagnostic.yaml')
    assert cfg['windows']['sunt']['features'] == ['n_boardings', 'n_alightings']
    assert cfg['training']['max_epochs'] == 100
    assert cfg['training']['evaluate_test'] is False
    assert not cfg['training'].get('lr_decay_epochs')
    original_epoch = train.train_epoch
    rows = []; paired_audits = {}
    for seed in [7, 42, 101]:
        for arm in ['preservado', 'reiniciado']:
            run_cfg = copy.deepcopy(cfg); run_cfg['output'] = str(ROOT/'results'/arm)
            audit = []; audit_path = ROOT/f'audit_{arm}_s{seed}.json'
            def epoch_fn(model, loader, optimizer, device, clip, batches_seen=0, target_scale=None):
                epoch = len(audit) + 1
                initial = model_hash(model) if epoch == 1 else None
                steps_before = [int(s['step'].item()) for s in optimizer.state.values() if 'step' in s]
                if epoch > 1:
                    assert steps_before and min(steps_before) > 0
                if arm == 'reiniciado':
                    weights = model_hash(model); rng = torch.get_rng_state().clone()
                    reset_adam(optimizer)
                    assert model_hash(model) == weights and torch.equal(rng, torch.get_rng_state())
                    assert len(optimizer.state) == 0
                record = {'epoch': epoch, 'initial_model': initial,
                          'loader_rng': digest(loader.generator.get_state()),
                          'steps_before': sorted(set(steps_before)),
                          'states_at_epoch_start': len(optimizer.state)}
                audit.append(record); save(audit_path, audit)
                save(ROOT/'status.json', {'state':'running','arm':arm,'seed':seed,'epoch':epoch,
                                         'completed':len(rows),'total':6,'updated':time.time()})
                return original_epoch(model, loader, optimizer, device, clip, batches_seen, target_scale)
            train.train_epoch = epoch_fn
            print(f'START {arm} seed={seed}', flush=True)
            result = train.train_centralized('sunt', seed, run_cfg, 'cuda', resume=False)
            assert result['test_evaluated'] is False and 'test' not in result
            assert result['epochs_completed'] == 100 and len(audit) == 100
            assert all(x['states_at_epoch_start'] == 0 for x in audit) if arm == 'reiniciado' else all(x['states_at_epoch_start'] > 0 for x in audit[1:])
            paired_audits[arm] = audit
            rows.append({'arm':arm,'seed':seed,'epochs':100,'best_epoch':result['best_epoch'],
                         'MAE':result['validation']['MAE'],'RMSE':result['validation']['RMSE'],
                         'seconds':result['training_seconds'],'parameter_count':result['parameter_count']})
            save(ROOT/'summary.json',rows)
            if arm == 'reiniciado':
                other = paired_audits['preservado']
                assert other[0]['initial_model'] == audit[0]['initial_model']
                assert [a['loader_rng'] for a in other] == [a['loader_rng'] for a in audit]
            torch.cuda.empty_cache()
    paired=[]
    for seed in [7,42,101]:
        a,b=[next(r for r in rows if r['seed']==seed and r['arm']==arm) for arm in ['preservado','reiniciado']]
        paired.append({'seed':seed,'MAE_reset_minus_preserved':b['MAE']-a['MAE'],
                       'RMSE_reset_minus_preserved':b['RMSE']-a['RMSE']})
    save(ROOT/'paired_validation.json',paired)
    save(ROOT/'status.json',{'state':'complete','completed':6,'total':6,'test_evaluated':False,'updated':time.time()})

if __name__ == '__main__':
    main()
