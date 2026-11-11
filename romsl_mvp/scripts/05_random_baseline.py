"""Post-hoc random frozen encoder diagnostic on the existing E01 split."""
import argparse
import json
import os
from pathlib import Path
import sys

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

import torch
from torch import nn
from romsl.experiment import (
    attribute_scores, classification_scores, extract_features, fit_probes,
    reserve_test, save_checkpoint, seed_all, verify_lock, write_json,
)
from romsl.model import RadarEncoder
from romsl.pipeline import digest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--job', type=Path, required=True)
    args = parser.parse_args()
    job = args.job
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(.12)
    device = torch.device('cuda:0')
    lock_sha = verify_lock(job)
    config = json.loads((PROJECT / 'configs/mvp.yaml').read_text())
    run = job / 'runs/R'
    run.mkdir(parents=True, exist_ok=False)
    seed_all()
    encoder = RadarEncoder()
    save_checkpoint(run / 'best.pt', {
        'encoder': encoder.state_dict(), 'epoch': 0, 'seed': 42,
        'origin': 'random initialization with zero encoder updates',
        'experiment_lock_sha': lock_sha,
    })
    encoder_sha = digest(run / 'best.pt')
    fit_probes(job, 'R', device, config, {'probe_epochs': 50})
    assert digest(run / 'best.pt') == encoder_sha, 'Encoder changed during frozen probes'
    lock = {'role': 'posthoc diagnostic on already-used E01 test subjects',
            'experiment_lock_sha': lock_sha,
            'models': {filename: digest(run / filename) for filename in
                       ('best.pt', 'probe_classification_best.pt', 'probe_motion_best.pt')}}
    reserve_test(job, lock)
    data = extract_features(job, 'R', 'test', device)
    heads = {}
    for task, dimensions in [('classification', 8), ('motion', 5)]:
        state = torch.load(run / f'probe_{task}_best.pt', map_location='cpu', weights_only=True)
        assert state['encoder_sha'] == encoder_sha
        head = nn.Linear(128, dimensions).to(device)
        head.load_state_dict(state['head'])
        heads[task] = head.eval()
    results = {
        'role': lock['role'], 'seed': 42, 'encoder_updates': 0,
        'encoder_unchanged': digest(run / 'best.pt') == encoder_sha,
        'classification': classification_scores(heads['classification'], data, config['data']['probe_actions'], device),
        'motion': attribute_scores(heads['motion'], data, device), 'test_lock': lock,
    }
    write_json(job / 'random_results.json', results)
    print('RANDOM_COMPLETE', results['classification']['macro_f1'], results['motion']['mean_mae'], flush=True)


if __name__ == '__main__':
    main()
