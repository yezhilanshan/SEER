"""Run controlled readout training against immutable Stage1 GRU feature caches."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import sys

import numpy as np
import torch

from continuous_mmwave.radar_text.head_ablation import (
    arm_specs, assert_reference, evaluate_head, fit_head, load_cache, sha,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--diagnostics', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=False)
    cache_root, checkpoint = Path(args.diagnostics), Path(args.checkpoint)
    torch.set_num_threads(4)
    source = Path(__file__).resolve().parents[1]
    source_hashes = json.loads((source/'source-manifest.json').read_text())
    for name, expected in source_hashes.items():
        assert sha(source/name) == expected, name
    started = datetime.now().astimezone().isoformat()
    initial_hash = sha(checkpoint)
    completed = []

    def dump(name, value):
        path = root/name
        temporary = path.with_suffix(path.suffix+'.tmp')
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
        temporary.replace(path)

    def status(state, **extra):
        dump('status.json', dict(state=state, started=started, completed=completed, **extra))

    status('loading_verified_cache')
    try:
        arrays, raw, subjects, cache_hashes = load_cache(cache_root)
        saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
        training = saved['settings']['training']
        assert training['epochs'] == 3 and training['lr'] == .001
        assert training['weight_decay'] == .01 and training['grad_clip'] == 1.
        reference = saved['classifier']
        provenance = dict(started=started, command=sys.argv, source_sha256=source_hashes,
                          checkpoint=str(checkpoint), checkpoint_sha256=initial_hash,
                          checkpoint_epoch=saved['epoch'], original_training=training,
                          subjects=subjects, feature='stage1_gru', cache_root=str(cache_root),
                          cache_sha256=cache_hashes, seed=17, base_llm_loaded=False,
                          encoder_loaded=False, trainable_parameters=3598,
                          frames={s: len(a[0]) for s, a in arrays.items()},
                          selection='validation macro-F1, ties earliest',
                          context='same cached nonoverlap12s eval-mode chunks as prior diagnostic',
                          cache_index_sha256=sha(cache_root/'feature-index.json'),
                          labels_note='command_active_until_next_command; short support is limited')
        dump('provenance.json', provenance)
        width = reference['weight'].shape[1]
        reference_payload = dict(head=reference, mean=torch.zeros(width), std=torch.ones(width))
        baseline, baseline_predictions = evaluate_head(reference_payload, arrays, device=args.device)
        original = json.loads((cache_root/'frame-classification.json').read_text())
        head = torch.nn.Linear(width, 14).to(args.device).eval()
        head.load_state_dict(reference)
        logit_difference = {}
        with torch.no_grad():
            for split, data in arrays.items():
                actual = head(torch.from_numpy(data[0]).to(args.device)).cpu().numpy()
                logit_difference[split] = assert_reference(actual, raw[split][0], raw[split][1])
                for subset, metric in [('all_valid', baseline[split]), ('steady', baseline[split]['steady'])]:
                    assert metric['confusion_matrix'] == original[split][subset]['confusion_matrix']
        result = dict(reference=dict(name='H0@0: zero-update original checkpoint readout',
                                     predictions_identical=True, max_logit_difference=logit_difference,
                                     results=baseline), panels={})
        dump('head-ablation.json', result)
        del head
        # Train receives only train and validation. Test is inspected after each saved selection.
        fitting = {s: arrays[s] for s in ('train','validation')}
        for panel in ('original_optimizer', 'probe_replay'):
            rows = {}
            result['panels'][panel] = rows
            for spec in arm_specs(panel):
                arm = spec['arm']
                status('running', panel=panel, arm=arm)
                path = root/f'{panel}-{arm}.pt'
                row = fit_head(fitting, spec, reference, path, device=args.device)
                best = torch.load(path, map_location='cpu', weights_only=True)
                last = torch.load(path.with_name(path.stem+'-last.pt'), map_location='cpu', weights_only=True)
                row['results'], predictions = evaluate_head(best, arrays, device=args.device)
                row['last_results'], _ = evaluate_head(last, arrays, device=args.device)
                row['checkpoint'] = path.name
                row['checkpoint_sha256'] = sha(path)
                for split in ('validation','test'):
                    np.savez(root/f'{panel}-{arm}-{split}-predictions.npz',
                             labels=arrays[split][1], predictions=predictions[split],
                             short=arrays[split][2], steady=arrays[split][3])
                rows[arm] = row
                completed.append(f'{panel}/{arm}')
                dump('head-ablation.json', result)
            assert rows['H0']['history'] == rows['H1']['history'][:3], 'unequal order/init prefix'
        # Check the independent probe recipe against previously saved predictions and weights.
        replay = torch.load(root/'probe_replay-H4.pt', map_location='cpu', weights_only=True)
        old = torch.load(cache_root/'stage1_gru-probe.pt', map_location='cpu', weights_only=True)
        old_report = json.loads((cache_root/'linear-probes.json').read_text())['stage1_gru']
        replay_info = dict(selected_epoch=replay['epoch'], previous_selected_epoch=old['epoch'],
                           max_head_difference=max(float((replay['head'][k]-v).abs().max())
                                                   for k, v in old['head'].items()),
                           prediction_differences={}, mean_identical=torch.equal(replay['mean'], old['mean']),
                           std_identical=torch.equal(replay['std'], old['std']),
                           max_history_macro_f1_difference=max(abs(a['validation_macro_f1']-b['validation_macro_f1'])
                               for a,b in zip(result['panels']['probe_replay']['H4']['history'],old_report['history'])))
        for split in ('validation','test'):
            with np.load(root/f'probe_replay-H4-{split}-predictions.npz') as a, np.load(cache_root/f'stage1_gru-probe-{split}-predictions.npz') as b:
                assert np.array_equal(a['labels'],b['labels'])
                replay_info['prediction_differences'][split] = int((a['predictions']!=b['predictions']).sum())
        replay_info['predictions_identical'] = not any(replay_info['prediction_differences'].values())
        result['probe_replay_verification'] = replay_info
        dump('head-ablation.json', result)
        assert replay_info['predictions_identical'] and replay_info['mean_identical'] and replay_info['std_identical'], replay_info
        assert replay_info['selected_epoch'] == replay_info['previous_selected_epoch'], replay_info
        assert sha(checkpoint) == initial_hash, 'original checkpoint changed'
        for name, entry in cache_hashes.items():
            assert sha(cache_root/'features'/name) == entry['sha256'], name
        verification = dict(original_checkpoint_unchanged=True, feature_caches_unchanged=len(cache_hashes),
                            reference_predictions_identical=True, h0_h1_training_prefix_identical=True,
                            probe_replay=replay_info, source_files_verified=len(source_hashes),
                            no_encoder_or_llm_loaded=True)
        dump('verification.json', verification)
        status('complete', finished=datetime.now().astimezone().isoformat())
    except BaseException as error:
        status('failed', error=str(error), finished=datetime.now().astimezone().isoformat())
        raise


if __name__ == '__main__':
    main()
