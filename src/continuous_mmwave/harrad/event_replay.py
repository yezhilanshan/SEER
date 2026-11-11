"""Read whole recordings, reset only at official instance starts, audit readouts."""
import argparse
import json
from pathlib import Path
import time
import traceback

import numpy as np
import torch

from .event_semantic import build_production
from .model import SegmentClassifier
from .preprocess import sha256, write_json
from .streaming import FrozenRDStream
from .training import classification_metrics, seed_everything
from .verification import verify_predictions


class OracleStartReplay:
    """Strictly sequential single-recording state machine; no class enters model.

    A requested odd-frame readout advances a temporary state with one valid frame.
    It never flushes/changes the live stream, whose pair buffer persists until a
    real second frame or the next official start. Stops only schedule readouts.
    """

    def __init__(self, model, instances):
        self.model = model
        ordered = sorted(instances, key=lambda row: row['start'])
        previous = 0
        self.starts = {}
        self.requests = {}
        for row in ordered:
            start, stop = row['start'], row['stop']
            if start < previous or stop <= start:
                raise ValueError('invalid or overlapping instances')
            previous = stop
            self.starts[start] = row
            for slot, size in [('end', stop - start), ('u3', 6), ('u6', 12), ('u12', 24)]:
                if size <= stop - start:
                    self.requests.setdefault(start + size, []).append((slot, row))
        self.position = self.resets = self.unlabelled_frames = self.tail_readouts = 0
        self.outputs = 0
        self.state = self.observed = self.pending = self.current = None

    @torch.no_grad()
    def push(self, cells, first):
        if first != self.position:
            raise ValueError('input must be sequential')
        if cells.ndim != 3 or cells.shape[1:] != (4, 64):
            raise ValueError('expected [T,4,64] cells')
        predictions = []
        for frame in cells:
            start = self.position in self.starts
            if self.state is None or start:
                self.state = self.model.former.initial_state(1)
                self.observed = torch.zeros(1, dtype=torch.long, device=cells.device)
                self.pending = None
            if start:
                self.resets += 1
                self.current = self.starts[self.position]
            if self.current is None or self.position >= self.current['stop']:
                self.unlabelled_frames += 1
            if self.pending is None:
                self.pending = frame[None]
            else:
                pair = torch.cat((self.pending, frame[None]), 0)[None]
                self.state, self.observed = self.model.advance(
                    pair, torch.ones(1, 2, dtype=torch.bool, device=cells.device), self.state, self.observed)
                self.pending = None
            self.position += 1
            if self.position not in self.requests:
                continue
            read_state, read_age = self.state, self.observed
            if self.pending is not None:
                pair = torch.cat((self.pending, torch.zeros_like(self.pending)), 0)[None]
                read_state, read_age = self.model.advance(
                    pair, torch.tensor([[True, False]], device=cells.device), self.state, self.observed)
                self.tail_readouts += 1
            logits = self.model.read(read_state)[0].float().cpu().numpy().copy()
            for slot, row in self.requests[self.position]:
                predictions.append({**row, 'slot': slot, 'logits': logits,
                                    'requested_end': self.position, 'actual_observation_end': self.position,
                                    'observed_frames': int(read_age.item())})
                self.outputs += 1
        return predictions

    def finish(self, frames):
        if self.position != frames or self.resets != len(self.starts):
            raise RuntimeError('incomplete recording or missing reset')
        if self.outputs != sum(map(len, self.requests.values())):
            raise RuntimeError('missing requested readout')
        return {'frames': frames, 'resets': self.resets, 'unlabelled_frames': self.unlabelled_frames,
                'tail_readouts': self.tail_readouts, 'outputs': self.outputs}


def _save_groups(predictions, output, prefix):
    metrics = {}
    for slot in ('end', 'u3', 'u6', 'u12'):
        rows = [row for row in predictions if row['slot'] == slot]
        arrays = {'logits': np.stack([r['logits'] for r in rows]),
                  'labels': np.array([r['label'] for r in rows]),
                  'subjects': np.array([r['subject'] for r in rows]),
                  'uids': np.array([r['uid'] for r in rows]),
                  'requested_end': np.array([r['requested_end'] for r in rows]),
                  'actual_observation_end': np.array([r['actual_observation_end'] for r in rows]),
                  'observed_frames': np.array([r['observed_frames'] for r in rows])}
        if not np.array_equal(arrays['requested_end'], arrays['actual_observation_end']):
            raise RuntimeError('readout cutoff mismatch')
        path = output / f'{prefix}-{slot}.npz'
        np.savez_compressed(path, **arrays)
        metrics[slot] = classification_metrics(arrays['logits'], arrays['labels'], arrays['subjects'])
        verify_predictions(path, metrics[slot])
    return metrics


def _compare(output, prefix, reference, reference_prefix):
    result = {}
    for slot in ('end', 'u3', 'u6', 'u12'):
        with np.load(output / f'{prefix}-{slot}.npz', allow_pickle=False) as new, \
                np.load(reference / f'{reference_prefix}-{slot}.npz', allow_pickle=False) as old:
            assert len(set(old['uids'].tolist())) == len(old['uids'])
            indexes = {uid: i for i, uid in enumerate(old['uids'])}
            if set(new['uids']) != set(old['uids']):
                raise RuntimeError('instance support mismatch')
            order = [indexes[uid] for uid in new['uids']]
            if not np.array_equal(new['labels'], old['labels'][order]):
                raise RuntimeError('label mismatch')
            scores = old['logits'][order]
            result[slot] = {'n': len(scores), 'max_abs': float(np.max(np.abs(new['logits'] - scores))),
                            'argmax_changes': int(np.count_nonzero(new['logits'].argmax(1) != scores.argmax(1)))}
    return result


def run(warmup, output):
    warmup, output = Path(warmup), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.time()
    try:
        seed_everything(42, 4)
        config = json.loads((warmup / 'config.json').read_text())
        trained = json.loads((warmup / 'training/complete.json').read_text())
        weights = warmup / 'training/best.pt'
        if sha256(weights) != trained['best_sha256']:
            raise ValueError('EventFormer best checkpoint changed')
        baseline = Path(config['baseline_output'])
        for path, expected in [(Path(config['radar_checkpoint']), config['radar_checkpoint_sha256']),
                               (baseline / 'normalizers.npz', config['normalizers_sha256']),
                               (baseline / 'canonical-manifest.json', config['baseline_manifest_sha256'])]:
            if sha256(path) != expected:
                raise ValueError(f'locked source changed: {path.name}')
        qwen_locks = json.loads((warmup / 'input-lock.json').read_text())['qwen_files']
        for name, row in qwen_locks.items():
            if sha256(Path(config['qwen_path']) / name) != row['sha256']:
                raise ValueError('Qwen source changed')
        all_records = json.loads((baseline / 'canonical-manifest.json').read_text())
        records = [r for r in all_records if r['role'] == 'validation']
        if len(records) != 4 or any(r['subject'] != 'subject7' for r in records):
            raise ValueError('unexpected validation split')
        cache = {r['file']: r for r in json.loads((warmup / 'feature-manifest.json').read_text())}
        model = build_production(config)
        saved = torch.load(weights, map_location='cpu', weights_only=False)
        model.load_learned(saved['learned']); model.requires_grad_(False).eval()
        del saved
        radar = SegmentClassifier('rd')
        state = torch.load(config['radar_checkpoint'], map_location='cpu', weights_only=False)
        radar.load_state_dict(state['model']); del state
        frontend = FrozenRDStream(radar).cuda().eval()
        with np.load(baseline / 'normalizers.npz') as norms:
            mean, scale = norms['rd_mean'], norms['rd_scale']
        predictions, isolated, coverage = [], [], []
        max_cells_gap = 0.
        write_json(output / 'input-lock.json', {'config': config, 'event_best_sha256': trained['best_sha256'],
                   'qwen_files': qwen_locks, 'test_signal_values_read': False})
        with torch.no_grad():
            for number, record in enumerate(records):
                path = Path(record['features']['rd'])
                if sha256(path) != record['artifacts']['rd.npy']['sha256']:
                    raise ValueError('source RD changed')
                prior = cache[record['file']]
                if sha256(prior['cells_path']) != prior['artifacts']['cells.npy']['sha256']:
                    raise ValueError('frozen CNN cells changed')
                rd = np.load(path, mmap_mode='r')
                cells = np.load(prior['cells_path'], mmap_mode='r')
                instances = [{**r, 'uid': f"{record['file']}:{r['start']}:{r['stop']}",
                              'subject': record['subject']} for r in record['instances']]
                engine = OracleStartReplay(model, instances)
                carried = None
                first = chunk = 0
                while first < len(rd):
                    size = (1, 63, 17, 64, 33)[chunk % 5]
                    end = min(len(rd), first + size)
                    batch = torch.from_numpy(np.clip((np.array(rd[first:end]) - mean) / scale, -8, 8))[None].cuda()
                    features = frontend.forward_chunk(batch, state=carried)
                    carried = features['state']
                    actual = features['cells'][0]
                    gap = float(np.max(np.abs(actual.cpu().numpy() - cells[first:end])))
                    max_cells_gap = max(max_cells_gap, gap)
                    predictions.extend(engine.push(actual, first))
                    first, chunk = end, chunk + 1
                stats = engine.finish(len(rd)); stats['file'] = record['file']; coverage.append(stats)
                # Batch-one isolated bridge separates streaming protocol from batch-four rounding.
                for row in instances:
                    x = torch.from_numpy(np.array(cells[row['start']:row['stop']]))[None].cuda()
                    result = model(x, torch.ones(1, x.shape[1], dtype=torch.bool, device='cuda'))
                    for slot, size in [('end', x.shape[1]), ('u3', 6), ('u6', 12), ('u12', 24)]:
                        if size > x.shape[1]:
                            continue
                        logits = result['final_logits'] if slot == 'end' else result['prefix'][int(slot[1:])]['logits']
                        isolated.append({**row, 'slot': slot, 'logits': logits[0].float().cpu().numpy(),
                                         'requested_end': row['start'] + size, 'actual_observation_end': row['start'] + size,
                                         'observed_frames': size})
                write_json(output / 'status.json', {'phase': 'replaying', 'recordings_completed': number + 1,
                           'recordings_total': 4, 'frames': sum(r['frames'] for r in coverage), 'updated_unix': time.time()})
                print(f"oracle replay {number + 1}/4: {record['file']} frames={len(rd)} resets={stats['resets']}", flush=True)
        metrics = _save_groups(predictions, output, 'stream-oracle')
        _save_groups(isolated, output, 'isolated-b1')
        protocol = _compare(output, 'stream-oracle', output, 'isolated-b1')
        saved_comparison = _compare(output, 'stream-oracle', warmup / 'training', 'validation-best')
        bridge_comparison = _compare(output, 'isolated-b1', warmup / 'training', 'validation-best')
        passed = max_cells_gap <= 1e-4 and all(r['max_abs'] <= 1e-4 and r['argmax_changes'] == 0 for r in protocol.values())
        if sum(r['frames'] for r in coverage) != 25244 or sum(r['resets'] for r in coverage) != 511:
            raise RuntimeError('incomplete validation coverage')
        if sha256(weights) != trained['best_sha256'] or sha256(config['radar_checkpoint']) != config['radar_checkpoint_sha256']:
            raise RuntimeError('source weights changed')
        report = {'phase': 'complete', 'protocol_passed': passed, 'metrics': metrics, 'coverage': coverage,
                  'cnn_cells_max_abs': max_cells_gap, 'stream_vs_isolated_b1': protocol,
                  'stream_vs_saved_b4': saved_comparison, 'isolated_b1_vs_saved_b4': bridge_comparison,
                  'saved_full_instance_macro_f1': trained['validation']['end']['macro_f1'],
                  'reproduced_saved_classes': all(r['argmax_changes'] == 0 for r in saved_comparison.values()),
                  'test_signal_values_read': False, 'new_optimizer_steps': 0, 'source_weights_unchanged': True,
                  'seconds': time.time() - started, 'completed_unix': time.time(),
                  'readout_protocol': 'event-relative pairs; pending frame reset only at official starts; odd cutoff uses non-mutating one-frame snapshot'}
        write_json(output / 'complete.json', report)
        write_json(output / 'status.json', {'phase': 'complete', 'protocol_passed': passed, 'updated_unix': time.time()})
        print(json.dumps({k: report[k] for k in ('protocol_passed', 'reproduced_saved_classes', 'seconds')}), flush=True)
    except BaseException as exc:
        failure = {'phase': 'failed', 'error': repr(exc), 'traceback': traceback.format_exc(), 'updated_unix': time.time()}
        write_json(output / 'failed.json', failure); write_json(output / 'status.json', failure)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--warmup', required=True); parser.add_argument('--output', required=True)
    args = parser.parse_args()
    run(args.warmup, args.output)
