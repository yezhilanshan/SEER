"""Fixed language prior, sensor/duration matched controls and real order pairs."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import RecordingWindowSampler, collate_windows
from .diagnostic_models import make_language_prior
from .diagnostics import match_controls
from .diagnostic_runner import dump, init, means
from .metrics import score_caption
from .runner import build_model, datasets, restore_caption_model


def dummy(captions, device='cuda'):
    n = len(captions)
    return (torch.zeros(n, 1, 1, 1, 1, device=device), torch.zeros(n, 1, 1, device=device),
            torch.ones(n, dtype=torch.long, device=device))


class CaptionOnly:
    def __init__(self, dataset):
        self.windows = dataset.windows
        self.phrases, self.overrides = dataset.phrases, dataset.overrides

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index):
        from .data import caption_from_actions
        _, _, _, key, actions = self.windows[index]
        return self.overrides.get(key, caption_from_actions(actions, self.phrases))


def prior(args):
    root, settings = init(args)
    sets, phrases = datasets(settings, ('train', 'validation', 'test'))
    texts = {key: CaptionOnly(dataset) for key, dataset in sets.items()}
    model = make_language_prior(build_model(settings, 'align', 14)).cuda()
    assert [name for name, value in model.named_parameters() if value.requires_grad] == ['encoder.prefix']
    optimizer = torch.optim.AdamW([model.encoder.prefix], lr=.001, weight_decay=.01)
    loaders = {key: DataLoader(texts[key], batch_size=2,
                              sampler=RecordingWindowSampler(texts[key]) if key == 'train' else None)
               for key in ('train', 'validation')}
    history, best = [], float('inf')
    for epoch in range(1, 4):
        losses = {}
        for split, loader in loaders.items():
            model.train(split == 'train')
            total, weight = 0., 0
            optimizer.zero_grad(set_to_none=True)
            for i, captions in enumerate(loader):
                with torch.set_grad_enabled(split == 'train'):
                    loss = model(*dummy(captions), list(captions)).loss
                    if not torch.isfinite(loss):
                        raise FloatingPointError('non-finite prefix loss')
                    if split == 'train':
                        size = min(4, len(loader) - (i // 4) * 4)
                        (loss / size).backward()
                        if (i + 1) % 4 == 0 or i + 1 == len(loader):
                            torch.nn.utils.clip_grad_norm_([model.encoder.prefix], 1., error_if_nonfinite=True)
                            optimizer.step()
                            optimizer.zero_grad(set_to_none=True)
                # Same answer-token weighting as the original runner.
                count = sum(len(model.answer_ids(caption)) for caption in captions)
                total += float(loss.detach()) * count
                weight += count
                if split == 'train' and (i + 1) % 100 == 0:
                    print(json.dumps(dict(stage='prior', epoch=epoch, batch=i+1, loss=total/weight)), flush=True)
            losses[split] = total / weight
        history.append(dict(epoch=epoch, **losses))
        if losses['validation'] < best:
            best = losses['validation']
            torch.save(dict(prefix=model.encoder.prefix.detach().cpu(), epoch=epoch,
                            training_subjects=['subject_02','subject_03','subject_04'],
                            selection_subjects=['subject_05'], llm=settings['llm'], prompt=model.prompt,
                            trainable_parameters=model.encoder.prefix.numel()), root / 'language-prior-best.pt')
        dump(root / 'language-prior-training.json', dict(history=history, best_loss=best,
                                                        frozen_llm=True, actual_radar_input=False))
        print(json.dumps(history[-1]), flush=True)
    saved = torch.load(root / 'language-prior-best.pt', map_location='cpu', weights_only=True)
    model.encoder.prefix.data.copy_(saved['prefix'].to('cuda'))
    model.eval()
    generated = model.generate(*dummy(['']), max_new_tokens=96)[0]
    result = dict(selected_epoch=saved['epoch'], prefix_tokens=8, trainable_parameters=8*896,
                  generated=generated, same_prompt_and_greedy_decode=True, actual_radar_input=False)
    for split in ('validation', 'test'):
        rows = [dict(sample_id=window[3], **score_caption(generated, window[4], phrases)) for window in sets[split].windows]
        result[split] = dict(windows=len(rows), metrics=means(rows), results=rows)
    dump(root / 'language-prior.json', result)


def real_order_pairs(rows):
    # Synchronized sensor views are one physical window. Keep one deterministic view.
    physical, unique = set(), []
    for i, row in enumerate(rows):
        # Fixed-width synchronized crops may differ by one terminal frame between
        # sensors. Their stop index must not make the same start a second event.
        key = (row['subject'], row['session'], row['start'])
        if key not in physical:
            physical.add(key)
            unique.append(i)
    pairs = []
    for offset, i in enumerate(unique):
        a = rows[i]
        for j in unique[offset + 1:]:
            b = rows[j]
            if (a['sensor'] != b['sensor'] or a['actions'] == b['actions']
                    or Counter(a['actions']) != Counter(b['actions'])
                    or len(set(a['actions'])) < 2):
                continue
            if a['recording'] == b['recording'] and a['start'] < b['stop'] and b['start'] < a['stop']:
                continue
            pairs.append((i,j))
    return pairs, len(unique)


@torch.no_grad()
def perturbations(args):
    root, settings = init(args)
    sets, phrases = datasets(settings, ('validation', 'test'))
    model, _ = restore_caption_model(Path(args.pilot) / 'align/best.pt', 'cuda')
    report = {}
    for split, dataset in sets.items():
        metadata = json.loads((root / f'caption-windows-{split}.json').read_text())
        selected = np.linspace(0, len(dataset)-1, min(64,len(dataset)), dtype=int).tolist()
        matched = {i:match_controls(metadata, i, metadata[i]['period']) for i in selected}
        pairs, physical = real_order_pairs(metadata)
        chosen_pairs = [pairs[i] for i in np.linspace(0,len(pairs)-1,min(32,len(pairs)),dtype=int)] if pairs else []
        needed = sorted(set(selected) | {i for i in matched.values() if i is not None}
                        | {i for pair in chosen_pairs for i in pair})
        # Sorted access preserves one-recording cache locality; only selected crops retained.
        samples = {i:dataset[i] for i in needed}
        generation_cache = {}
        def evaluate_sample(index, target=None, permutation=None):
            sample = samples[index]
            rd, md = sample['rd'], sample['md']
            if permutation is not None:
                rd, md = rd[permutation], md[permutation]
            batch = collate_windows([dict(sample, rd=rd, md=md)])
            inputs = (batch['rd'].cuda(), batch['md'].cuda(), batch['lengths'].cuda())
            loss = float(model(*inputs, [sample['caption'] if target is None else target]).loss)
            if permutation is None and index in generation_cache:
                text = generation_cache[index]
            else:
                text = model.generate(*inputs, max_new_tokens=96)[0]
                if permutation is None:
                    generation_cache[index] = text
            return loss, text
        rows = []
        for offset, i in enumerate(selected):
            sample = samples[i]
            loss, text = evaluate_sample(i)
            permutation = torch.from_numpy(np.random.default_rng(17+i).permutation(len(sample['rd'])))
            permuted_loss, permuted_text = evaluate_sample(i, permutation=permutation)
            row = dict(sample_id=sample['sample_id'], reference=sample['caption'], actions=sample['actions'],
                       matched_loss=loss, matched_generation=text, matched_score=score_caption(text,sample['actions'],phrases),
                       shuffled_loss=permuted_loss, shuffled_generation=permuted_text,
                       shuffled_score=score_caption(permuted_text,sample['actions'],phrases))
            wrong = matched[i]
            if wrong is not None:
                wrong_loss, wrong_text = evaluate_sample(wrong, target=sample['caption'])
                row.update(wrong_sample_id=samples[wrong]['sample_id'], wrong_reference=samples[wrong]['caption'],
                           matched_sensor=metadata[i]['sensor'], wrong_sensor=metadata[wrong]['sensor'],
                           matched_duration=metadata[i]['duration'], wrong_duration=metadata[wrong]['duration'],
                           wrong_loss=wrong_loss, wrong_generation=wrong_text,
                           wrong_score_against_original=score_caption(wrong_text,sample['actions'],phrases),
                           wrong_score_against_own=score_caption(wrong_text,samples[wrong]['actions'],phrases))
            rows.append(row)
            if (offset+1) % 16 == 0:
                print(json.dumps(dict(stage='perturbation',split=split,windows=offset+1)),flush=True)
        wrong_rows = [row for row in rows if 'wrong_loss' in row]
        summary = dict(windows=len(rows), mismatch_pairs=len(wrong_rows),
                       matched_metrics=means([row['matched_score'] for row in rows]),
                       shuffled_metrics=means([row['shuffled_score'] for row in rows]),
                       matched_loss=float(np.mean([row['matched_loss'] for row in rows])),
                       shuffled_loss=float(np.mean([row['shuffled_loss'] for row in rows])),
                       shuffled_generation_changed=float(np.mean([row['matched_generation']!=row['shuffled_generation'] for row in rows])))
        if wrong_rows:
            summary.update(wrong_loss=float(np.mean([row['wrong_loss'] for row in wrong_rows])),
                           mismatch_matched_loss=float(np.mean([row['matched_loss'] for row in wrong_rows])),
                           mismatch_matched_better=float(np.mean([row['wrong_loss']>row['matched_loss'] for row in wrong_rows])),
                           wrong_against_original_metrics=means([row['wrong_score_against_original'] for row in wrong_rows]),
                           wrong_against_own_metrics=means([row['wrong_score_against_own'] for row in wrong_rows]))
        pair_results = []
        for a,b in chosen_pairs:
            _, first = evaluate_sample(a)
            _, second = evaluate_sample(b)
            first_score = score_caption(first, samples[a]['actions'], phrases)
            second_score = score_caption(second, samples[b]['actions'], phrases)
            pair_results.append(dict(first_id=samples[a]['sample_id'],second_id=samples[b]['sample_id'],
                                     first_reference=samples[a]['caption'],second_reference=samples[b]['caption'],
                                     first_generation=first,second_generation=second,
                                     both_sequence_exact=first_score['sequence_exact']*second_score['sequence_exact'],
                                     first_score=first_score,second_score=second_score))
        report[split] = dict(summary=summary,results=rows,
                             real_order=dict(physical_windows=physical,eligible_nonoverlapping_pairs=len(pairs),
                                             evaluated_pairs=len(pair_results),
                                             both_sequence_exact=float(np.mean([x['both_sequence_exact'] for x in pair_results])) if pair_results else None,
                                             caution='Pairs share windows and are not independent; same labeled multiset, different real order, no synthetic labels.',
                                             results=pair_results))
        dump(root / 'controlled-perturbations.json', report)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['prior','perturbations'])
    parser.add_argument('--pilot',required=True)
    parser.add_argument('--output',required=True)
    args = parser.parse_args()
    globals()[args.stage](args)


if __name__ == '__main__':
    main()
