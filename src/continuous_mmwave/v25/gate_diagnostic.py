"""Zero-optimizer gate ablations. Never selects a new training architecture."""
from __future__ import annotations

from collections import defaultdict
import copy
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .models import AgeFreeStateGate


def sigmoid_to_tanh_coordinate(logits):
    # atanh(sigmoid(z)) = .5 log(1+2 exp(z)); stable even at z=+1000.
    return .5 * F.softplus(logits + math.log(2.))


def _condition(value):
    if not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError('fixed maturity must be finite in [0,1]')
    return float(value)


class TransferredGate(AgeFreeStateGate):
    def __init__(self, dim, *, exact):
        super().__init__(dim)
        self.exact = bool(exact)
        self.architecture = ('v25-age-free-tanh-equivalent-diagnostic-v1' if exact else
                             'v25-age-free-tanh-warmstart-v1')

    def forward(self, previous, proposal, _legacy_maturity=None):
        if previous.shape != proposal.shape or previous.ndim != 3:
            raise ValueError('matching [B,E,D] states required')
        delta = proposal - previous
        z = self.network(torch.cat((self.norm(previous), self.norm(delta)), -1))
        alpha = (sigmoid_to_tanh_coordinate(z) if self.exact else z).tanh()
        return previous + alpha * delta, alpha


def transferred_gate(legacy, *, maturity, exact):
    maturity = _condition(maturity)
    dim = legacy.norm.normalized_shape[0]
    if legacy.network[0].in_features != 2 * dim + 1:
        raise ValueError('legacy fitted maturity column required')
    gate = TransferredGate(dim, exact=exact).to(legacy.network[0].weight)
    with torch.no_grad():
        gate.norm.load_state_dict(legacy.norm.state_dict())
        gate.network[0].weight.copy_(legacy.network[0].weight[:, :2*dim])
        gate.network[0].bias.copy_(legacy.network[0].bias + maturity*legacy.network[0].weight[:, -1])
        gate.network[2].load_state_dict(legacy.network[2].state_dict())
    gate.initialization = dict(kind='legacy-column-fold',maturity_condition=maturity,
                               exact_output_coordinate=bool(exact))
    return gate


class FixedMaturityGate(nn.Module):
    architecture = 'legacy-fixed-maturity-diagnostic-v1'

    def __init__(self, legacy, maturity):
        super().__init__()
        self.gate, self.maturity = copy.deepcopy(legacy), _condition(maturity)

    def forward(self, previous, proposal, _legacy_maturity=None):
        return self.gate(previous, proposal, self.maturity)


def fixed_maturity_gate(legacy, *, maturity):
    return FixedMaturityGate(legacy, maturity)


def tensor_summary(values):
    values = torch.as_tensor(values,dtype=torch.float64).flatten()
    finite = values[torch.isfinite(values)]
    return dict(total=len(values),nonfinite=int(len(values)-len(finite)),
                mean=float(finite.mean()) if len(finite) else None,
                median=float(finite.median()) if len(finite) else None,
                p95=float(torch.quantile(finite,.95)) if len(finite) else None,
                p99=float(torch.quantile(finite,.99)) if len(finite) else None,
                maximum=float(finite.max()) if len(finite) else None)


def warmstart_usable(fit_summary):
    negative, ratio = fit_summary['negative_alpha_fraction'], fit_summary['state_norm_ratio_p99']
    if not isinstance(negative,(int,float)) or not math.isfinite(negative):
        raise ValueError('finite fit-only warmstart measurements required')
    missing_ratio=ratio is None and fit_summary['all_finite'] is False
    if not missing_ratio and (not isinstance(ratio,(int,float)) or not math.isfinite(ratio)):
        raise ValueError('finite fit-only norm ratio required for finite candidates')
    reasons = []
    if fit_summary['all_finite'] is not True:
        reasons.append('nonfinite-fit-output')
    if negative > .5:
        reasons.append('negative-alpha-fraction-over-0.5')
    if missing_ratio:
        reasons.append('no-finite-student-state-ratio')
    elif ratio > 4.:
        reasons.append('state-norm-ratio-p99-over-4')
    return dict(passed=not reasons,reasons=reasons,criteria=dict(max_negative_alpha_fraction=.5,
                max_state_norm_ratio_p99=4.,all_finite_required=True),selection_split='fit')


def _load(record):
    from .checkpoint import file_sha
    if file_sha(record['path']) != record['sha256']:
        raise ValueError('frozen cache SHA mismatch')
    return torch.load(record['path'],map_location='cpu',weights_only=False)


def validate_inference_source(config,index):
    from .checkpoint import file_sha
    measured=file_sha(config['frontend_config'])
    if measured!=index.get('config_sha256'):
        raise ValueError('frontend inference config SHA mismatch')
    if index.get('checkpoint_sha256')!=config['source_checkpoint_sha256']:
        raise ValueError('encoder source checkpoint SHA mismatch')
    return measured


def _print(**value):
    print(json.dumps(value,allow_nan=False),flush=True)


def write_status(path,payload):
    """Status is mutable; all source/result reports remain exclusive writes."""
    content=json.dumps(payload,indent=2,allow_nan=False)
    path=Path(path)
    temporary=path.with_suffix('.json.tmp')
    with temporary.open('x',encoding='utf-8') as stream:
        stream.write(content)
    temporary.replace(path)


@torch.inference_mode()
def fit_probe(frontend, index, legacy, raw, exact, output):
    """Same teacher inputs for alpha; raw student's OWN state for trajectories."""
    former = frontend.event_former
    prompt = former.build_prompt_cache(1)
    device = next(frontend.parameters()).device
    grouped = defaultdict(list)
    for run in index['runs']:
        if run['role'] == 'fit':
            grouped[run['record_uid']].append(run)
    differences, raw_negative, ratios, trace_rows, exact_errors, paired = [], [], [], [], [], []
    all_finite = True
    for source in index['records']:
        if source['role'] != 'fit':
            continue
        payload = _load(source)
        run = sorted(grouped[source['uid']],key=lambda r:(-(r['end']-r['start']),r['uid']))[0]
        radar = payload['radar'][run['start']:run['end']].to(device)
        masks = payload['radar_mask'][run['start']:run['end']].to(device)
        norms = []
        former.state_gate = legacy
        state = former.initial_state(1)
        for i in range(len(radar)):
            def capture(trace):
                nonlocal all_finite
                proposal = trace['E_hat']
                # Native _update casts the previous state to proposal dtype
                # before calling the gate; trace E_prev is the pre-cast value.
                previous = trace['E_prev'].to(proposal.dtype)
                _, a0 = legacy(previous,proposal,0.)
                _, a1 = legacy(previous,proposal,1.)
                _, ar = raw(previous,proposal)
                all_finite &= bool(torch.isfinite(ar).all())
                es, ae = exact(previous,proposal)
                target, at = legacy(previous,proposal,1.)
                differences.append((a1-a0).abs().float().cpu().flatten())
                raw_negative.append((ar<0).float().cpu().flatten())
                exact_errors.append(torch.tensor([float((ae-at).abs().max()),float((es-target).abs().max())]))
            state = former.stream_update(state,radar[i:i+1],min((i+1)*.18/1.2,1.),masks[i:i+1],prompt,
                                         trace=capture).event_tokens
            norms.append(float(state.float().norm()))
        former.state_gate = raw
        state = former.initial_state(1)
        for i in range(len(radar)):
            state = former.stream_update(state,radar[i:i+1],1.,masks[i:i+1],prompt).event_tokens
            finite = bool(torch.isfinite(state).all())
            all_finite &= finite
            value = float(state.float().norm()) if finite else None
            ratio = value/max(norms[i],1e-12) if value is not None else None
            if ratio is not None:
                ratios.append(ratio)
            trace_rows.append(dict(uid=run['uid'],step=i+1,within_first64=i<64,teacher_norm=norms[i],
                                   student_norm=value,norm_ratio=ratio,finite=finite))
            if not finite:
                break
        # Unlike the common-input alpha check, these two states independently
        # generate subsequent proposals, so FP16 recurrence drift stays visible.
        fixed=fixed_maturity_gate(legacy,maturity=1.).eval().requires_grad_(False)
        fixed_state=former.initial_state(1); exact_state=former.initial_state(1)
        for i in range(len(radar)):
            former.state_gate=fixed
            fixed_state=former.stream_update(fixed_state,radar[i:i+1],1.,masks[i:i+1],prompt).event_tokens
            former.state_gate=exact
            exact_state=former.stream_update(exact_state,radar[i:i+1],1.,masks[i:i+1],prompt).event_tokens
            error=exact_state.float()-fixed_state.float()
            key_error=None; key_cosine=None
            if i+1==min(64,len(radar)) or i+1==len(radar):
                fixed_key=frontend.semantic_projection(fixed_state).float()
                exact_key=frontend.semantic_projection(exact_state).float()
                key_error=float((fixed_key-exact_key).abs().max())
                key_cosine=float(F.cosine_similarity(fixed_key,exact_key,dim=-1).mean())
            paired.append(dict(uid=run['uid'],step=i+1,within_first64=i<64,
                state_max_error_fp32=float(error.abs().max()),
                normalized_state_error=float(error.norm()/fixed_state.float().norm().clamp_min(1e-12)),
                fixed_norm=float(fixed_state.float().norm()),exact_norm=float(exact_state.float().norm()),
                endpoint_key_max_error=key_error,endpoint_key_cosine=key_cosine))
        _print(phase='fit-zero-gate-probe',recording=source['uid'],run=run['uid'],updates=len(radar))
    if not raw_negative:
        raise ValueError('fit probe missing measurements')
    errors = torch.stack(exact_errors)
    summary = dict(all_finite=all_finite,negative_alpha_fraction=float(torch.cat(raw_negative).mean()),
                   state_norm_ratio_p99=tensor_summary(ratios)['p99'],
                   norm_ratio=tensor_summary(ratios),delta_alpha_m0_m1=tensor_summary(torch.cat(differences)),
                   same_input_exact_alpha_max_error=float(errors[:,0].max()),
                   same_input_exact_state_max_error=float(errors[:,1].max()),records=len(grouped),
                   actual_optimizer_steps=0)
    summary['usability'] = warmstart_usable(summary)
    from .prepare import write_json
    write_json(Path(output)/'fit-probe.json',dict(summary=summary,trajectory=trace_rows,
                                               paired_fixed_exact_trajectory=paired))
    return summary


@torch.inference_mode()
def evaluate_variant(frontend,tokenizer,index,teachers,output,*,legacy_rule):
    from .checkpoint import file_sha,module_digest
    from .train_stage2 import query_keys
    from .training import fixed_query_retrieval
    from .prepare import write_json
    output=Path(output); output.mkdir(parents=True,exist_ok=False)
    device=next(frontend.parameters()).device
    former=frontend.event_former
    prompt=former.build_prompt_cache(1)
    queries=query_keys(frontend,tokenizer)
    by_record=defaultdict(list)
    for run in index['runs']:
        if run['role']=='dev':
            by_record[run['record_uid']].append(run)
    teacher_records={r['uid']:r for r in teachers['records']}
    predictions=[]; failures=[]
    for source in index['records']:
        if source['role']!='dev':
            continue
        payload=_load(source); teacher_payload=_load(teacher_records[source['uid']])
        for run in by_record[source['uid']]:
            radar=payload['radar'][run['start']:run['end']].to(device)
            mask=payload['radar_mask'][run['start']:run['end']].to(device)
            state=former.initial_state(1)
            finite=True
            for i in range(len(radar)):
                state=former.stream_update(state,radar[i:i+1],min((i+1)*.18/1.2,1.) if legacy_rule else 1.,
                                          mask[i:i+1],prompt).event_tokens
                if not torch.isfinite(state).all():
                    failures.append(dict(uid=run['uid'],stage='stream',step=i+1,reason='nonfinite'))
                    finite=False; break
            commit=former.commit_refine(radar.flatten(0,1)[None],mask.flatten()[None],prompt).event_tokens
            teacher=teacher_payload[f"{run['uid']}:{run['end']-run['start']}"]
            if (teacher['visible_start'],teacher['visible_end'])!=(run['start'],run['end']):
                raise ValueError('teacher coverage mismatch')
            sk=frontend.semantic_projection(state)[0].float().cpu() if finite else None
            ck=frontend.semantic_projection(commit)[0].float().cpu()
            if not torch.isfinite(ck).all():
                ck=None; failures.append(dict(uid=run['uid'],stage='commit',reason='nonfinite'))
            if sk is not None and not torch.isfinite(sk).all():
                sk=None; failures.append(dict(uid=run['uid'],stage='stream-key',reason='nonfinite'))
            predictions.append(dict(uid=run['uid'],label=run['label'],metadata=source['metadata'],
                stream_key=sk,commit_key=ck,teacher_stream_key=teacher['stream_key'],teacher_commit_key=teacher['commit_key']))
        _print(phase='zero-gate-dev',variant=output.name,recording=source['uid'],events=len(by_record[source['uid']]))
    if not predictions:
        raise ValueError('no development predictions')
    torch.save(dict(predictions=predictions,queries=queries),output/'retrieval-predictions.pt')
    # Portable raw keys for independent local recomputation. Invalid keys stay NaN,
    # are explicitly listed as failures, and never enter passing macro metrics.
    arrays=dict(uid=np.array([r['uid'] for r in predictions]),labels=np.array([r['label'] for r in predictions]),
                subject=np.array([r['metadata']['subject_id'] for r in predictions]),
                sensor=np.array([r['metadata']['sensor_id'] for r in predictions]),
                role=np.array([r['metadata']['role'] for r in predictions]),queries=queries.numpy())
    for name in ('stream_key','commit_key','teacher_stream_key','teacher_commit_key'):
        arrays[name]=np.stack([r[name].numpy() if r[name] is not None else np.full(queries.shape[1],np.nan,dtype=np.float32)
                               for r in predictions])
    np.savez_compressed(output/'retrieval-arrays.npz',**arrays)
    groups={'dev':predictions,'dev_regression':[r for r in predictions if r['metadata']['role']=='dev_regression']}
    for subject,sensor in sorted(set(zip(arrays['subject'],arrays['sensor']))):
        groups[f'{subject}/{sensor}']=[r for r in predictions if
            (r['metadata']['subject_id'],r['metadata']['sensor_id'])==(subject,sensor)]
    metrics={}; passed=True
    for group,rows in groups.items():
        metrics[group]={}
        for path in ('stream','commit'):
            invalid=sum(r[f'{path}_key'] is None for r in rows)
            if invalid or not rows:
                metrics[group][path]=dict(passed=False,invalid_events=invalid,current=None,delta=None)
                passed=False; continue
            labels=torch.tensor([r['label'] for r in rows])
            current=fixed_query_retrieval(torch.stack([r[f'{path}_key'] for r in rows]),labels,queries)
            original=fixed_query_retrieval(torch.stack([r[f'teacher_{path}_key'] for r in rows]),labels,queries)
            delta={key:None if current[key] is None or original[key] is None else current[key]-original[key]
                   for key in ('macro_map','macro_r_at_1')}
            eligible=all(v is not None and v>=-(.01 if group in ('dev','dev_regression') else .03)-1e-10 for v in delta.values())
            cosine=F.cosine_similarity(torch.stack([r[f'{path}_key'] for r in rows]),
                                      torch.stack([r[f'teacher_{path}_key'] for r in rows]),dim=-1)
            metrics[group][path]=dict(passed=eligible,current=current,original=original,delta=delta,
                                     teacher_cosine=tensor_summary(cosine))
            passed &= eligible
    report=dict(passed=bool(passed),metrics=metrics,failures=failures,events=len(predictions),
                frontend_digest=module_digest(frontend),actual_optimizer_steps=0,
                predictions_sha256=file_sha(output/'retrieval-predictions.pt'),
                arrays_sha256=file_sha(output/'retrieval-arrays.npz'))
    write_json(output/'retrieval-gate.json',report)
    return report


def main():
    import argparse
    import yaml
    from .cache import load_original
    from .checkpoint import file_sha,module_digest
    from .prepare import write_json
    from .train_stage2 import frozen_digest
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    output=Path(args.output); output.mkdir(parents=True,exist_ok=False)
    config=yaml.safe_load(Path(args.config).read_text())
    prep=Path(config['native_preparation'])
    status=dict(phase='source-check',actual_optimizer_steps=0,holdout_access=False,official_data_access=False)
    def update(**kwargs):
        status.update(kwargs); write_status(output/'status.json',status); _print(**status)
    try:
        if (config.get('actual_optimizer_steps')!=0 or config.get('maturity_condition')!=1.
            or config.get('candidate_initialization')!='v25-age-free-tanh-warmstart-v1'
            or config.get('fit_usability')!=dict(max_negative_alpha_fraction=.5,
                                                max_state_norm_ratio_p99=4.,all_finite_required=True)):
            raise ValueError('diagnostic config differs from preregistered zero-step rules')
        for name in ('source_checkpoint','stage1_checkpoint'):
            if file_sha(config[name+'_path'])!=config[name+'_sha256']:
                raise ValueError(f'{name} SHA mismatch')
        index_path=prep/'encoder/index.json'; teacher_path=prep/'teachers/index.json'
        for path,key in ((index_path,'encoder_index_sha256'),(teacher_path,'teachers_index_sha256')):
            if file_sha(path)!=config[key]:
                raise ValueError(f'{key} mismatch')
        index=json.loads(index_path.read_text()); teachers=json.loads(teacher_path.read_text())
        frontend_config_sha=validate_inference_source(config,index)
        if any(r['role'] not in ('fit','dev') for r in index['records']+index['runs']):
            raise ValueError('sealed role in diagnostic index')
        frontend,tokenizer=load_original(config['frontend_config'],config['source_checkpoint_path'],torch.device('cuda'))
        frontend.eval().requires_grad_(False)
        if module_digest(frontend)!=teachers['original_frontend_digest']:
            raise ValueError('original teacher frontend digest mismatch')
        frozen=frozen_digest(frontend); legacy=frontend.event_former.state_gate
        legacy_before=module_digest(legacy)
        raw=transferred_gate(legacy,maturity=1.,exact=False).eval().requires_grad_(False)
        exact=transferred_gate(legacy,maturity=1.,exact=True).eval().requires_grad_(False)
        update(phase='fit-probe')
        fit=fit_probe(frontend,index,legacy,raw,exact,output)
        reports={}
        variants=[('A-original',legacy,True),('B-m0',fixed_maturity_gate(legacy,maturity=0.),False),
                  ('B-m1',fixed_maturity_gate(legacy,maturity=1.),False),
                  ('C-raw-m1',raw,False),('C-exact-m1',exact,False)]
        for name,gate,legacy_rule in variants:
            frontend.event_former.state_gate=gate.eval().requires_grad_(False)
            update(phase='development-diagnostic',variant=name,fit_warmstart_passed=fit['usability']['passed'])
            reports[name]=evaluate_variant(frontend,tokenizer,index,teachers,output/name,legacy_rule=legacy_rule)
            if frozen_digest(frontend)!=frozen or module_digest(legacy)!=legacy_before:
                raise RuntimeError('frozen source changed during diagnostic')
        frontend.event_former.state_gate=legacy
        result=dict(actual_optimizer_steps=0,fit=fit,variants=reports,frozen_unchanged=True,
                    teacher_gate_unchanged=True,holdout_access=False,official_data_access=False,formal_go=False,
                    config_sha256=file_sha(args.config),encoder_index_sha256=file_sha(index_path),
                    frontend_config_sha256=frontend_config_sha,
                    teachers_index_sha256=file_sha(teacher_path),legacy_gate_digest=legacy_before,
                    warmstart_gate_digest=module_digest(raw),training_started=False,
                    next_action='migration-implementation' if fit['usability']['passed'] else 'stop-unusable-warmstart')
        write_json(output/'result.json',result)
        update(phase='completed',next_action=result['next_action'],fit_warmstart_passed=fit['usability']['passed'])
    except BaseException as exc:
        update(phase='failed',error=repr(exc),training_started=False)
        raise


if __name__=='__main__':
    main()
