"""Raw-radar full-pipeline replay with fixed questions and paired old baseline."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict
import json
from pathlib import Path
import time

import numpy as np
import torch

from .checkpoint import file_sha,module_digest
from .collect import RecordingController,recording_intervals
from .controller import ControllerConfig
from .data import SENSORS,require_array_access
from .evidence import EvidenceConfig
from .pipeline import V25Pipeline,FrozenQA
from .prepare import write_json
from .runtime import V25Runtime


def fixed_questions(record_uid,intervals):
    from ..parrad_single_event_qa import IDENTIFY_QUERY,CONFUSIONS,presence_query
    from ..parrad_stage4_qa import ACTIVITIES
    rows=[]
    for number,(start,end,label) in enumerate(intervals):
        if label<0:
            continue
        prefix=f'During the time window [{start:.2f}, {end:.2f}) seconds, '
        base=dict(record_uid=record_uid,start=start,end=end,label=label)
        rows.append(dict(**base,uid=f'{record_uid}:gt{number}:identify',kind='identify',
            query=prefix+IDENTIFY_QUERY,reference=ACTIVITIES[label]))
        for answer,query_label in (('Yes',label),('No',CONFUSIONS[label][0])):
            rows.append(dict(**base,uid=f'{record_uid}:gt{number}:{answer}',kind='presence',
                query=prefix+presence_query(query_label),reference=answer,pair_id=f'{record_uid}:gt{number}'))
    return rows


def compare_boundary(current,original):
    current_ids=[r['record_uid'] for r in current['recordings']]
    original_ids=[r['record_uid'] for r in original['recordings']]
    if sorted(current_ids)!=sorted(original_ids) or len(set(current_ids))!=len(current_ids):
        raise ValueError('paired boundary comparison requires same recording IDs')
    def metric(report):
        rows=report['recordings']
        return sum(r['metrics']['0.36']['f1'] for r in rows)/len(rows),sum(r['metrics']['0.36']['precision'] for r in rows)/len(rows)
    new_f1,precision=metric(current); old_f1,_=metric(original)
    groups={}
    for subject in {r['subject_id'] for r in original['recordings']}:
        old=[r['metrics']['0.36']['f1'] for r in original['recordings'] if r['subject_id']==subject]
        new=[r['metrics']['0.36']['f1'] for r in current['recordings'] if r['subject_id']==subject]
        groups[subject]=None if not new else sum(new)/len(new)-sum(old)/len(old)
    return dict(passed=bool(new_f1-old_f1>=.10-1e-10 and precision>=.45 and
                           all(delta is not None and delta>=-.02-1e-10 for delta in groups.values())),
                original_macro_f1=old_f1,current_macro_f1=new_f1,delta=new_f1-old_f1,
                current_macro_precision=precision,subject_delta=groups)


class CheckedController(RecordingController):
    def reset(self):
        super().reset()
        self.rebuild_checks=[]

    def _rebuild(self,event_id,start_time,frames,initial):
        result,history=super()._rebuild(event_id,start_time,frames,initial)
        # Independently reconstruct the recurrent Event Tokens from the same
        # immutable initial state and frame range, without _advance/_rebuild.
        tokens=self.event_former.initial_state(1) if initial is None else initial.event_tokens.clone()
        for frame in frames:
            tokens=self.event_former.stream_update(tokens,frame.tokens,1.,frame.mask,self.prompt_cache).event_tokens
        error=float((tokens.float()-result.event_tokens.float()).abs().max())
        self.rebuild_checks.append(dict(event_id=event_id,watermark=result.watermark,reference_watermark=frames[-1].sequence,
                                        max_abs_error_fp32=error,passed=error<=1e-5))
        if error>1e-5:
            raise RuntimeError('independent same-watermark event replay mismatch')
        return result,history


def load_memory_and_qa(config,frontend,tokenizer,device):
    from ..models.memory import QueryCompressor
    from ..models.qa_adapter import EventQAAdapter
    from ..parrad_memory import EventMemory
    for name in ('qa','compressor'):
        if file_sha(config[name+'_path'])!=config[name+'_sha256']:
            raise ValueError(f'protected {name} source SHA mismatch')
    payload=torch.load(config['compressor_path'],map_location='cpu',weights_only=False)
    compressor=QueryCompressor(896,internal_dim=1024,heads=8,layers=2,summary_tokens=4,dropout=.1)
    compressor.load_state_dict(payload['model_state'],strict=True)
    compressor.to(device).eval().requires_grad_(False)
    saved=torch.load(config['qa_path'],map_location='cpu',weights_only=False)
    adapter=EventQAAdapter(**saved['model_settings'])
    adapter.load_state_dict(saved['model_state'],strict=True)
    adapter.to(device).eval().requires_grad_(False)
    qa=FrozenQA.from_native_reader(adapter,frontend.event_former.adapter,tokenizer,max_new_tokens=24)
    return EventMemory(compressor,frontend.semantic_projection),qa


def _boundary_report(records):
    from ..parrad_stage3_diagnostics import match_boundaries_seconds
    for row in records:
        row['metrics']={}
        for tolerance in (.18,.36,.54,1.):
            match=match_boundaries_seconds(sorted(row['predictions']),row['truth'],tolerance)
            row['metrics'][str(tolerance)]=dict(tp=match.true_positives,fp=match.false_positives,fn=match.false_negatives,
                precision=match.precision,recall=match.recall,f1=match.f1,
                mae=sum(match.absolute_errors_s)/len(match.absolute_errors_s) if match.absolute_errors_s else None)
    return dict(recordings=records)


def qa_scores(rows):
    from ..parrad_single_event_qa import score_predictions
    report=score_predictions(rows)
    groups={}
    for field in ('role','subject_id','sensor_id'):
        for value in sorted({r[field] for r in rows}):
            name=value if field=='role' else f'{field}/{value}'
            groups[name]=score_predictions([r for r in rows if r[field]==value])
    report['groups']=groups
    return report


@torch.inference_mode()
def evaluate_continuous(frontend,stage,index,memory,qa,tokenizer,output,*,controller_config,evidence_config,
                        role='dev',terminal_locked=False,paced=False):
    from ..stages.parrad_stage4 import encode_query
    from ..parrad_single_event_qa import score_predictions
    from ..parrad_stage4_qa import grade_answer
    output=Path(output); output.mkdir(parents=True,exist_ok=False)
    device=next(frontend.parameters()).device
    qa_digest=module_digest(qa.adapter); compressor_digest=module_digest(memory.compressor)
    records=[]; answers=[]; raw=[]
    for source in index['records']:
        if source['role']!=role:
            continue
        metadata=source['metadata']
        require_array_access(metadata,allowed_roles=('dev_regression','dev_new','session_holdout','subject_holdout'),
                             terminal_locked=terminal_locked)
        with np.load(metadata['path'],allow_pickle=False) as arrays:
            md=torch.from_numpy(np.asarray(arrays['md'],dtype=np.float32))
            labels=torch.from_numpy(np.asarray(arrays['labels'],dtype=np.int64))
            timestamps=torch.from_numpy(np.asarray(arrays['timestamps'],dtype=np.float64))
        frames=len(md)//2*2
        offline=dict(labels=labels[:frames].reshape(-1,2),timestamps=timestamps[:frames].reshape(-1,2))
        intervals=recording_intervals(offline)
        questions=fixed_questions(source['uid'],intervals)
        memory.levels=[[],[],[]]; memory.promotions=[0,0]
        controller=CheckedController(frontend.event_former,frontend.semantic_projection,stage.heads,controller_config,evidence_config)
        runtime=V25Runtime(frontend.radar_encoder,controller)
        contexts=[]; latency=[]; backlogs=[]; question_index=0; begin=time.monotonic()
        with V25Pipeline(runtime,memory,qa=qa) as pipeline:
            for update,start in enumerate(range(0,frames,2)):
                if paced:
                    wait=begin+update*.18-time.monotonic()
                    if wait>0:
                        time.sleep(wait)
                wall=time.monotonic()
                pipeline.step(torch.empty((1,2,0,0,0),device=device),md[start:start+2][None].to(device),
                    torch.tensor([SENSORS.index(metadata['sensor_id'])],device=device),
                    torch.ones((1,2),dtype=torch.bool,device=device),float(timestamps[start]),
                    metadata={k:metadata[k] for k in ('subject_id','session_id','sensor_id','environment_id')})
                if device.type=='cuda':
                    torch.cuda.synchronize()
                latency.append(time.monotonic()-wall)
                backlogs.append(len(pipeline.retained_jobs)+len(controller._boundaries)+(controller._pending is not None))
                if not paced:
                    now=float(timestamps[start+1])+.09
                    while question_index<len(questions) and questions[question_index]['end']<=now+1e-8:
                        question=questions[question_index]; question_index+=1
                        key=encode_query(frontend.event_former.adapter,frontend.semantic_projection,tokenizer,question['query'],device)
                        context,selected=pipeline.context(key,now=now,include_active=True)
                        contexts.append(dict(question=question,context=context.cpu().clone(),
                            memory_ids=[unit.uid for unit in selected],captured_at=now))
            pipeline.finish(end_time=float(timestamps[frames-1])+.09,include_unconfirmed_tail=True,timeout_s=120.)
            pipeline.flush(timeout_s=120.)
            # Drain experiments' orphan workers before using any other frontend.
            controller._executor.shutdown(wait=True,cancel_futures=True)
            controller._boundary_executor.shutdown(wait=True,cancel_futures=True)
            events=list(pipeline.completed_events)
            confirms=[e for e in events if e['confirmed']]
            accepted=sum(ep.decision=='confirm' for ep in controller.episodes)
            no_drop=len(confirms)==accepted and not pipeline.pending_finalizations and not pipeline.retained_jobs and not pipeline.failed_jobs
            p95=float(np.quantile(latency,.95))
            growing=(np.mean(backlogs[-min(300,len(backlogs)):])>np.mean(backlogs[:min(300,len(backlogs))])+1)
            engineering=dict(record_uid=source['uid'],updates=len(latency),accepted_confirms=accepted,committed_confirms=len(confirms),
                eof_unconfirmed=sum(not e['confirmed'] for e in events),no_drop=no_drop,flush_pending=0,
                p95_update_seconds=p95,latencies=latency,backlogs=backlogs,metrics=dict(controller.metrics),
                rebuild_checks=list(controller.rebuild_checks),paced=paced,backlog_growing=bool(growing),
                actual_atomic_swaps=controller.metrics['atomic_swaps'],cancelled_or_stale_jobs=controller.metrics['stale_jobs'])
            raw.append(dict(record_uid=source['uid'],events=events,contexts=contexts,engineering=engineering,memory=memory.snapshot()))
            truth=[right[0] for left,right in zip(intervals,intervals[1:]) if left[2]>=0 and right[2]>=0 and left[2]!=right[2]]
            records.append(dict(record_uid=source['uid'],subject_id=metadata['subject_id'],sensor_id=metadata['sensor_id'],
                predictions=sorted(e['end'] for e in confirms),truth=truth,engineering=engineering))
        if not paced:
            for captured in contexts:
                question={**captured['question'],**{k:metadata[k] for k in ('role','subject_id','sensor_id')}}
                answer=qa.answer(captured['context'].to(device),question['query'])
                answers.append(dict(**question,generated=answer,correct=grade_answer(answer,question['reference'],
                    'presence' if question['kind']=='presence' else 'first'),memory_ids=captured['memory_ids'],
                    captured_at=captured['captured_at']))
            if len(contexts)!=len(questions):
                raise RuntimeError('fixed question context capture incomplete')
        print(json.dumps(dict(phase='paced-replay' if paced else 'full-memory-qa-replay',recording=source['uid'],questions=len(contexts),p95=p95)),flush=True)
        if paced:
            break  # One complete ~10min recording, never a short smoke substituted.
    if not records:
        raise ValueError('no authorized evaluation recordings')
    torch.save(dict(records=raw),output/'raw-replay.pt')
    write_json(output/'answers.json',answers)
    unchanged=module_digest(qa.adapter)==qa_digest and module_digest(memory.compressor)==compressor_digest
    report=dict(boundary=_boundary_report(records),qa=qa_scores(answers) if answers else None,
        engineering_passed=bool(unchanged and all(r['engineering']['no_drop'] and
            all(c['passed'] for c in r['engineering']['rebuild_checks']) for r in records)),
        realtime_passed=bool(paced and len(records[0]['engineering']['latencies'])*.18>=600 and
            records[0]['engineering']['p95_update_seconds']<=.18 and not records[0]['engineering']['backlog_growing']),
        frozen_qa_compressor_unchanged=unchanged,answers_sha256=file_sha(output/'answers.json'),
        raw_replay_sha256=file_sha(output/'raw-replay.pt'),formal_go=False)
    write_json(output/'report.json',report)
    return report


@torch.inference_mode()
def evaluate_original(config,index,output,*,role='dev',terminal_locked=False):
    """Fixed old Stage2/Verify-controller; never reads official val/test arrays."""
    from ..parrad_stage3_calibrate import build_runtime,_thresholds_from_artifact
    from ..parrad_stage3_diagnostics import _apply_thresholds
    from ..stages.parrad_stage4 import encode_query
    from ..parrad_single_event_qa import score_predictions
    from ..parrad_stage4_qa import grade_answer
    from ..parrad_memory import MemoryUnit
    from transformers import AutoTokenizer
    import yaml
    output=Path(output); output.mkdir(parents=True,exist_ok=False)
    for name in ('legacy_stage3','legacy_threshold'):
        if file_sha(config[name+'_path'])!=config[name+'_sha256']:
            raise ValueError(f'old baseline locked {name} SHA mismatch')
    device=torch.device('cuda')
    runtime=build_runtime(Path(config['frontend_config']),Path(config['legacy_stage3_path']),device)
    _apply_thresholds(runtime,_thresholds_from_artifact(Path(config['legacy_threshold_path'])))
    raw_config=yaml.safe_load(Path(config['frontend_config']).read_text())
    tokenizer=AutoTokenizer.from_pretrained(raw_config['llm']['name_or_path'])
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token=tokenizer.eos_token
    front=torch.nn.Module()
    front.radar_encoder=runtime.radar_encoder; front.event_former=runtime.controller.event_former
    front.semantic_projection=runtime.controller.semantic_projection
    front.eval().requires_grad_(False)
    memory,qa=load_memory_and_qa(config,front,tokenizer,device)
    records=[]; answers=[]; snapshots=[]
    for source in index['records']:
        if source['role']!=role:
            continue
        metadata=source['metadata']
        require_array_access(metadata,allowed_roles=('dev_regression','dev_new','session_holdout','subject_holdout'),terminal_locked=terminal_locked)
        with np.load(metadata['path'],allow_pickle=False) as arrays:
            md=torch.from_numpy(np.asarray(arrays['md'],dtype=np.float32))
            labels=torch.from_numpy(np.asarray(arrays['labels'],dtype=np.int64))
            timestamps=torch.from_numpy(np.asarray(arrays['timestamps'],dtype=np.float64))
        frames=len(md)//2*2
        intervals=recording_intervals(dict(labels=labels[:frames].reshape(-1,2),timestamps=timestamps[:frames].reshape(-1,2)))
        questions=fixed_questions(source['uid'],intervals)
        memory.levels=[[],[],[]]; memory.promotions=[0,0]
        runtime.reset(); contexts=[]; predictions=[]; event_count=0; qindex=0
        prompt=front.event_former.build_prompt_cache(1)
        for start in range(0,frames,2):
            now=float(timestamps[start+1])+.09
            result=runtime.step(torch.empty((1,2,0,0,0),device=device),md[start:start+2][None].to(device),
                torch.tensor([SENSORS.index(metadata['sensor_id'])],device=device),
                torch.ones((1,2),dtype=torch.bool,device=device),float(timestamps[start]),metadata)
            job=result.finalization_job
            if job is not None:
                tokens=front.event_former.commit_refine(job.radar_tokens,job.radar_mask,prompt).event_tokens[0]
                key=front.semantic_projection(tokens[None])[0]
                memory.add(MemoryUnit(f'old:{event_count}',tokens,job.start_time,job.end_time,now,1.,key))
                event_count+=1; predictions.append(job.end_time)
            while qindex<len(questions) and questions[qindex]['end']<=now+1e-8:
                question=questions[qindex]; qindex+=1
                key=encode_query(front.event_former.adapter,front.semantic_projection,tokenizer,question['query'],device)
                branch=runtime.controller.shadow if runtime.controller.candidate is not None else runtime.controller.active
                context,selected=memory.context(key,now=now,current=branch.event_tokens)
                contexts.append(dict(question=question,context=context.cpu().clone(),memory_ids=[u.uid for u in selected],captured_at=now))
        if len(contexts)!=len(questions):
            raise RuntimeError('old baseline fixed question contexts incomplete')
        for captured in contexts:
            question={**captured['question'],**{k:metadata[k] for k in ('role','subject_id','sensor_id')}}
            answer=qa.answer(captured['context'].to(device),question['query'])
            answers.append(dict(**question,generated=answer,correct=grade_answer(answer,question['reference'],
                'presence' if question['kind']=='presence' else 'first'),memory_ids=captured['memory_ids'],captured_at=captured['captured_at']))
        truth=[right[0] for left,right in zip(intervals,intervals[1:]) if left[2]>=0 and right[2]>=0 and left[2]!=right[2]]
        records.append(dict(record_uid=source['uid'],subject_id=metadata['subject_id'],sensor_id=metadata['sensor_id'],predictions=predictions,truth=truth))
        snapshots.append(dict(record_uid=source['uid'],contexts=contexts,memory=memory.snapshot()))
        print(json.dumps(dict(phase='old-complete-baseline',recording=source['uid'],questions=len(contexts))),flush=True)
    torch.save(snapshots,output/'raw-replay.pt'); write_json(output/'answers.json',answers)
    report=dict(boundary=_boundary_report(records),qa=qa_scores(answers),answers_sha256=file_sha(output/'answers.json'),
                source={name:config[name+'_sha256'] for name in ('legacy_stage3','legacy_threshold','qa','compressor')})
    write_json(output/'report.json',report)
    return report


def compare_qa(current,original):
    deltas={key:None if current[key] is None or original[key] is None else current[key]-original[key]
            for key in ('identify_macro_recall','presence_balanced_accuracy')}
    missing=[key for key,value in current['identify_classes'].items() if value is None]
    regression=None
    if 'dev_regression' in current.get('groups',{}) or 'dev_regression' in original.get('groups',{}):
        if not all('dev_regression' in report.get('groups',{}) for report in (current,original)):
            raise ValueError('paired dev regression QA group required')
        regression=compare_qa(current['groups']['dev_regression'],original['groups']['dev_regression'])
    return dict(passed=bool(not missing and all(value is not None and value>=-.02-1e-10 for value in deltas.values())
                           and (regression is None or regression['passed'])),
                delta=deltas,missing_labels=missing,dev_regression=regression,current=current,original=original)
