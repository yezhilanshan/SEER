"""Real controller collection; GT is attached only after online decisions."""
from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

import torch

from .checkpoint import file_sha,module_digest
from .controller import V25Controller,ControllerConfig
from .buffer import FeatureFrame
from .evidence import EvidenceConfig
from .prepare import write_json
from .training import config_digest


COLLECTION_CONTROLLER=ControllerConfig(proposal_threshold=.55,proposal_persistence=2,delta_pre_s=.54)
COLLECTION_EVIDENCE=EvidenceConfig(rho=.95,tau_difference=.6,tau_stability=.5,h_plus=3.,h_minus=-3.,
                                  evidence_weight=1.,tau_evidence=.2,warmup_s=.54,timeout_s=1.98)


def candidate_training_bucket(age_seconds):
    ordinal=int(round(age_seconds/.18))+1
    if not 1<=ordinal<=11:
        return None
    return 0 if ordinal<=3 else 1 if ordinal<=6 else 2 if ordinal<=9 else 3


def visible_pair_label(intervals,*,reference_time,current_time,min_stable_s=.54):
    def at(timestamp):
        return next((r for r in intervals if r[0]-1e-8<=timestamp<r[1]-1e-8),None)
    reference,current=at(reference_time),at(current_time)
    if reference is None or current is None or reference[2]<0 or current[2]<0:
        return None
    # The reference endpoint must itself be established, not a mixed transition.
    if reference_time-reference[0]+.18<min_stable_s-1e-8:
        return None
    if current[2]==reference[2]:
        return 0
    if current_time-current[0]+.18<min_stable_s-1e-8:
        return None
    return 1


def recording_intervals(record):
    labels=record['labels'].flatten().tolist()
    timestamps=record['timestamps'].flatten().tolist()
    result=[]; i=0
    while i<len(labels):
        stop=i+1
        while stop<len(labels) and labels[stop]==labels[i]:
            stop+=1
        # Ignored spans remain explicit; do not bridge across them.
        result.append((timestamps[i],timestamps[stop-1]+.09,labels[i]))
        i=stop
    return result


class RecordingController(V25Controller):
    """Capture exactly the key pair supplied to the head at each real update."""
    def reset(self):
        super().reset()
        self.pairs=[]; self.reference_times={}; self.snapshots=[]
        self.boundary_windows=[]

    def _observe(self,timestamp):
        uid=self._episode.candidate_id
        self.reference_times[uid]=self.reference.timestamp
        result=super()._observe(timestamp)
        self.pairs.append(dict(candidate_id=uid,active=self._reference_key[0].float().cpu().clone(),
            candidate=self._candidate_records[-1][0][0].float().cpu().clone(),timestamp=timestamp,
            reference_time=self.reference.timestamp,age_seconds=self._episode.observations[-1]['age_s']))
        return result

    def _localize(self,frames,proposal_time,confirm_time,active_start,parent):
        result,resolved_start=super()._localize(frames,proposal_time,confirm_time,active_start,parent)
        cpu_frames=[FeatureFrame(f.sequence,f.timestamp,f.tokens.cpu(),f.mask.cpu(),f.local_probability,
                    f.radar_probability,f.semantic_key.cpu(),f.evidence.cpu()) for f in frames]
        self.boundary_windows.append(dict(frames=cpu_frames,proposal_time=proposal_time,
            confirm_time=confirm_time,active_start=resolved_start,selected_boundary=result.timestamp))
        return result,resolved_start


@torch.no_grad()
def collect_candidate_episodes(frontend,stage,index,output,*,role,controller_config=COLLECTION_CONTROLLER,
                               evidence_config=COLLECTION_EVIDENCE):
    if role not in ('fit','dev'):
        raise ValueError('only authorized fit/dev collection before terminal lock')
    output=Path(output); output.mkdir(parents=True,exist_ok=False)
    device=next(frontend.parameters()).device
    all_pairs=[]; episodes=[]; counts={}; traces=[]; windows=[]
    for source in index['records']:
        if source['role']!=role:
            continue
        if file_sha(source['path'])!=source['sha256']:
            raise ValueError('collection encoder cache source SHA changed')
        record=torch.load(source['path'],map_location='cpu',weights_only=False)
        controller=RecordingController(frontend.event_former,frontend.semantic_projection,stage.heads,
                                       controller_config,evidence_config)
        try:
            for i in range(len(record['radar'])):
                controller.step_tokens(record['radar'][i:i+1].to(device),record['radar_mask'][i:i+1].to(device),
                    float(record['timestamps'][i,0]),metadata={k:source['metadata'][k]
                        for k in ('subject_id','session_id','sensor_id','environment_id')})
            controller.finish()
            controller.flush()
            if controller.failures:
                raise RuntimeError('controller background failures: '+str(controller.failures))
            intervals=recording_intervals(record)
            excluded=0
            for pair in controller.pairs:
                target=visible_pair_label(intervals,reference_time=pair['reference_time'],current_time=pair['timestamp'])
                if target is None:
                    excluded+=1
                    continue
                bucket=candidate_training_bucket(pair['age_seconds'])
                if bucket is None:
                    excluded+=1
                    continue
                all_pairs.append(dict(**pair,target=target,age_bucket=bucket,
                    episode_id=f"{source['uid']}:{pair['candidate_id']}",record_uid=source['uid'],
                    subject_id=source['metadata']['subject_id'],sensor_id=source['metadata']['sensor_id']))
            for episode in controller.episodes:
                row=episode.to_dict()
                reference_time=controller.reference_times[episode.candidate_id]
                row.update(record_uid=source['uid'],episode_id=f"{source['uid']}:{episode.candidate_id}",
                    reference_time=reference_time,gt_intervals=intervals,
                    gt_transition_exists=visible_pair_label(intervals,reference_time=reference_time,current_time=episode.end_time))
                episodes.append(row)
            counts[source['uid']]=dict(pairs=len(controller.pairs),excluded_pairs=excluded,
                episodes=len(controller.episodes),metrics=dict(controller.metrics),failures=list(controller.failures))
            traces.append(dict(record_uid=source['uid'],trace=list(controller.decision_trace)))
            windows.extend(dict(**window,record_uid=source['uid']) for window in controller.boundary_windows)
            print(json.dumps(dict(phase='candidate-collection',role=role,recording=source['uid'],**counts[source['uid']])),flush=True)
        finally:
            controller.close()
            controller._executor.shutdown(wait=True,cancel_futures=True)
            controller._boundary_executor.shutdown(wait=True,cancel_futures=True)
    torch.save(dict(pairs=all_pairs,episodes=episodes,traces=traces,boundary_windows=windows),output/'collection.pt')
    provenance=dict(frontend_digest=module_digest(frontend),heads_digest=module_digest(stage),
        semantic_head_digest=module_digest(stage.heads.semantic),controller_config_digest=config_digest(asdict(controller_config)),
        episodes_sha256=file_sha(output/'collection.pt'))
    manifest=dict(schema_version=25,role=role,counts=counts,provenance=provenance,
                  controller_config=asdict(controller_config),evidence_config=asdict(evidence_config),
                  gt_online_inputs=False,collection_sha256=file_sha(output/'collection.pt'))
    write_json(output/'manifest.json',manifest)
    if not all_pairs:
        raise ValueError('no eligible closed-loop pairs; full collection evidence saved')
    return dict(pairs=all_pairs,episodes=episodes,traces=traces,boundary_windows=windows,manifest=manifest)
