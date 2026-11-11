"""Explicit v2.5 bundles. No legacy Verify checkpoint fallback."""
from __future__ import annotations

import hashlib
from dataclasses import asdict
from pathlib import Path
import torch

from .controller import ControllerConfig
from .evidence import EvidenceConfig
from .training import validate_bias_probe,config_digest

GATE_ARCHITECTURES = ('v25-age-free-tanh', 'v25-age-free-positive-coordinate-v1')


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):
            digest.update(block)
    return digest.hexdigest()


def state_digest(state):
    digest = hashlib.sha256()
    for name,value in sorted(state.items()):
        value = value.detach().cpu().contiguous()
        digest.update(f'{name}:{tuple(value.shape)}:{value.dtype}'.encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def module_digest(module):
    return state_digest(module.state_dict())


def code_digest():
    digest = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob('*.py')):
        digest.update(path.name.encode()); digest.update(path.read_bytes())
    # Shared implementation used by the new path is part of the source lock.
    root = Path(__file__).parent.parent
    for name in ('models/event_former.py','models/semantic.py','models/transition.py','models/memory.py',
                 'parrad_memory.py','stages/stage2.py','stages/parrad_stage4.py','train.py'):
        digest.update(name.encode()); digest.update((root/name).read_bytes())
    return digest.hexdigest()


def _sha_valid(value):
    return isinstance(value,str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def _ready(payload):
    if payload.get('frontend_gate_architecture','v25-age-free-tanh') != 'v25-age-free-tanh':
        return False  # Positive-coordinate bridge is development-only.
    counts = payload.get('optimizer_steps',{})
    if set(counts) != {'physical','semantic','evidence'} or not all(isinstance(v,int) and v > 0 for v in counts.values()):
        return False
    if not isinstance(payload.get('frontend_eventformer_steps'),int) or payload['frontend_eventformer_steps'] <= 0:
        return False
    try:
        semantic_state = {k.removeprefix('heads.semantic.'):v for k,v in payload['stage3_state'].items() if k.startswith('heads.semantic.')}
        semantic_digest = state_digest(semantic_state)
        validate_bias_probe(payload.get('initialization_probe'),frontend_digest=payload['frontend_digest'],semantic_head_digest=semantic_digest)
        probe=payload['initialization_probe']
        if probe.get('probe_schema')!=2 or probe.get('early_range')!=[.54,.72] or probe.get('mature_range')!=[1.08,1.98]:
            return False
        calibration = payload.get('calibration')
        if not isinstance(calibration,dict) or calibration.get('split') != 'development-episodes':
            return False
        selected = calibration.get('selected')
        source = calibration.get('source',{})
        limit = calibration.get('max_false_confirm_rate')
        return bool(isinstance(selected,dict) and selected.get('eligible') is True and selected.get('incomplete') == 0
            and selected in calibration.get('all_configurations',[]) and selected.get('config') == payload['evidence_config']
            and isinstance(limit,(int,float)) and 0 <= limit <= 1
            and 0 <= selected['false_confirm_rate'] <= limit and 0 <= selected['recall'] <= 1
            and source.get('frontend_digest') == payload['frontend_digest']
            and source.get('heads_digest') == payload['stage3_digest']
            and source.get('semantic_head_digest') == semantic_digest
            and source.get('controller_config_digest') == config_digest(payload['controller_config'])
            and _sha_valid(source.get('episodes_sha256'))
            and calibration.get('probe_criteria_sha256') == payload['initialization_probe']['criteria_sha256'])
    except (ValueError,TypeError,KeyError):
        return False


def save_bundle(path,frontend,stage3,controller_config,evidence_config,*,provenance,
                frontend_eventformer_steps=0,initialization_probe=None,calibration=None):
    path = Path(path)
    if path.exists():
        raise FileExistsError('write a new v2.5 checkpoint; do not overwrite evidence')
    gate_architecture = getattr(frontend.event_former.state_gate,'architecture',None)
    if gate_architecture not in GATE_ARCHITECTURES:
        raise ValueError('v2.5 age-free frontend required')
    if not all(_sha_valid(provenance.get(k)) for k in ('frontend_config_sha256','data_sha256')):
        raise ValueError('source SHA provenance required')
    payload = dict(schema_version=25,architecture='v25-sequential-evidence',
        frontend_gate_architecture=gate_architecture,
        status='development-untrained',
        frontend_state=frontend.state_dict(),stage3_state=stage3.state_dict(),
        frontend_digest=module_digest(frontend),stage3_digest=module_digest(stage3),
        heads_settings=stage3.heads.settings,controller_config=asdict(controller_config),evidence_config=asdict(evidence_config),
        optimizer_steps=dict(stage3.optimizer_steps),frontend_eventformer_steps=frontend_eventformer_steps,
        provenance={**provenance,'v25_code_sha256':code_digest()},initialization_probe=initialization_probe,
        calibration=calibration,formal_go=False)
    if _ready(payload):
        payload['status'] = 'calibrated-development'
    path.parent.mkdir(parents=True,exist_ok=True)
    # Exclusive creation prevents silently replacing any source checkpoint.
    with path.open('xb') as stream:
        torch.save(payload,stream)
    return file_sha(path)


def read_bundle(path,*,expected_sha256,allow_development=False):
    if not _sha_valid(expected_sha256) or file_sha(path) != expected_sha256:
        raise ValueError('v2.5 checkpoint SHA mismatch')
    payload = torch.load(path,map_location='cpu',weights_only=True)
    if payload.get('schema_version') != 25 or payload.get('architecture') != 'v25-sequential-evidence':
        raise ValueError('explicit v2.5 bundle required; legacy Verify weights cannot be loaded')
    if payload.get('frontend_gate_architecture','v25-age-free-tanh') not in GATE_ARCHITECTURES:
        raise ValueError('unsupported frontend gate architecture')
    if payload.get('status') == 'calibrated-development' and not _ready(payload):
        raise ValueError('v2.5 calibration readiness/source checks failed')
    if not allow_development and payload.get('status') != 'calibrated-development':
        raise ValueError('development checkpoint requires explicit allow_development=True')
    if payload['provenance'].get('v25_code_sha256') != code_digest():
        raise ValueError('v2.5 runtime code source SHA mismatch')
    if state_digest(payload['frontend_state']) != payload['frontend_digest'] or state_digest(payload['stage3_state']) != payload['stage3_digest']:
        raise ValueError('v2.5 saved tensor digest mismatch')
    ControllerConfig(**payload['controller_config']); EvidenceConfig(**payload['evidence_config'])
    return payload


def load_bundle(path,frontend,stage3,*,expected_sha256,allow_development=False):
    payload = read_bundle(path,expected_sha256=expected_sha256,allow_development=allow_development)
    if getattr(frontend.event_former.state_gate,'architecture',None) != payload.get('frontend_gate_architecture','v25-age-free-tanh'):
        raise ValueError('instantiate the exact saved gate architecture before loading')
    if stage3.heads.settings != payload['heads_settings']:
        raise ValueError('v2.5 head dimension mismatch')
    frontend.load_state_dict(payload['frontend_state'],strict=True)
    stage3.load_state_dict(payload['stage3_state'],strict=True)
    stage3.optimizer_steps = dict(payload['optimizer_steps'])
    return payload
