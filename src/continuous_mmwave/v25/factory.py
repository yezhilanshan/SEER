"""Explicit integration into existing PARrad construction and dataset interfaces."""
from pathlib import Path
import torch

from .checkpoint import read_bundle,load_bundle,file_sha
from .models import install_age_free_gate
from .stage3 import V25Stage3
from .controller import V25Controller,ControllerConfig
from .evidence import EvidenceConfig
from .runtime import V25Runtime


def load_frontend(config_path,bundle_path,*,bundle_sha256,device,allow_development=False):
    # Validate version/provenance before loading the native LLM.
    saved = read_bundle(bundle_path,expected_sha256=bundle_sha256,allow_development=allow_development)
    if file_sha(config_path) != saved['provenance']['frontend_config_sha256']:
        raise ValueError('frontend configuration source SHA mismatch')
    from ..train import _build_parrad_stage2,load_experiment_config
    experiment = load_experiment_config(Path(config_path))
    frontend,tokenizer = _build_parrad_stage2(experiment)
    install_age_free_gate(frontend.event_former)
    stage = V25Stage3(**saved['heads_settings'],update_period_s=saved['controller_config']['update_period_s'])
    load_bundle(bundle_path,frontend,stage,expected_sha256=bundle_sha256,allow_development=allow_development)
    return frontend.to(device),stage.to(device),tokenizer,saved


def build_runtime(config_path,bundle_path,*,bundle_sha256,device='cpu',allow_development=False):
    frontend,stage,tokenizer,saved = load_frontend(config_path,bundle_path,bundle_sha256=bundle_sha256,
                                                device=device,allow_development=allow_development)
    frontend.eval().requires_grad_(False)
    stage.eval().requires_grad_(False)
    controller = V25Controller(frontend.event_former,frontend.semantic_projection,stage.heads,
        ControllerConfig(**saved['controller_config']),EvidenceConfig(**saved['evidence_config']))
    return V25Runtime(frontend.radar_encoder,controller),tokenizer,saved
