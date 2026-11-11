"""Run a synthetic wiring smoke or train sealed Stage3 feature batches."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import torch
from torch import nn

from .checkpoint import file_sha,module_digest


def smoke(output,updates=40):
    from ..config import ModelConfig
    from ..models.event_former import TinyLLMAdapter,StreamingSemanticEventFormer
    from ..models.semantic import SemanticProjection
    from ..models.memory import QueryCompressor
    from ..models.qa_adapter import EventQAAdapter
    from ..parrad_memory import EventMemory
    from .models import install_age_free_gate,V25Heads
    from .controller import V25Controller,ControllerConfig
    from .evidence import EvidenceConfig
    from .runtime import V25Runtime
    from .pipeline import V25Pipeline,FrozenQA
    torch.manual_seed(23)
    torch.set_num_threads(1)
    output = Path(output)
    output.mkdir(parents=True,exist_ok=False)
    class Encoder(nn.Module):
        def __init__(self):
            super().__init__(); self.calls = 0
        def forward_update(self,rd,md,**kwargs):
            self.calls += 1
            return SimpleNamespace(radar_tokens=rd,radar_token_mask=kwargs['frame_mask'],state=self.calls)
    config = ModelConfig(radar_dim=8,llm_dim=16,llm_heads=4,llm_layers=2,cross_attn_dim=16,
                         cross_attn_heads=4,semantic_dim=8,dropout=0.,cross_attention_init_scale=.1)
    former = install_age_free_gate(StreamingSemanticEventFormer(TinyLLMAdapter(32,16,2,4),config,torch.tensor([1,2])))
    p = SemanticProjection(16,8)
    before = module_digest(p)
    heads = V25Heads(radar_dim=8,event_dim=16,semantic_dim=8)
    with torch.no_grad():
        for last in (heads.local.output[-1],heads.semantic.network[-1]):
            last.weight.zero_(); last.bias.fill_(20.)
        heads.radar.classifier[-1].weight.zero_(); heads.radar.classifier[-1].bias.fill_(-20.)
    controller = V25Controller(former,p,heads,ControllerConfig(proposal_persistence=1,boundary_semantic_weight=0.,boundary_radar_weight=0.),
                               EvidenceConfig(warmup_s=.3,h_plus=.1,tau_stability=0.,evidence_weight=0.))
    encoder = Encoder()
    memory = EventMemory(QueryCompressor(16,internal_dim=16,heads=4,layers=1,dropout=0.),p,
                         l0_capacity=2,l1_capacity=2,l2_capacity=2,group_size=2)
    reader = TinyLLMAdapter(32,16,2,4).eval()
    def answerer(context,query):
        ids = torch.tensor([[3,4]])
        return reader.caption_logits(context,ids,torch.ones_like(ids,dtype=torch.bool)).argmax(-1).tolist()
    with V25Pipeline(V25Runtime(encoder,controller),memory,qa=FrozenQA(EventQAAdapter(16,8),answerer)) as pipeline:
        for index in range(updates):
            pipeline.step(torch.randn(1,4,8),torch.zeros(1),torch.tensor([0]),torch.ones(1,4,dtype=torch.bool),index*.18)
        report = pipeline.finish(include_unconfirmed_tail=True)
        report.update(scope='synthetic TinyLLM interface smoke; forced heads; no accuracy measurement',
            updates=updates,encoder_calls=encoder.calls,p_unchanged=before == module_digest(p),
            memory_counts=[len(level) for level in memory.levels],promotions=memory.promotions,
            qa_interface= pipeline.answer('what happened?',torch.randn(8),now=updates*.18,include_active=False))
        torch.save(pipeline.snapshot(),output/'memory.pt')
        report['memory_sha256'] = file_sha(output/'memory.pt')
        (output/'episodes.json').write_text(json.dumps([ep.to_dict() for ep in controller.episodes],indent=2),encoding='utf-8')
        (output/'result.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False))


def train_heads(args):
    from .factory import load_frontend
    from .checkpoint import save_bundle
    from .stage3 import validate_feature_cache,fixed_budget_train
    from .training import probe_semantic_head
    from .controller import ControllerConfig
    from .evidence import EvidenceConfig
    torch.manual_seed(args.seed)
    output = Path(args.output)
    output.mkdir(parents=True,exist_ok=False)
    frontend,stage,_,saved = load_frontend(args.config,args.bundle,bundle_sha256=args.bundle_sha256,
                                          device=args.device,allow_development=True)
    if file_sha(args.cache) != args.cache_sha256:
        raise ValueError('training cache SHA mismatch')
    cache = torch.load(args.cache,map_location='cpu',weights_only=True)
    validate_feature_cache(cache,module_digest(frontend))
    rows = cache['batches'][args.phase]
    probe = None if args.phase == 'semantic' else saved['initialization_probe']
    report = fixed_budget_train(stage,rows,phase=args.phase,steps=args.steps,lr=args.lr)
    if args.probe_cache is not None:
        if args.probe_cache_sha256 is None or file_sha(args.probe_cache) != args.probe_cache_sha256:
            raise ValueError('development probe cache SHA mismatch')
        dev_cache = torch.load(args.probe_cache,map_location='cpu',weights_only=True)
        validate_feature_cache(dev_cache,module_digest(frontend),split='development')
        criteria = json.loads(Path(args.probe_criteria).read_text(encoding='utf-8')) if args.probe_criteria else None
        probe = probe_semantic_head(stage.heads.semantic,dev_cache['batches']['semantic'],
            frontend_digest=module_digest(frontend),cache_sha256=args.probe_cache_sha256,
            early_max_age=args.early_max_age,split='development',criteria=criteria)
    torch.save(report,output/'optimizer-readback.pt')
    # A new training phase invalidates previously selected thresholds.
    digest = save_bundle(output/'v25.pt',frontend,stage,ControllerConfig(**saved['controller_config']),
        EvidenceConfig(**saved['evidence_config']),provenance={**saved['provenance'],'data_sha256':args.cache_sha256},
        frontend_eventformer_steps=saved['frontend_eventformer_steps'],initialization_probe=probe,calibration=None)
    result = dict(phase=args.phase,actual_optimizer_steps=report['actual_optimizer_steps'],seed=args.seed,
                  checkpoint_sha256=digest,formal_go=False,calibration_required=True)
    (output/'result.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command',required=True)
    quick = sub.add_parser('smoke'); quick.add_argument('--output',required=True); quick.add_argument('--updates',type=int,default=40)
    train = sub.add_parser('train-heads')
    for name in ('config','bundle','bundle-sha256','cache','cache-sha256','output'):
        train.add_argument('--'+name,required=True)
    train.add_argument('--phase',choices=['physical','semantic','evidence'],required=True)
    train.add_argument('--steps',type=int,required=True); train.add_argument('--lr',type=float,default=1e-3)
    train.add_argument('--seed',type=int,default=23); train.add_argument('--early-max-age',type=int,default=2)
    train.add_argument('--device',default='cpu')
    train.add_argument('--probe-cache'); train.add_argument('--probe-cache-sha256'); train.add_argument('--probe-criteria')
    args = parser.parse_args()
    if args.command == 'smoke':
        smoke(args.output,args.updates)
    else:
        train_heads(args)


if __name__ == '__main__':
    main()
