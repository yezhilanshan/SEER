"""Run fixed-panel G0 diagnostics before granting access to G2 experiments."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from continuous_mmwave.harrad.preprocess import write_json,sha256
from ..config import LatentMemoryConfig
from ..data import load_records,PHRASES
from ..runner import build_model,source_manifest
from .episodes import make_panel,choose_pilot,natural_order_pairs
from .grammar import DescriptionGrammar
from .core import new_state,load_new_state
from .training import train_stage,load_stage
from .evaluation import prepare_examples,donor_map,evaluate,text_oracle
from .workflow import pilot_gate,oracle_pass

INSTRUCTION=("Describe the observed human actions in chronological order. Output exactly one sentence: "
    "'The person <action>, then <action>.' Use only these action phrases: "+'; '.join(PHRASES)+
    ". Preserve each repeated instance. Add ' (ongoing)' after an action continuing at the end. "
    "For no annotated actions output 'No annotated action has been observed yet.'.")


def _load_config(path):
    settings=json.loads(Path(path).read_text(encoding='utf-8-sig'))
    allowed={'base_config','pilot_size','pilot_steps','formal_steps','max_new_tokens'}
    if set(settings)-allowed: raise ValueError('unknown experimental setting')
    for key,default in [('pilot_size',32),('pilot_steps',200),('formal_steps',1000),('max_new_tokens',512)]:
        settings.setdefault(key,default)
        if type(settings[key]) is not int or settings[key]<=0: raise ValueError('positive experimental budgets required')
    if settings['pilot_steps']>200: raise ValueError('pilot budget exceeds authorized 200-step cap')
    base=LatentMemoryConfig.from_dict(json.loads(Path(settings['base_config']).read_text(encoding='utf-8-sig')))
    base.fixed_instruction=INSTRUCTION; base.max_new_tokens=settings['max_new_tokens']
    return settings,base


def prepare(records, settings, output):
    panel=make_panel(records); candidates=[e for e in panel['fit'] if e['seconds'] in (4,8,16)]
    pilot=choose_pilot(candidates,settings['pilot_size'],42)
    controls=donor_map(pilot)
    protocol=dict(schema=1,pilot_seed=42,pilot_size=settings['pilot_size'],pilot_steps=settings['pilot_steps'],
                  fixed_instruction=INSTRUCTION,settings=settings,
                  pilot_criterion='all 32 fit sequences and ongoing states exact; correct EditScore exceeds both same-shape controls; all text diagnostic states exact',
                  no_automatic_budget_extension=True,subject7_signal_values_used_for_training=False,
                  primary_g2='E4(R2)-E3(R2), same checkpoint',
                  cohorts={c:{str(s):sum(e['seconds']==s for e in es) for s in (4,8,16,32)} for c,es in panel.items()})
    write_json(output/'protocol.json',protocol); write_json(output/'panel.json',panel)
    write_json(output/'pilot-panel.json',pilot); write_json(output/'donors.json',controls)
    write_json(output/'natural-order-pairs.json',{c:natural_order_pairs(es) for c,es in panel.items()})
    return panel,pilot,controls


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['prepare','pilot','align','latent','g2'])
    parser.add_argument('--config',required=True); parser.add_argument('--output',required=True)
    parser.add_argument('--device',choices=['cpu','cuda'],default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed',type=int,choices=[42,43,44],default=42)
    parser.add_argument('--checkpoint'); parser.add_argument('--g0-result'); parser.add_argument('--formal-protocol')
    args=parser.parse_args(argv)
    if args.mode in ('align','latent','g2'):
        if not args.formal_protocol: parser.error('formal stages require a locked --formal-protocol with minimum useful thresholds')
        formal=json.loads(Path(args.formal_protocol).read_text())
        thresholds=('minimum_g0_edit_score','minimum_g0_state_em','minimum_radar_control_gap','minimum_g2_edit_gain')
        if formal.get('locked') is not True or formal.get('seeds')!=[42,43,44] or any(
            type(formal.get(k)) not in (int,float) or not 0<formal[k]<=1 for k in thresholds):
            parser.error('formal protocol must lock all positive minimum-useful thresholds and seeds')
        if not args.g0_result: parser.error('formal stages require a passed --g0-result')
        if not json.loads(Path(args.g0_result).read_text())['passed']: parser.error('G0 has not passed; G2/formal stage prohibited')
        prior_source=Path(args.g0_result).parent/'source.json'
        if formal.get('g0_result_sha256')!=sha256(args.g0_result) or formal.get('g0_source_sha256')!=sha256(prior_source):
            parser.error('formal protocol is not bound to this G0 evidence')
    if args.mode in ('latent','g2') and not args.checkpoint: parser.error('stage requires --checkpoint')
    settings,base=_load_config(args.config); base.seed=args.seed
    if args.mode=='pilot' and args.seed!=42: parser.error('pilot uses only the locked seed42')
    output=Path(args.output); output.mkdir(parents=True,exist_ok=False)
    try:
        write_json(output/'status.json',dict(status='running',phase='verify_sources'))
        records=load_records(base.manifest_path,base.manifest_sha256)
        panel,pilot,donors=prepare(records,settings,output)
        if args.mode=='prepare':
            write_json(output/'status.json',dict(status='complete',mode='prepare')); return
        source=dict(base=source_manifest(base,records),experiment_settings=settings,
                    protocol_sha256=sha256(output/'protocol.json'),panel_sha256=sha256(output/'panel.json'))
        if args.mode in ('align','latent','g2'):
            prior=json.loads(prior_source.read_text())
            def without_seed(value):
                value=json.loads(json.dumps(value)); value['base']['config'].pop('seed',None); return value
            if without_seed(prior)!=without_seed(source): raise ValueError('G0 and formal experiment source mismatch')
            source['formal_protocol_sha256']=sha256(args.formal_protocol)
        write_json(output/'source.json',source)
        model,tokenizer,prompt=build_model(base,args.device); grammar=DescriptionGrammar(tokenizer)
        if args.device=='cuda': torch.cuda.reset_peak_memory_stats()
        if args.mode=='pilot':
            examples=prepare_examples(pilot)
            oracle_rows,oracle_summary=text_oracle(model,examples,tokenizer,grammar,INSTRUCTION,output/'text-oracle.jsonl',base.max_new_tokens)
            write_json(output/'text-oracle-summary.json',oracle_summary)
            _,before=evaluate(model,examples,prompt,tokenizer,grammar,output/'before-correct.jsonl',max_new_tokens=base.max_new_tokens)
            checkpoint=train_stage(model,examples,prompt,grammar,output/'alignment-pilot',steps=settings['pilot_steps'],
                                   phase='alignment',source=source,seed=42,learning_rate=base.learning_rate)
            # Verify real serialization after clearing every newly introduced tensor.
            saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
            if saved['source']['source']!=source: raise ValueError('pilot checkpoint provenance mismatch')
            with torch.no_grad():
                for name,p in model.named_parameters():
                    if not name.startswith(('backend.','local_adapter.')): p.zero_()
            load_stage(checkpoint,model,saved['source'])
            correct_rows,correct=evaluate(model,examples,prompt,tokenizer,grammar,output/'after-correct.jsonl',max_new_tokens=base.max_new_tokens)
            _,null=evaluate(model,examples,prompt,tokenizer,grammar,output/'after-zero.jsonl',control='zero',max_new_tokens=base.max_new_tokens)
            _,swapped=evaluate(model,examples,prompt,tokenizer,grammar,output/'after-swapped.jsonl',control='swapped',donors=donors,max_new_tokens=base.max_new_tokens)
            exact=oracle_pass(oracle_rows)
            # Gate fits each window equally; macro scores are separately reported by recording.
            window_summary=lambda rows:{k:sum(r['metrics'][k] for r in rows)/len(rows) for k in ('sequence_em','state_em','edit_score')}
            zero_rows=[json.loads(x) for x in (output/'after-zero.jsonl').read_text().splitlines()]
            swap_rows=[json.loads(x) for x in (output/'after-swapped.jsonl').read_text().splitlines()]
            gate=pilot_gate(exact,window_summary(correct_rows),window_summary(zero_rows),window_summary(swap_rows),settings['pilot_size'],len(correct_rows))
            # Phase freezing changes requires_grad, so compare only base/local parameters explicitly.
            write_json(output/'result.json',dict(**gate,before=before,correct=correct,zero=null,swapped=swapped,
                oracle=oracle_summary,checkpoint=str(checkpoint),successful_updates=settings['pilot_steps'],
                peak_cuda_bytes=torch.cuda.max_memory_allocated() if args.device=='cuda' else 0,
                scope='32-window fit diagnostic, not subject7 or G2 efficacy',formal_started=False,g2_started=False))
            write_json(output/'status.json',dict(status='diagnostic_passed' if gate['passed'] else 'stopped_g0',successful_updates=settings['pilot_steps'],g2_started=False))
            return gate
        if args.mode in ('align','latent'):
            if args.checkpoint:
                saved=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
                if saved['phase']!='alignment' or saved['step']!=settings['formal_steps']: raise ValueError('latent stage needs final G0 alignment checkpoint')
                if saved['source']['source']!=source: raise ValueError('alignment checkpoint source mismatch')
                load_stage(args.checkpoint,model,saved['source'])
                source={**source,'alignment_checkpoint_sha256':sha256(args.checkpoint)}
            examples=prepare_examples([e for e in panel['fit'] if e['seconds'] in (4,8,16)])
            train_stage(model,examples,prompt,grammar,output/'training',steps=settings['formal_steps'],
                        phase='alignment' if args.mode=='align' else 'latent',source=source,seed=args.seed)
            write_json(output/'status.json',dict(status='complete',mode=args.mode)); return
        saved=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
        prior=saved['source']['source']
        if saved['phase']!='latent' or saved['step']!=settings['formal_steps'] or any(prior.get(k)!=v for k,v in source.items()):
            raise ValueError('G2 requires final shared latent checkpoint with identical provenance')
        load_stage(args.checkpoint,model,saved['source']); summary={}
        for view in ('raw','compressed'):
            examples=prepare_examples([e for e in panel['validation'] if e['seconds']==16],view)
            conditions=[('E0',0),('E1',0),('E2',0),('E3',1),('E4',1),('E3',2),('E4',2),('E3',3),('E4',3)] if view=='raw' else [('E1',0),('E3',2),('E4',2)]
            first_hashes={}
            for arm,rounds in conditions:
                key=f'{view}-{arm}-r{rounds}'
                rows,metrics=evaluate(model,examples,prompt,tokenizer,grammar,output/(key+'.jsonl'),arm=arm,rounds=rounds,view=view,max_new_tokens=base.max_new_tokens)
                summary[key]=metrics
                if rounds==1:
                    hashes={r['uid']:r['prefix_sha256'] for r in rows}
                    if arm=='E3': first_hashes=hashes
                    elif hashes!=first_hashes: raise RuntimeError('E3/E4 first-round identity failed')
        write_json(output/'summary.json',summary); write_json(output/'status.json',dict(status='complete',mode='g2',seed=args.seed))
    except BaseException as error:
        write_json(output/'status.json',dict(status='failed',error=repr(error))); raise


if __name__=='__main__': main()
