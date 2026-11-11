"""Validate complete saved prediction cohorts independently of score summaries."""
from continuous_mmwave.parrad_stage4_qa import grade_answer
from continuous_mmwave.parrad_event_boundary_qa import development_gates


def assert_same_evaluation(previous,current,*,names=('temporal_query','text_oracle_query','temporal_context','evaluate','new_model','oracle_gate')):
    import ast
    from pathlib import Path
    def functions(path):
        return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(Path(path).read_text()).body
            if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}
    old,new=functions(previous),functions(current)
    if any(name not in old or name not in new or old[name]!=new[name] for name in names):
        raise ValueError('baseline evaluation code changed')


def assert_same_baseline_source(previous,current):
    keys=('provenance','protected_sha256','frozen_upstream_digest','task_sha256','batch_sha256','base_qa_updates','multi_reader')
    if any(previous.get(k)!=current.get(k) for k in keys): raise ValueError('baseline source changed')


def restore_baseline(previous,output,settings,tasks,events,source):
    import json
    import shutil
    from pathlib import Path
    from continuous_mmwave.parrad_stage4_runner import sha
    from continuous_mmwave.parrad_event_boundary_tasks import score_boundary
    from continuous_mmwave.parrad_temporal_qa import oracle_gate
    previous,output=Path(previous),Path(output)
    status=json.loads((previous/'status.json').read_text())
    if status['status'] not in ('failed','complete') or status['phase'] not in ('train-evaluate','complete'):
        raise ValueError('previous run did not complete all baselines')
    strip=lambda s:{k:v for k,v in s.items() if k not in ('output_dir','baseline_source')}
    if strip(status['settings'])!=strip(settings): raise ValueError('baseline settings changed')
    old_source=json.loads((previous/'source.json').read_text()); assert_same_baseline_source(old_source,source)
    old_code=Path(json.loads((previous.with_name(previous.name+'.logs')/'launcher.json').read_text())['code'])/'src/continuous_mmwave'
    code=Path(__file__).parent
    for name,digest in status['code_sha256'].items():
        if sha(old_code/name)!=digest: raise ValueError('previous baseline code changed')
        if name not in ('parrad_temporal_qa.py','parrad_qa_readback.py') and sha(code/name)!=digest:
            raise ValueError('baseline dependency changed')
    assert_same_evaluation(old_code/'parrad_temporal_qa.py',code/'parrad_temporal_qa.py')
    results={}; members={}
    specs={'fit-text-preflight.json':('fit','text',True),'dev-text-oracle.json':('dev','text',True),
        'dev-before.json':('dev','raw',False),'dev-original-reader.json':('dev','raw',True),'dev-none.json':('dev','none',False)}
    for name,(cohort,mode,multi) in specs.items():
        selected=[t for t in tasks['tasks'] if t['cohort']==cohort and (not multi or t['size']>1)]
        result=json.loads((previous/name).read_text())
        validate_rows(result,selected,events,mode,'none',new_reader=True,original_multi=name=='dev-original-reader.json')
        if result['metrics']!=score_boundary(result['rows']): raise ValueError('baseline metrics mismatch')
        results[name]=result; members[name]=sha(previous/name)
    gate=json.loads((previous/'text-preflight-gates.json').read_text())
    if gate['gates']!=oracle_gate(results['fit-text-preflight.json']['metrics']) or not gate['passed'] or not all(gate['gates'].values()):
        raise ValueError('baseline text gate failed')
    for name in ('text-preflight-gates.json','retrieval-before.json'): members[name]=sha(previous/name)
    for name in members:
        if (output/name).exists(): raise FileExistsError('refuse overwrite reused baseline')
        shutil.copy2(previous/name,output/name)
    return results['dev-before.json'],results['dev-none.json'],json.loads((previous/'retrieval-before.json').read_text()),dict(
        previous_output=str(previous),previous_code=str(old_code),previous_status=status['status'],
        previous_status_sha256=sha(previous/'status.json'),previous_source_sha256=sha(previous/'source.json'),artifact_sha256=members,
        generation_code_equal=True,frozen_inputs_equal=True)


def validate_rows(result,tasks,events,mode,arm,*,swaps=None,query_labels=None,new_reader=False,original_multi=False):
    rows=result['rows']
    if (result['mode']!=mode or result['arm']!=arm
        or [r['uid'] for r in rows]!=[t['uid'] for t in tasks]
        or len({r['uid'] for r in rows})!=len(rows)):
        raise ValueError('incomplete/duplicate/misbound evaluation cohort')
    for row,task in zip(rows,tasks):
        fields=('kind','size','label','position','pair_id','source_ids','query')
        if any(row.get(k)!=task.get(k) for k in fields) or row['reference']!=task['answer']:
            raise ValueError('saved task metadata mismatch')
        if new_reader and row.get('cohort')!=task['cohort']: raise ValueError('saved task cohort mismatch')
        expected_sources=swaps[task['uid']]['source_ids'] if mode=='swapped' else task['source_ids']
        if row['actual_source_ids']!=expected_sources: raise ValueError('saved actual evidence mismatch')
        if row['mode']!=mode or row['arm']!=arm: raise ValueError('saved mode/arm mismatch')
        expected_tokens=0 if mode in ('text','none') else 8*task['size']
        if row['context_tokens']!=expected_tokens: raise ValueError('saved context token count mismatch')
        if new_reader:
            expected_reader='original' if task['size']==1 or original_multi else 'text-qa-500'
            if row.get('reader')!=expected_reader: raise ValueError('saved QA reader mismatch')
        kind='summary' if task['kind']=='sequence' else 'presence' if task['kind']=='presence' else 'first'
        if row['correct']!=grade_answer(row['generated'],task['answer'],kind):
            raise ValueError('saved answer grade mismatch')
        if mode=='swapped':
            if task['kind']!='presence' or query_labels is None or task['pair_id'] not in query_labels:
                raise ValueError('missing queried activity for swapped evidence')
            activity=query_labels[task['pair_id']]
            original='Yes' if activity in [events[u]['label'] for u in task['source_ids']] else 'No'
            actual='Yes' if activity in [events[u]['label'] for u in expected_sources] else 'No'
            if task['answer']!=original or row['evidence_reference']!=actual:
                raise ValueError('swapped answer does not match actual evidence')
        if mode=='text':
            from continuous_mmwave.parrad_temporal_qa import text_oracle_query
            if row['oracle_query']!=text_oracle_query(task,events): raise ValueError('saved oracle prompt mismatch')


def validate_gates(before,after,native):
    gates=development_gates(before,after)
    if native['gates']!=gates or native['development_pass']!=all(gates.values()):
        raise ValueError('saved development gate mismatch')
    return gates
