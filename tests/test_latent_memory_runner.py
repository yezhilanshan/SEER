"""Runs the real CLI against local tiny Qwen files and complete synthetic recordings."""
import hashlib
import json
import numpy as np
import pytest
import torch


def test_cli_prepare_probe_train_and_replay_are_isolated(tmp_path):
    from transformers import PreTrainedTokenizerFast
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from test_latent_memory_model import make_model
    from continuous_mmwave.latent_memory.runner import main
    base=tmp_path/'qwen'; model=make_model(); model.backend.lm.save_pretrained(base)
    tokenizer=Tokenizer(WordLevel({'<pad>':0,'<bos>':1,'<eos>':2,'<unk>':3,'The':4,'person':5,'stood':6,'up':7},unk_token='<unk>'))
    tokenizer.pre_tokenizer=Whitespace()
    fast=PreTrainedTokenizerFast(tokenizer_object=tokenizer,pad_token='<pad>',bos_token='<bos>',eos_token='<eos>',unk_token='<unk>')
    fast.chat_template="{% for message in messages %}{{ message['content'] }}{% endfor %}"
    fast.save_pretrained(base)
    records=[]
    for subject,role in [('subject1','train'),('subject7','validation')]:
        cells=tmp_path/f'{subject}.npy'; np.save(cells,np.ones((7,4,64),np.float32))
        records.append(dict(file=f'{subject}_01',subject=subject,role=role,frames=7,cells_path=str(cells),
                            artifacts={'cells.npy':{'sha256':hashlib.sha256(cells.read_bytes()).hexdigest()}},
                            instances=[dict(start=0,stop=6,label=0)]))
    manifest=tmp_path/'manifest.json'; manifest.write_text(json.dumps(records))
    config=dict(llm_path=str(base),manifest_path=str(manifest),manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
                adapter_checkpoint=None,adapter_sha256=None,query_frames=3,capacities=[2,2,2],group_size=2,
                lmt_counts=[2,2,2],max_new_tokens=4,gradient_checkpointing=False)
    path=tmp_path/'config.json'; path.write_text(json.dumps(config))
    def run(mode,folder,*extra):
        main([mode,'--config',str(path),'--output',str(tmp_path/folder),'--device','cpu',*extra])
    run('prepare','prepare')
    assert json.loads((tmp_path/'prepare'/'summary.json').read_text())['frames']==14
    run('probe','probe')
    report=json.loads((tmp_path/'probe'/'probe.json').read_text())
    assert report['input_gradient_nonzero'] and report['frozen_unchanged'] and report['group_isolation_passed']
    with pytest.raises(SystemExit): run('train','implicit')
    assert not (tmp_path/'implicit').exists()
    run('train','train','--steps','1')
    run('replay','replay','--checkpoint',str(tmp_path/'train'/'step-000001.pt'),'--max-queries','1')
    rows=[json.loads(x) for x in (tmp_path/'replay'/'predictions.jsonl').read_text().splitlines()]
    assert len(rows)==1 and rows[0]['file']=='subject7_01'
    assert rows[0]['cutoff_frame']==3 and rows[0]['reference']['sequence']==[0]
    assert all(s['end_frame']<=3 for r in rows[0]['trace'] for s in r['selected'])
    assert json.loads((tmp_path/'replay'/'summary.json').read_text())['checkpoint_step']==1
    from continuous_mmwave.latent_memory.verification import verify_run
    assert verify_run(tmp_path/'train')['passed']
    assert verify_run(tmp_path/'replay')['passed']
    rows[0]['trace'][0]['selected'][0]['end_frame']=999
    (tmp_path/'replay'/'predictions.jsonl').write_text(json.dumps(rows[0])+'\n')
    with pytest.raises(ValueError,match='future|candidate|trace'):
        verify_run(tmp_path/'replay')
