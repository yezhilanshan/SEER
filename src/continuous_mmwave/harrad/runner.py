"""Durable HARrad prepare -> verify -> overfit gates -> sequential formal runs."""
from __future__ import annotations

import argparse
import ast
import importlib.metadata
import json
from pathlib import Path
import time
import traceback
import numpy as np
import torch

from .data import clip_rows
from .preprocess import frame_features,prepare,write_json,sha256
from .training import overfit_gate,train_run


def official_equivalence(config,output):
    import h5py
    source=Path(config['data']['source'])
    reference=source.parent/'format_probe_20261016_v1'
    report=json.loads((reference/'probe.json').read_text())
    script=source/'process.py'
    if sha256(script)!=report['script_sha256']: raise ValueError('official process source changed')
    tree=ast.parse(script.read_text()); fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='range_doppler')
    space={'np':np}; exec(compile(ast.Module(body=[fn],type_ignores=[]),str(script),'exec'),space)
    md_gap=rd_gap=reference_gap=0.; samples=0
    for row in report['rows']:
        if row['subject'] not in config['data']['splits']['train']: raise ValueError('official probe outside train')
        p=reference/row['artifact']
        if sha256(p)!=row['artifact_sha256']: raise ValueError('official probe artifact changed')
        with np.load(p,allow_pickle=False) as ref,h5py.File(source/row['file'],'r') as f:
            for i in sorted({row['start'],(row['start']+row['stop'])//2,row['stop']-1}):
                raw=f['radar'][i]; md,rd=frame_features(raw)
                db=20*np.log10(space['range_doppler'](raw))
                md_gap=max(md_gap,float(np.abs(md-db.sum(0).astype(np.float32)).max()))
                rd_gap=max(rd_gap,float(np.abs(rd-db.reshape(40,4,64,4).mean((1,3)).astype(np.float32)).max()))
                reference_gap=max(reference_gap,float(np.abs(md-ref['md'][i-row['start']]).max()))
                np.testing.assert_allclose(md,ref['md'][i-row['start']],atol=1e-4,rtol=1e-6)
                if i==(row['start']+row['stop'])//2:
                    np.testing.assert_allclose(rd,ref['rd_mid'].reshape(40,4,64,4).mean((1,3)),atol=1e-4,rtol=1e-6)
                samples+=1
    if md_gap>1e-4 or rd_gap>1e-4: raise RuntimeError('official FFT numerical mismatch')
    result={'passed':True,'samples':samples,'classes':len(report['rows']),'md_max_abs':md_gap,'rd_max_abs':rd_gap,
            'prior_probe_md_max_abs':reference_gap,'official_script_sha256':sha256(script),'test_signal_values_read':False}
    write_json(output/'official-equivalence.json',result)
    return result


def run(config,output):
    output=Path(output); output.mkdir(parents=True,exist_ok=False)
    started=time.time()
    try:
        if sha256(config['data']['source_audit'])!=config['data']['source_audit_sha256']: raise ValueError('source audit lock mismatch')
        write_json(output/'config.json',config)
        versions={k:importlib.metadata.version(k) for k in ('numpy','torch','h5py','pytest')}
        write_json(output/'environment.json',{'versions':versions,'cuda':torch.version.cuda,
                   'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,'started_unix':started})
        write_json(output/'status.json',{'phase':'official_equivalence','updated_unix':time.time()})
        official_equivalence(config,output)
        records,normalizers=prepare(config,output)
        results=[]; gates=[]
        for arm in ('md','rd'):
            write_json(output/'status.json',{'phase':'overfit_gate','arm':arm,'updated_unix':time.time()})
            norm={'mean':normalizers[arm+'_mean'],'scale':normalizers[arm+'_scale']}
            gates.append(overfit_gate(arm,clip_rows(records,arm),norm,config['training'],output/'gates'/arm))
        write_json(output/'gates-passed.json',gates)
        # Gate weights are never loaded into formal models; each seed starts afresh.
        for seed in config['training']['seeds']:
            for arm in ('md','rd'):
                write_json(output/'status.json',{'phase':'formal_training','arm':arm,'seed':seed,'completed_runs':len(results),'total_runs':6,'updated_unix':time.time()})
                norm={'mean':normalizers[arm+'_mean'],'scale':normalizers[arm+'_scale']}
                results.append(train_run(arm,seed,clip_rows(records,arm),norm,config['training'],output/'runs'/f'{arm}-seed{seed}'))
                write_json(output/'completed-runs.json',results)
        result={'phase':'complete','runs':results,'seconds':time.time()-started,'test_signal_values_read':False,'completed_unix':time.time()}
        write_json(output/'complete.json',result); write_json(output/'status.json',{'phase':'complete','completed_runs':len(results),'updated_unix':time.time()})
    except BaseException as exc:
        failure={'phase':'failed','error':repr(exc),'traceback':traceback.format_exc(),'elapsed_seconds':time.time()-started,'updated_unix':time.time()}
        write_json(output/'failed.json',failure); write_json(output/'status.json',failure)
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--config',required=True); parser.add_argument('--output',required=True)
    args=parser.parse_args(); run(json.loads(Path(args.config).read_text(encoding='utf-8')),args.output)
