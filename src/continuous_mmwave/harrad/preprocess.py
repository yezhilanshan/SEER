"""Read-only official-equivalent FFT conversion into separate immutable feature arrays."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np

from .data import read_instances,subject_role,frame_annotations,fit_normalizers,CLASSES


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''): h.update(block)
    return h.hexdigest()


def write_json(path, data):
    path=Path(path); temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(data,indent=2,ensure_ascii=False),encoding='utf-8'); temp.replace(path)


_RANGE_WINDOW=np.hanning(255)
_DOPPLER_WINDOW=np.hanning(256)
_RANGES=np.arange(512)/1024*2e6*3e8/(2*1171875.0e7)
_MIN=int(np.argmin(np.abs(_RANGES-.5))); _MAX=int(np.argmin(np.abs(_RANGES-4.5)))


def frame_features(raw: np.ndarray) -> tuple[np.ndarray,np.ndarray]:
    if raw.shape!=(256,256) or not np.isfinite(raw).all(): raise ValueError('raw frame must be finite [256,256]')
    data=raw.reshape(256,256).T[1:]*_RANGE_WINDOW[:,None]
    padded=np.zeros((1024,256)); padded[(1024-255)//2:(1024-255)//2+255]=data
    ranges=np.fft.fft(padded,1024,axis=0)/_RANGE_WINDOW.sum()*(2./2048.)
    ranges=ranges[_MIN:_MAX]*_DOPPLER_WINDOW[None,:]
    amplitude=np.abs(np.fft.fftshift(np.fft.fft(ranges,256,axis=1)/_DOPPLER_WINDOW.sum(),axes=1))
    db=20*np.log10(np.maximum(amplitude,1e-12))
    md=db.sum(axis=0).astype(np.float32)
    rd=db.reshape(40,4,64,4).mean(axis=(1,3)).astype(np.float32)
    return md,rd


def convert_record(record: dict, output_root: str) -> dict:
    if record['role'] not in ('train','validation'): raise ValueError('test signal conversion forbidden')
    import h5py
    source=Path(record['source']); before=source.stat(); length=record['frames']
    if 'source_stat' in record:
        expected=record['source_stat']
        if before.st_size!=expected['bytes'] or before.st_mtime_ns!=expected['mtime_ns']:
            raise ValueError(f'source changed since audit: {source}')
    dest=Path(output_root)/source.stem; dest.mkdir(parents=True,exist_ok=False)
    started=time.time()
    md=np.lib.format.open_memmap(dest/'md.npy',mode='w+',dtype=np.float32,shape=(length,256))
    rd=np.lib.format.open_memmap(dest/'rd.npy',mode='w+',dtype=np.float32,shape=(length,40,64))
    fingerprint=hashlib.sha256()
    with h5py.File(source,'r') as f:
        if f['radar'].shape!=(length,256,256) or f['radar'].dtype!=np.dtype('float32'):
            raise ValueError('unexpected raw shape/dtype')
        for i in range(length):
            frame=f['radar'][i]; fingerprint.update(frame.tobytes())
            md[i],rd[i]=frame_features(frame)
            if (i+1)%256==0:
                write_json(dest/'progress.json',{'frames':i+1,'total':length,'seconds':time.time()-started})
    md.flush(); rd.flush(); del md,rd
    labels,ids=frame_annotations(record['instances'],record['file'],length)
    for name,arr in [('labels',labels),('instance_id',ids),('timestamps',np.arange(length,dtype=np.float64)/15)]:
        np.save(dest/(name+'.npy'),arr)
    after=source.stat()
    if before.st_size!=after.st_size or before.st_mtime_ns!=after.st_mtime_ns: raise RuntimeError('raw source changed during read')
    artifacts={p.name:{'bytes':p.stat().st_size,'sha256':sha256(p)} for p in sorted(dest.glob('*.npy'))}
    result={**record,'features':{k:str(dest/(k+'.npy')) for k in ('md','rd')},'labels_path':str(dest/'labels.npy'),
            'instance_path':str(dest/'instance_id.npy'),'timestamps_path':str(dest/'timestamps.npy'),
            'artifact_dir':str(dest),'artifacts':artifacts,'raw_array_stream_sha256':fingerprint.hexdigest(),
            'source_stat':{'bytes':before.st_size,'mtime_ns':before.st_mtime_ns},'seconds':time.time()-started}
    write_json(dest/'complete.json',result)
    return result


def verify_records(records: list[dict]) -> dict:
    frames=0; instances=0
    for r in records:
        if r['role'] not in ('train','validation'): raise ValueError('test canonical record forbidden')
        for name,a in r['artifacts'].items():
            p=Path(r['artifact_dir'])/name
            if p.stat().st_size!=a['bytes'] or sha256(p)!=a['sha256']: raise ValueError('canonical hash mismatch')
        labels,ids=frame_annotations(r['instances'],r['file'],r['frames'])
        np.testing.assert_array_equal(labels,np.load(r['labels_path']))
        np.testing.assert_array_equal(ids,np.load(r['instance_path']))
        np.testing.assert_array_equal(np.load(r['timestamps_path']),np.arange(r['frames'])/15)
        for k,shape in [('md',(r['frames'],256)),('rd',(r['frames'],40,64))]:
            x=np.load(r['features'][k],mmap_mode='r')
            if x.shape!=shape: raise ValueError('canonical feature shape mismatch')
            for s in range(0,len(x),1024):
                if not np.isfinite(x[s:s+1024]).all(): raise ValueError('nonfinite canonical feature')
        if Path(r['source']).stat().st_mtime_ns!=r['source_stat']['mtime_ns']: raise ValueError('raw source modified')
        frames+=r['frames']; instances+=len(r['instances'])
    return {'passed':True,'recordings':len(records),'frames':frames,'instances':instances,'test_signal_values_read':False}


def prepare(config: dict, output_root: Path) -> tuple[list[dict],dict]:
    source=Path(config['data']['source']); audit=json.loads(Path(config['data']['source_audit']).read_text())
    lengths={r['file']:r['datasets']['radar']['shape'][0] for r in audit['hdf5']}
    for name,meta in audit['annotations'].items():
        if name.endswith('.csv') and sha256(source/'labels'/name)!=meta['sha256']: raise ValueError('annotation source changed')
    instances=read_instances(source/'labels'/'activities.csv',lengths)
    records=[]; test=[]
    for src in audit['hdf5']:
        file=src['file']; subject=file.split('_')[0]; role=subject_role(subject,config['data']['splits'])
        row={'file':file,'source':str(source/file),'subject':subject,'role':role,'frames':lengths[file],
             'source_stat':{'bytes':src['bytes'],'mtime_ns':src['mtime_ns']},'instances':[r for r in instances if r['file']==file]}
        (test if role=='test' else records).append(row)
    write_json(output_root/'input-lock.json',{'config':config,'classes':CLASSES,'instances':len(instances),
              'records':records,'test_metadata_only':test,'activities_sha256':sha256(source/'labels'/'activities.csv')})
    write_json(output_root/'test-metadata-only.json',test)
    counts={role:sum(len(r['instances']) for r in records+test if r['role']==role) for role in ('train','validation','test')}
    expected=config['data']['expected_instances']
    if counts!=expected: raise ValueError(f'split counts differ: {counts}')
    converted=[]
    with ProcessPoolExecutor(max_workers=config['data']['conversion_workers']) as pool:
        pending=[pool.submit(convert_record,r,str(output_root/'canonical')) for r in records]
        for future in as_completed(pending):
            record=future.result(); converted.append(record)
            write_json(output_root/'status.json',{'phase':'conversion','completed_recordings':len(converted),'total_recordings':len(records),'completed_frames':sum(r['frames'] for r in converted),'total_frames':sum(r['frames'] for r in records),'last_recording':record['file'],'updated_unix':time.time()})
            print(f"converted {len(converted)}/{len(records)} {record['file']}",flush=True)
    converted.sort(key=lambda r:r['file'])
    write_json(output_root/'canonical-verification.json',verify_records(converted))
    normalizers=fit_normalizers(converted)
    np.savez(output_root/'normalizers.npz',**normalizers)
    write_json(output_root/'canonical-manifest.json',converted)
    return converted,normalizers
