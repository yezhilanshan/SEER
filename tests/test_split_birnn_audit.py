from __future__ import annotations
import importlib
import json
import numpy as np
import pytest


def test_hdf5_audit_preserves_all_md_frames_and_reports_clock_assumptions(tmp_path):
    h5py = pytest.importorskip('h5py')
    audit = importlib.import_module('continuous_mmwave.split_birnn.audit')
    source = tmp_path/'raw'/'subject_01'/'captured_data'/'set000'; source.mkdir(parents=True)
    with h5py.File(source/'processed_data.h5','w') as f:
        f.create_dataset('radar/TI_xWR14xx/mDoppler',data=np.arange(12*128).reshape(12,128).astype('float32'))
        f.create_dataset('radar/TI_xWR68xx/mDoppler',data=np.arange(11*128).reshape(11,128).astype('float32'))
    (source/'timestamp_speech.csv').write_text('timestamp,command\n2020-01-01T10:00:00,walk to bed\n2020-01-01T10:00:00.450,sit down on bed\n',encoding='utf-8')
    out = tmp_path/'out'
    result = audit.prepare_data(tmp_path/'raw',out,make_plots=False)
    assert result['session_count'] == 1 and result['recording_count'] == 2
    assert result['sensor_frames'] == 23
    assert result['hospital_status'] == 'unconfirmed'
    assert result['formal_training_ready'] is False
    rows = [json.loads(line) for line in (out/'manifest.jsonl').read_text().splitlines()]
    for row in rows:
        with np.load(out/row['path']) as arrays:
            assert len(arrays['md']) == row['frame_count']
            assert arrays['labels'].tolist() == [0]*5+[3]*(row['frame_count']-5)
        assert row['transition_count'] == 1
        assert row['alignment']['clock_calibrated'] is False
    with pytest.raises(FileExistsError):
        audit.prepare_data(tmp_path/'raw',out,make_plots=False)
