import json
import numpy as np
import pytest
from continuous_mmwave.harrad.preprocess import sha256


def test_prepare_only_locks_panels_without_loading_llm(tmp_path):
    from continuous_mmwave.latent_memory.experiments.runner import main
    from continuous_mmwave.latent_memory.config import LatentMemoryConfig
    rows=[]
    for subject,suffix,role in [('subject1','01','train'),('subject1','02','train'),('subject7','01','validation')]:
        path=tmp_path/f'{subject}_{suffix}.npy'; np.save(path,np.ones((1920,4,64),np.float32))
        rows.append(dict(file=f'{subject}_{suffix}',subject=subject,role=role,frames=1920,cells_path=str(path),
            artifacts={'cells.npy':{'sha256':sha256(path)}},instances=[dict(start=i*120,stop=(i+1)*120,label=y)
                for i,y in enumerate([0,1,1,0,2,2,3,4,5,6,7,8,9,10,11,0])]))
    manifest=tmp_path/'manifest.json'; manifest.write_text(json.dumps(rows))
    base=LatentMemoryConfig(manifest_path=str(manifest),manifest_sha256=sha256(manifest),llm_path='missing-test-model',adapter_checkpoint=None,adapter_sha256=None)
    basepath=tmp_path/'base.json'; basepath.write_text(json.dumps(base.to_dict()))
    config=tmp_path/'experiment.json'; config.write_text(json.dumps({'base_config':str(basepath),'pilot_size':32,'pilot_steps':200}))
    output=tmp_path/'prepared'; main(['prepare','--config',str(config),'--output',str(output)])
    assert len(json.loads((output/'pilot-panel.json').read_text()))==32
    protocol=json.loads((output/'protocol.json').read_text())
    assert protocol['subject7_signal_values_used_for_training']==False


def test_formal_modes_require_a_locked_preregistration_before_loading_data(tmp_path):
    from continuous_mmwave.latent_memory.experiments.runner import main
    result=tmp_path/'result.json'; result.write_text(json.dumps({'passed':True}))
    with pytest.raises(SystemExit):
        main(['align','--config','nonexistent','--output',str(tmp_path/'out'),'--g0-result',str(result)])
    assert not (tmp_path/'out').exists()
