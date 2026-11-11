"""Independent Numpy readback of Stage 1 predictions and frame contracts."""
import json
from pathlib import Path
import sys

import numpy as np


def verify(root):
    root=Path(root)
    jobs=json.loads((root/'jobs.json').read_text())
    frames={r['file']:r['frames'] for r in json.loads((root/'cache-manifest.json').read_text())}
    count=0
    for job in jobs:
        directory=root/f"{job['kind']}-seed{job['seed']}"
        if job['kind']!='persistence':
            assert json.loads((directory/'checkpoint-replay.json').read_text())['maximum_absolute_error']==0
        for row in job['records']:
            with np.load(directory/(Path(row['file']).stem+'.npz'),allow_pickle=False) as data:
                assert np.array_equal(data['target_cutoff'],data['observed_cutoff'][:,None]+[3,9,15])
                assert np.all(data['target_cutoff']<=frames[row['file']])
                error=(data['mean'].astype('float64')-data['target'])**2
                nll=.5*(np.log(2*np.pi)+data['logvar']+error*np.exp(-data['logvar'].astype('float64')))
                for h, metric in enumerate(row['horizons']):
                    np.testing.assert_allclose(error[:,h].mean(),metric['mse'],rtol=1e-5,atol=1e-6)
                    np.testing.assert_allclose(nll[:,h].mean(),metric['nll'],rtol=1e-5,atol=1e-6)
                    for g,name in enumerate(('steady','transition','repeated','unknown')):
                        mask=data['groups'][:,h,g]
                        assert int(mask.sum())==metric['groups'][name]['n']
                        if mask.any():
                            np.testing.assert_allclose(error[mask,h].mean(),metric['groups'][name]['mse'],rtol=1e-5,atol=1e-6)
                    count+=1
    result=dict(passed=True,jobs=len(jobs),record_horizon_checks=count,
                checks='frame offsets and bounds; fresh-instance checkpoint replay; NLL/MSE; group support and MSE')
    (root/'independent-verification.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result))


if __name__=='__main__':
    verify(sys.argv[1])
