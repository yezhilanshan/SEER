"""Single isolated, resumable diagnostic worker; never changes the original pilot."""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pilot', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    root, code = Path(args.output).resolve(), Path(__file__).resolve().parents[1]
    root.mkdir(parents=True, exist_ok=False)
    env = {**os.environ,'PYTHONPATH':str(code/'src'),'OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4',
           'HF_HUB_OFFLINE':'1','PYTHONUNBUFFERED':'1','TOKENIZERS_PARALLELISM':'false'}
    def sha(path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    metadata = dict(state='starting', started=datetime.now().astimezone().isoformat(),pid=os.getpid(),
                    code=str(code),pilot=str(Path(args.pilot).resolve()),output=str(root),completed=[],
                    checkpoint_sha256={stage:sha(Path(args.pilot)/stage/'best.pt') for stage in ('stage1','align')},
                    source_sha256=sha(code/'source-manifest.json'),base_llm_frozen=True)
    def status(state,**extra):
        metadata.update(state=state,**extra)
        tmp = root/'status.json.tmp'
        tmp.write_text(json.dumps(metadata,indent=2))
        tmp.replace(root/'status.json')
        print(json.dumps(metadata),flush=True)
    try:
        for step in ('extract','probes','smoothing','misses','prior','perturbations'):
            module = 'diagnostic_controls' if step in ('prior','perturbations') else 'diagnostic_runner'
            command=[sys.executable,'-u','-m','continuous_mmwave.radar_text.'+module,step,
                     '--pilot',args.pilot,'--output',str(root)]
            with (root/(step+'.log')).open('xb') as log:
                child=subprocess.Popen(command,cwd=code,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
                status('running',active_stage=step,child_pid=child.pid)
                exitcode=child.wait()
            if exitcode:
                raise RuntimeError(f'{step} failed with exit code {exitcode}')
            metadata['completed'].append(step)
        assert all(sha(Path(args.pilot)/stage/'best.pt')==digest for stage,digest in metadata['checkpoint_sha256'].items())
        status('complete',finished=datetime.now().astimezone().isoformat())
    except BaseException as error:
        status('failed',error=str(error),finished=datetime.now().astimezone().isoformat())
        raise


if __name__=='__main__':
    main()
