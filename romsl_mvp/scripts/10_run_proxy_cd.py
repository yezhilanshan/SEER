import argparse
import os
from pathlib import Path
import sys
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from romsl.proxy_experiment import prepare_proxy,run_proxy

parser=argparse.ArgumentParser()
parser.add_argument('--job',type=Path,required=True)
parser.add_argument('--data-root',type=Path)
parser.add_argument('--seed',type=int,choices=[42,43,44],default=42)
parser.add_argument('--prepare',action='store_true')
args=parser.parse_args()
if args.prepare:
    if args.data_root is None:
        parser.error('--prepare requires --data-root')
    print(prepare_proxy(args.job,args.data_root),flush=True)
else:
    torch.cuda.set_per_process_memory_fraction(.12)
    run_proxy(args.job,torch.device('cuda:0'),args.seed)
