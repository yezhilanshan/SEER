import argparse
import json
import os
from pathlib import Path
import sys

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
import torch
from romsl.experiment import run_all

parser = argparse.ArgumentParser()
parser.add_argument('--job', type=Path, required=True)
parser.add_argument('--data-root', type=Path, required=True)
parser.add_argument('--device', default='cuda:0')
parser.add_argument('--seed', type=int, choices=[42,43,44], default=42)
parser.add_argument('--reuse-test', action='store_true')
args = parser.parse_args()
if args.device.startswith('cuda'):
    torch.cuda.set_per_process_memory_fraction(.12)
settings = {'pretrain_epochs': 30, 'probe_epochs': 50, 'seed': args.seed, 'early_stop_patience': 5,
            'lr': .001, 'weight_decay': .0001, 'amp': False, 'pretrain_batch_size': 32,
            'probe_batch_size': 64, 'num_workers': 0,
            'worker_reason': 'Bound host memory usage on shared server.', 'reuse_test': args.reuse_test}
run_all(args.job, args.data_root, torch.device(args.device), settings)
