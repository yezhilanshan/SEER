import argparse
import json
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from romsl.pipeline import prepare

parser = argparse.ArgumentParser()
parser.add_argument('--data-root', type=Path, required=True)
parser.add_argument('--job', type=Path, required=True)
args = parser.parse_args()
config = json.loads((PROJECT / 'configs/mvp.yaml').read_text())
joints = json.loads((PROJECT / 'configs/joints.yaml').read_text())
print(prepare(args.data_root, args.job, config, joints), flush=True)
