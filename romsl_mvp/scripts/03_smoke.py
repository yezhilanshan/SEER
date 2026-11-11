import argparse
import json
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from romsl.pipeline import smoke

parser = argparse.ArgumentParser()
parser.add_argument('--job', type=Path, required=True)
parser.add_argument('--exp', choices=['A', 'B'], required=True)
args = parser.parse_args()
print(json.dumps(smoke(args.job, args.exp), indent=2), flush=True)
