from pathlib import Path
import yaml


def test_formal_locked_source_digests_are_valid_sha256():
    config=yaml.safe_load(Path('configs/parrad-v25-training-pilot-v1.yaml').read_text())
    for key,value in config.items():
        if key.endswith('_sha256'):
            assert len(value)==64 and all(c in '0123456789abcdef' for c in value),key
