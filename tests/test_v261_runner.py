import importlib
import pytest
import json
import yaml


def api():return importlib.import_module('continuous_mmwave.v261.runner')


def test_stage_dependencies_identity_and_one_shot(tmp_path):
    root=tmp_path/'experiment';identity={'source':'locked'}
    api().begin_stage(root,'audit',identity)
    with pytest.raises(ValueError,match='already|exists'):
        api().begin_stage(root,'audit',identity)
    with pytest.raises(ValueError,match='successful|requires'):
        api().begin_stage(root,'prepare',identity)
    api().finish_stage(root,'audit',identity,{'passed':True})
    with pytest.raises(ValueError,match='identity'):
        api().begin_stage(root,'prepare',{'source':'changed'})
    api().begin_stage(root,'prepare',identity)
    api().finish_stage(root,'prepare',identity,{'passed':True})
    with pytest.raises(ValueError,match='successful|requires'):
        api().begin_stage(root,'train',identity)
    api().begin_stage(root,'smoke',identity)
    # Failed and incomplete smoke cannot authorize train.
    with pytest.raises(ValueError,match='successful|requires'):
        api().begin_stage(root,'train',identity)


def test_protocol_arms_seeds_budget_cannot_be_reinterpreted():
    config=dict(arms=['M','A','B','AB'],seeds=[42,43,44],training=dict(steps=1000,max_attempts=1100,lr=1e-4,weight_decay=.01,clip_norm=1.))
    api().validate_protocol(config)
    config['seeds']=[42]
    with pytest.raises(ValueError,match='seed|protocol'):
        api().validate_protocol(config)


def test_frontend_yaml_and_non_tensor_qwen_assets_are_locked(tmp_path):
    from continuous_mmwave.v26.source import frontend_asset_lock
    assets=tmp_path/'qwen';assets.mkdir()
    (assets/'weights.bin').write_bytes(b'weights')
    metadata=assets/'config.json';metadata.write_text('{"rms_norm_eps":1e-6}')
    legacy=tmp_path/'old.yaml';legacy.write_text(yaml.safe_dump({'llm':{'name_or_path':str(assets)}}))
    current=tmp_path/'current.yaml';current.write_bytes(legacy.read_bytes())
    original=tmp_path/'original';original.mkdir()
    (original/'protocol.json').write_text(json.dumps({'frontend_assets':frontend_asset_lock(legacy)}))
    config={'original_root':str(original),'frontend_config':str(current)}
    api().frontend_identity(config)
    metadata.write_text('{"rms_norm_eps":0.01}')
    with pytest.raises(ValueError,match='SHA'):
        api().frontend_identity(config)
    metadata.write_text('{"rms_norm_eps":1e-6}')
    current.write_text(current.read_text()+'\nmodel: {llm_heads: 2}\n')
    with pytest.raises(ValueError,match='SHA'):
        api().frontend_identity(config)


def test_summary_must_match_saved_hash_and_recomputed_arrays(tmp_path):
    from continuous_mmwave.v26.source import file_sha
    summary=tmp_path/'evaluation-summary.json';summary.write_text(json.dumps({'score':1.}))
    completed=tmp_path/'evaluate-completed.json'
    completed.write_text(json.dumps({'result':{'summary_sha256':file_sha(summary)}}))
    api().verify_summary_artifact(tmp_path,{'score':1.})
    summary.write_text(json.dumps({'score':0.}))
    with pytest.raises(ValueError,match='SHA'):
        api().verify_summary_artifact(tmp_path,{'score':1.})
    completed.write_text(json.dumps({'result':{'summary_sha256':file_sha(summary)}}))
    with pytest.raises(ValueError,match='summary'):
        api().verify_summary_artifact(tmp_path,{'score':1.})
