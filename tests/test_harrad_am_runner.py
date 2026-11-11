import copy
import pytest


def test_runner_rejects_budget_or_arm_drift_before_opening_sources():
    from continuous_mmwave.harrad.am_runner import validate_config
    cfg={'arms':['B0','B_align','B_mod','B_both'],'seeds':[42,43,44],'bank_per_class':32,
         'training':{'steps':1000,'checkpoints':[0,300,1000],'temperature':.1,'alignment_weight':.1,
                     'evaluation_batch_size':2,'max_batch_attempts':3}}
    validate_config(cfg)
    wrong=copy.deepcopy(cfg); wrong['training']['steps']=300
    with pytest.raises(ValueError,match='budget'): validate_config(wrong)
    wrong=copy.deepcopy(cfg); wrong['arms']=['B0','B_align']
    with pytest.raises(ValueError,match='arms'): validate_config(wrong)


def test_preflight_is_bound_to_config_and_sealed_source_before_reuse(tmp_path):
    from continuous_mmwave.harrad.am_runner import preflight_binding,validate_preflight_binding
    manifest=tmp_path/'source-manifest.json'; manifest.write_text('[{"sha256":"original"}]')
    config={'training':{'learning_rate':.0001},'arms':['B0']}
    proof={'passed':True,**preflight_binding(config,manifest)}
    validate_preflight_binding(config,proof,manifest)
    changed=copy.deepcopy(config); changed['training']['learning_rate']=.001
    with pytest.raises(ValueError,match='config'): validate_preflight_binding(changed,proof,manifest)
    manifest.write_text('[{"sha256":"changed"}]')
    with pytest.raises(ValueError,match='source'): validate_preflight_binding(config,proof,manifest)
