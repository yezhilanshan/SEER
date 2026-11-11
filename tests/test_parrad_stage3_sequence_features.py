from __future__ import annotations
import numpy as np
import pytest
import torch


def matrices(sources=("F", "E"), history=3):
    dimensions = {"F": 4, "E": 8, "z": 2}
    rng = np.random.default_rng(4)
    current = {s: rng.normal(size=(4,history,dimensions[s])).astype(np.float32) for s in sources}
    reference = {s: np.ones_like(current[s]) for s in sources}
    return current, reference


@pytest.mark.parametrize("sources", [("F",), ("E",), ("z",), ("F","E"), ("F","z")])
@pytest.mark.parametrize("history", [1,3,8])
def test_projected_history_capacity_for_all_sources_and_lengths(sources, history):
    from continuous_mmwave.parrad_stage3_sequence_features import project_history_block
    current, reference = matrices(sources, history)
    scales = {s: 1. for s in sources}
    first = project_history_block(current, reference, sources=sources, scales=scales, seed=3407)
    repeated = project_history_block(current, reference, sources=sources, scales=scales, seed=3407)
    other = project_history_block(current, reference, sources=sources, scales=scales, seed=7211)
    assert first.shape == (4,130)
    np.testing.assert_array_equal(first, repeated)
    assert not np.array_equal(first[:,:128], other[:,:128])
    np.testing.assert_array_equal(first[:,-2:], other[:,-2:])


def test_history_order_and_single_update_geometry():
    from continuous_mmwave.parrad_stage3_sequence_features import project_history_block
    c = {"F": np.array([[[1.,2.],[3.,4.],[5.,6.]]], np.float32)}
    r = {"F": np.ones((1,3,2), np.float32)}
    generator = torch.Generator().manual_seed(3407)
    matrix = (torch.randint(0,2,(6,128), generator=generator).float()*2-1)/np.sqrt(128)
    result = project_history_block(c,r,sources=("F",),scales={"F":2.},seed=3407)
    expected = (torch.tensor([[0.,.5,1.,1.5,2.,2.5]]) @ matrix).numpy()
    np.testing.assert_allclose(result[:,:128], expected, atol=1e-6)
    reversed_result = project_history_block({"F":c["F"][:,::-1]},r,sources=("F",),scales={"F":2.},seed=3407)
    assert not np.array_equal(result[:,:128], reversed_result[:,:128])
    one = project_history_block({"F": np.array([[[3.,4.]]])},{"F":np.array([[[0.,5.]]])},sources=("F",),scales={"F":2.},seed=3407)
    assert one[0,-2] == pytest.approx(.2)
    assert one[0,-1] == pytest.approx(np.sqrt(10)/2)


def test_fold_only_rms_scale_ignores_holdout_extremes():
    from continuous_mmwave.parrad_stage3_sequence_features import fit_source_scales
    c = {"F":np.array([[[2.]],[[4.]]])}; r = {"F":np.zeros((2,1,1))}
    scales = fit_source_scales(iter([(np.array([0,1]),c,r)]))
    assert scales["F"] == pytest.approx(np.sqrt(10))
    assert fit_source_scales(iter([(np.array([0,1]),c,r)])) == scales
    zero = fit_source_scales(iter([(np.array([0,1]),r,r)]))
    assert zero["F"] == 1e-6


def test_hybrid_source_rescaling_preserves_features():
    from continuous_mmwave.parrad_stage3_sequence_features import project_history_block
    c,r = matrices(); scales = {"F":2.,"E":3.}
    first = project_history_block(c,r,sources=("F","E"),scales=scales,seed=3407)
    c["E"] *= 100; r["E"] *= 100
    second = project_history_block(c,r,sources=("F","E"),scales={"F":2.,"E":300.},seed=3407)
    np.testing.assert_allclose(first,second,atol=2e-6)


def test_blockwise_features_match_dense_and_reject_nonfinite(tmp_path):
    from continuous_mmwave.parrad_stage3_sequence_data import SequenceEndpoint
    from continuous_mmwave.parrad_stage3_sequence_features import iter_history_blocks, project_history_block
    endpoints=[]
    for key in ("a","b"):
        (tmp_path/key).mkdir()
        c=np.arange(16, dtype=np.float32).reshape(8,2)
        np.savez_compressed(tmp_path/key/"layers.npz",current_F=c,reference_F=np.ones_like(c))
        endpoints.append(SequenceEndpoint(key,key,"natural",7,(5,6,7),{}))
    blocks=list(iter_history_blocks(tmp_path,endpoints,("F",),3,block_size=1))
    assert [i.tolist() for i,_,_ in blocks] == [[0],[1]]
    np.testing.assert_array_equal(blocks[0][1]["F"], [[[10,11],[12,13],[14,15]]])
    dense_c={"F":np.concatenate([c["F"] for _,c,_ in blocks])}
    dense_r={"F":np.concatenate([r["F"] for _,_,r in blocks])}
    dense=project_history_block(dense_c,dense_r,sources=("F",),scales={"F":1.},seed=3407)
    parts=np.concatenate([project_history_block(c,r,sources=("F",),scales={"F":1.},seed=3407) for _,c,r in blocks])
    np.testing.assert_allclose(dense,parts,atol=2e-6)
    dense_c["F"][0,0,0]=np.nan
    with pytest.raises(ValueError,match="finite"):
        project_history_block(dense_c,dense_r,sources=("F",),scales={"F":1.},seed=3407)


def test_compact_history_pack_matches_dense_for_fold_scales():
    from continuous_mmwave.parrad_stage3_sequence_features import pack_history_block, features_from_pack, project_history_block
    c,r=matrices(); pack=pack_history_block(c,r,sources=("F","E"),seed=3407)
    scales={"F":2.7,"E":4.2}
    first=features_from_pack(pack,scales)
    second=project_history_block(c,r,sources=("F","E"),scales=scales,seed=3407)
    np.testing.assert_allclose(first,second,atol=2e-6)
