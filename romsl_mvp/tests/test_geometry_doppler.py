import json
import numpy as np
import pytest

from romsl_mvp.romsl.geometry import load_extrinsics, transform_points, transform_velocities, estimate_los, project_radial
from romsl_mvp.romsl.doppler import load_doppler, read_doppler, verify_sign, echo_support, weight_statistics
from romsl_mvp.romsl.targets import build_cd_labels
from romsl_mvp.romsl.targets import compute_targets


def extrinsics():
    return {'verified': True, 'R': [[0,-1,0],[1,0,0],[0,0,1]], 't': [100,0,0],
            'translation_unit': 'cm', 'world_unit': 'm', 'from_frame': 'world',
            'to_frame': 'radar', 'radar_origin_world_m': [0,1,0],
            'world_axes':'synthetic right handed XYZ', 'radar_axes':'synthetic right handed XYZ',
            'source': 'synthetic exact calibration', 'verified_by': 'test',
            'validation_recordings': ['synthetic_validation'],
            'fit_recordings': ['synthetic_train'], 'mean_error_m': 0, 'p95_error_m': 0}


def test_transform_and_origin(tmp_path):
    path = tmp_path/'extrinsics.json'
    path.write_text(json.dumps(extrinsics()))
    g = load_extrinsics(path)
    np.testing.assert_allclose(transform_points([[1,0,0]],g), [[1,1,0]])
    np.testing.assert_allclose(transform_points([[0,1,0]],g), [[0,0,0]], atol=1e-12)
    np.testing.assert_allclose(transform_velocities([[1,0,0]],g), [[0,1,0]])
    homogeneous = extrinsics()
    rotation = np.array(homogeneous.pop('R'))
    translation = np.array(homogeneous.pop('t'))
    matrix = np.eye(4)
    matrix[:3,:3], matrix[:3,3] = rotation, translation
    homogeneous['T_world_to_radar'] = matrix.tolist()
    path.write_text(json.dumps(homogeneous))
    np.testing.assert_allclose(load_extrinsics(path)['t'], g['t'])


@pytest.mark.parametrize('change', [
    {'verified':False}, {'R':[[1,0,0],[0,1,0],[0,0,-1]]},
    {'from_frame':'radar'}, {'world_unit':'unknown'}, {'source':''},
    {'radar_axes':None},
    {'radar_origin_world_m':[0,0,0]}, {'t':[float('nan'),0,0]},
    {'fit_recordings':['synthetic_validation']},
])
def test_geometry_blocks_bad_config(tmp_path,change):
    value = {**extrinsics(), **change}
    path = tmp_path/'bad.json'
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError,match='BLOCKED_GEOMETRY'):
        load_extrinsics(path)


def test_joint_los_and_signed_radial():
    points = np.array([[2,0,0],[0,3,0],[0,0,0],[np.nan,0,0]])
    los, mask = estimate_los(points)
    np.testing.assert_array_equal(mask,[True,True,False,False])
    speeds, valid = project_radial(points,[[1,0,0],[0,-2,0],[1,0,0],[0,1,0]])
    np.testing.assert_allclose(speeds[:2],[1,-2])
    sideways, _ = project_radial(points[:2],[[0,1,0],[1,0,0]])
    np.testing.assert_allclose(sideways,0)
    assert np.isfinite(los).all() and not valid[2:].any()


def doppler():
    return {'verified':True, 'doppler_col':4, 'units':'cm/s',
            'positive_direction':'toward_radar', 'source':'synthetic sensor definition',
            'sign_verified':True, 'sign_verification_source':'known motion test',
            'verified_by':'test'}


def test_doppler_units_and_sign(tmp_path):
    path=tmp_path/'doppler.json'
    path.write_text(json.dumps(doppler()))
    spec=load_doppler(path)
    points=np.zeros((3,5))
    points[:,4]=[100,-200,np.nan]
    speed,valid=read_doppler(points,spec)
    np.testing.assert_allclose(speed,[-1,2,0])
    np.testing.assert_array_equal(valid,[True,True,False])
    assert verify_sign([-1,2,0],[-1,2,0])['passed']
    assert not verify_sign([1,-2],[-1,2])['passed']
    assert not verify_sign([0,0],[0,0])['passed']
    assert not verify_sign([1,2],[1,2])['passed']


@pytest.mark.parametrize('change',[{'verified':False},{'units':'unknown'},
    {'positive_direction':'unknown'},{'sign_verified':False},{'doppler_col':0}])
def test_doppler_gate(tmp_path,change):
    path=tmp_path/'bad.json'
    path.write_text(json.dumps({**doppler(),**change}))
    with pytest.raises(ValueError,match='BLOCKED_DOPPLER'):
        load_doppler(path)


def test_weight_distribution_and_empty_frames(tmp_path):
    path=tmp_path/'doppler.json'
    path.write_text(json.dumps(doppler()))
    spec=load_doppler(path)
    frames=[]
    for active in [0,1,2,8]:
        p=np.zeros((8,5))
        p[:active,4]=20
        frames.append(p)
    frames.append(np.empty((0,5)))
    result=echo_support(frames,spec)
    np.testing.assert_allclose(result['weights'],[.25,.625,1,1,.25])
    np.testing.assert_array_equal(result['frame_mask'],[True,True,True,True,False])
    summary=weight_statistics(result)
    assert summary['valid_frames']==4 and summary['empty_or_invalid_frames']==1
    assert summary['fraction_near_one']==.5
    saturated=echo_support([frames[3]]*20,spec)
    assert weight_statistics(saturated)['status']=='LOW_DIAGNOSTIC_VARIATION'
    assert weight_statistics(echo_support([frames[-1]],spec))['status']=='NO_VALID_FRAMES'


def test_formal_labels_gate_and_transformed_motion(tmp_path):
    geometry_path=tmp_path/'g.json'
    doppler_path=tmp_path/'d.json'
    g=extrinsics()
    g['verified']=False
    geometry_path.write_text(json.dumps(g))
    doppler_path.write_text(json.dumps(doppler()))
    joints={'verified':True,'left_hip':4,'right_hip':1,'groups':{
        'left_arm':[11,12,13],'right_arm':[14,15,16],
        'left_leg':[4,5,6],'right_leg':[1,2,3]}}
    pose=np.zeros((10,17,3))
    pose[...,1]=-1-np.arange(10)[:,None]*.1
    frames=[np.ones((8,5)) for _ in range(10)]
    with pytest.raises(ValueError,match='BLOCKED_GEOMETRY'):
        build_cd_labels(pose,frames,joints,.1,geometry_path,pose_unit='m')
    g['verified']=True
    geometry_path.write_text(json.dumps(g))
    output=build_cd_labels(pose,frames,joints,.1,geometry_path,pose_unit='m',doppler_path=doppler_path)
    np.testing.assert_allclose(output['C'][3:,0],1,atol=1e-6)
    np.testing.assert_allclose(output['C'][3:,1:],0,atol=1e-6)
    assert output['units']=='m/s' and output['los_mode']=='shared_root'
    assert not output['C_mask'][:3].any()
    bad=doppler()
    bad['sign_verified']=False
    doppler_path.write_text(json.dumps(bad))
    with pytest.raises(ValueError,match='BLOCKED_DOPPLER'):
        build_cd_labels(pose,frames,joints,.1,geometry_path,pose_unit='m',doppler_path=doppler_path)


def test_tangential_body_motion_is_near_zero(tmp_path):
    g=extrinsics()
    g.update(R=np.eye(3).tolist(),t=[0,0,0],translation_unit='m',radar_origin_world_m=[0,0,0])
    path=tmp_path/'g.json'
    path.write_text(json.dumps(g))
    geometry=load_extrinsics(path)
    angle=np.arange(20)*.001
    center=3*np.column_stack([np.cos(angle),np.sin(angle),np.zeros(20)])
    pose=np.repeat(center[:,None,:],17,axis=1)
    joints={'left_hip':4,'right_hip':1,'groups':{'left_arm':[11,12,13],
        'right_arm':[14,15,16],'left_leg':[4,5,6],'right_leg':[1,2,3]}}
    output=compute_targets(pose,joints,.1,radial=True,geometry=geometry)
    assert np.max(np.abs(output['C'][3:,0])) < 2e-5


def test_formal_cli_blocks_unverified_export(tmp_path):
    import subprocess
    import sys
    from pathlib import Path
    project=Path(__file__).resolve().parents[1]
    output=tmp_path/'blocked'
    result=subprocess.run([sys.executable,str(project/'scripts/08_cd_preflight.py'),
        '--output',str(output),'--data-root',str(tmp_path/'absent_data'),'--emit-labels'],
        capture_output=True,text=True)
    assert result.returncode==2, result.stderr
    report=json.loads((output/'report.json').read_text())
    assert not report['labels_emitted'] and not (output/'labels').exists()
    assert report['geometry_status']=='BLOCKED_GEOMETRY'


def test_verified_synthetic_cli_export_and_saturation_gate(tmp_path):
    import subprocess
    import sys
    from pathlib import Path
    project=Path(__file__).resolve().parents[1]
    config=json.loads((project/'configs/mvp.yaml').read_text())
    config['data']['splits']['train']=['S01']
    (tmp_path/'mvp.yaml').write_text(json.dumps(config))
    (tmp_path/'joints.yaml').write_bytes((project/'configs/joints.yaml').read_bytes())
    (tmp_path/'g.json').write_text(json.dumps(extrinsics()))
    (tmp_path/'d.json').write_text(json.dumps(doppler()))
    for action in range(1,28):
        directory=tmp_path/'data/E01/S01'/f'A{action:02}'
        (directory/'mmwave').mkdir(parents=True)
        pose=np.zeros((10,17,3))
        pose[...,1]=-1-np.arange(10)[:,None]*.1
        np.save(directory/'ground_truth.npy',pose)
        points=np.zeros((8,5))
        points[:,4]=20
        for frame in range(1,11):
            points.astype('<f8').tofile(directory/'mmwave'/f'frame{frame:03}.bin')
    output=tmp_path/'export'
    result=subprocess.run([sys.executable,str(project/'scripts/08_cd_preflight.py'),
        '--output',str(output),'--config',str(tmp_path/'mvp.yaml'),
        '--data-root',str(tmp_path/'data'),'--extrinsics',str(tmp_path/'g.json'),
        '--doppler',str(tmp_path/'d.json'),'--exp','D','--emit-labels'],
        capture_output=True,text=True)
    assert result.returncode==0, result.stderr
    report=json.loads((output/'report.json').read_text())
    assert report['labels_emitted'] and len(list((output/'labels').glob('*.npz')))==27
    assert report['weights']['status']=='LOW_DIAGNOSTIC_VARIATION'
    assert report['D_semantics_ready'] and not report['D_ready']
    assert (output/'weight_distribution.png').exists()
    assert (output/'per_action_weights.png').exists()
    with np.load(next((output/'labels').glob('*.npz'))) as data:
        np.testing.assert_allclose(data['C'][3:,0],1,atol=1e-6)
        assert data['D_mask'].shape==(10,5)
