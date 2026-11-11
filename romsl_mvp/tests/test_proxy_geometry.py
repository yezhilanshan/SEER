import numpy as np
from romsl_mvp.romsl.proxy_geometry import rigid_fit, nearest_bones, fit_icp, axis_rotations


def test_rigid_fit_recovers_known_transform():
    rng=np.random.default_rng(12)
    world=rng.normal(size=(100,3))
    rotation=np.array([[0,0,1],[-1,0,0],[0,-1,0.]])
    translation=np.array([.15,-.2,.3])
    observed=world@rotation.T+translation
    r,t=rigid_fit(world,observed,np.ones(100))
    np.testing.assert_allclose(r,rotation,atol=1e-12)
    np.testing.assert_allclose(t,translation,atol=1e-12)
    rotations=axis_rotations()
    assert len(rotations)==24
    assert all(np.isclose(np.linalg.det(r),1) for r in rotations)


def test_icp_uses_segment_interiors_and_recovers_local_fit():
    rng=np.random.default_rng(2)
    a=rng.normal(size=(300,1,3))
    b=a+rng.normal(size=(300,1,3))*.3
    alpha=rng.uniform(.1,.9,size=(300,1))
    world=a[:,0]+alpha*(b[:,0]-a[:,0])
    angle=.15
    rotation=np.array([[np.cos(angle),-np.sin(angle),0],[np.sin(angle),np.cos(angle),0],[0,0,1]])
    translation=np.array([.2,-.1,.05])
    observed=world@rotation.T+translation
    r,t,score=fit_icp(a,b,observed,np.ones(300),np.eye(3),np.zeros(3),iterations=80,trim=.9)
    np.testing.assert_allclose(r,rotation,atol=1e-5)
    np.testing.assert_allclose(t,translation,atol=1e-5)
    assert score < 1e-5
    matched,distance=nearest_bones(a,b,observed,r,t)
    assert distance.max()<1e-5
