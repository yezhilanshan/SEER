"""Exploratory registration. Bone proximity is not a calibration correspondence."""
import itertools
import numpy as np


def axis_rotations():
    result=[]
    for permutation in itertools.permutations(range(3)):
        for signs in itertools.product((-1,1),repeat=3):
            matrix=np.eye(3)[list(permutation)]*np.array(signs)[:,None]
            if np.linalg.det(matrix)>.5:
                result.append(matrix)
    return result


def rigid_fit(world, observed, weights):
    weights=np.asarray(weights,dtype=float)
    weights=weights/weights.sum()
    c1=(world*weights[:,None]).sum(0)
    c2=(observed*weights[:,None]).sum(0)
    covariance=(world-c1).T@((observed-c2)*weights[:,None])
    u,_,vh=np.linalg.svd(covariance)
    correction=np.eye(3)
    correction[2,2]=np.linalg.det(vh.T@u.T)
    rotation=vh.T@correction@u.T
    return rotation,c2-rotation@c1


def nearest_bones(a,b,points,rotation,translation):
    start=a@rotation.T+translation
    delta=(b-a)@rotation.T
    length=(delta**2).sum(-1)
    alpha=np.clip(((points[:,None]-start)*delta).sum(-1)/np.maximum(length,1e-12),0,1)
    projected=start+alpha[...,None]*delta
    squared=((points[:,None]-projected)**2).sum(-1)
    squared[length<1e-12]=np.inf
    nearest=squared.argmin(axis=1)
    index=np.arange(len(points))
    matched=a[index,nearest]+alpha[index,nearest,None]*(b-a)[index,nearest]
    return matched,np.sqrt(squared[index,nearest])


def fit_icp(a,b,points,weights,rotation,translation,*,iterations=40,trim=.7):
    rotation,translation=rotation.copy(),translation.copy()
    best=(rotation.copy(),translation.copy(),float('inf'))
    for _ in range(iterations):
        matched,distance=nearest_bones(a,b,points,rotation,translation)
        keep=distance<=np.quantile(distance,trim)
        score=float(np.sqrt(np.average(distance[keep]**2,weights=weights[keep])))
        if score<best[2]:
            best=(rotation.copy(),translation.copy(),score)
        update_r,update_t=rigid_fit(matched[keep],points[keep],weights[keep])
        change=np.linalg.norm(update_r-rotation)+np.linalg.norm(update_t-translation)
        rotation,translation=update_r,update_t
        if change<1e-8:
            break
    return best


def angle_degrees(first,second):
    return float(np.degrees(np.arccos(np.clip((np.trace(first@second.T)-1)/2,-1,1))))
