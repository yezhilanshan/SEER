import numpy as np
import pytest


def test_roles_rejected_before_any_signal_file_is_opened(tmp_path):
    from continuous_mmwave.harrad.boundary_runner import prepare_records
    with pytest.raises(ValueError, match='role'):
        prepare_records([dict(role='test', continuous_path='does-not-exist')], tmp_path / 'bad')


def test_normalization_is_train_only_and_continuous_cache_hash_is_checked(tmp_path):
    from continuous_mmwave.harrad.boundary_runner import prepare_records
    from continuous_mmwave.harrad.preprocess import sha256
    manifest = []
    for i, role in enumerate(('train', 'validation')):
        source = tmp_path / f'continuous-{i}.npy'
        np.save(source, np.random.default_rng(i).normal(size=(31, 128)).astype('float32') + i * 100)
        manifest.append(dict(role=role, subject=f'subject{i+1}', file=f'subject{i+1}_01.hdf5', frames=31,
                             continuous_path=str(source), instances=[dict(start=5, stop=20)],
                             artifacts={'continuous_hidden.npy': {'sha256': sha256(source)}}))
    records = prepare_records(manifest, tmp_path / 'first')
    assert len(records) == 2 and records[0]['x'].shape == (16, 256)
    with np.load(tmp_path / 'first/normalizers.npz') as a:
        mean1 = a['mean'].copy(); scale1 = a['scale'].copy()
    validation = manifest[1]
    np.save(validation['continuous_path'], np.zeros((31, 128), dtype='float32'))
    validation['artifacts']['continuous_hidden.npy']['sha256'] = sha256(validation['continuous_path'])
    prepare_records(manifest, tmp_path / 'second')
    with np.load(tmp_path / 'second/normalizers.npz') as a:
        np.testing.assert_array_equal(mean1, a['mean'])
        np.testing.assert_array_equal(scale1, a['scale'])
    validation['artifacts']['continuous_hidden.npy']['sha256'] = 'wrong'
    with pytest.raises(ValueError, match='hash'):
        prepare_records(manifest, tmp_path / 'changed')
