import numpy as np
import pandas as pd
import pytest

from common import ROOT
from conditions import noise_fingerprint, noise_image, wrong_permutation


def test_noise_matches_real_archived_fingerprints():
    source = pd.read_csv(ROOT.parent / 'combined_experiment_1' / 'results' / '4372858' / 'predictions.csv',
                         dtype={'sample_id': str})
    for index in (0, len(source) // 2, len(source) - 1):
        item = source.iloc[index]
        assert noise_fingerprint(item['sample_id'], 42, (512, 512, 3)) == item['noise_sha256']
        noise = noise_image(item['sample_id'], 42, (512, 512, 3))
        assert noise.dtype == np.float32
        np.testing.assert_allclose(noise.mean(dtype=np.float64), item['noise_mean'], atol=1e-8)


def test_wrong_satellites_replace_every_real_query():
    source = pd.read_csv(ROOT.parent / 'combined_experiment_1' / 'results' / '4372858' / 'predictions.csv',
                         dtype={'sample_id': str})
    ids = source['sample_id'].to_numpy()
    for seed in range(42, 47):
        indices = wrong_permutation(len(source), seed)
        assert (ids[indices] != ids).all()
        assert set(ids[indices]) == set(ids)
        np.testing.assert_array_equal(indices, wrong_permutation(len(source), seed))
    with pytest.raises(ValueError):
        wrong_permutation(1, 42)
