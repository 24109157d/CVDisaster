from pathlib import Path
import os

import numpy as np
import pandas as pd
import pytest
from pyproj import Geod

from common import EST_REFERENCE_SHA256, ROOT, check_manifest_settings, configuration, load_manifest, read_csv, sha256
from prepare import spatial_groups, validation_split


def manifest_path():
    return Path(os.environ.get('EST_TEST_MANIFEST', ROOT / '.local-cache' / 'manifest-est626'))


def test_real_manifest_access_and_neighbors():
    directory = manifest_path()
    manifest = load_manifest(directory)
    settings = configuration()
    check_manifest_settings(settings, manifest)
    sets = {}
    geod = Geod(ellps='WGS84')
    for split in ('train', 'validation', 'refit', 'test'):
        rows = {item['sample_id']: item for item in read_csv(directory / f'{split}.csv')}
        sets[split] = set(rows)
        adjacent = read_csv(directory / f'{split}_neighbors.csv')
        selected = set()
        for item in adjacent:
            a, b = rows[item['sample_id']], rows[item['neighbor_id']]
            assert a['sample_id'] != b['sample_id']
            assert a['sequence'] == b['sequence']
            assert (a['sample_id'], b['sample_id']) not in selected
            selected.add((a['sample_id'], b['sample_id']))
            _, _, distance = geod.inv(float(a['longitude']), float(a['latitude']),
                                     float(b['longitude']), float(b['latitude']))
            assert distance <= settings['radius_m'] + 1e-6
            assert abs(float(b['timestamp']) - float(a['timestamp'])) <= settings['time_seconds']
            np.testing.assert_allclose(distance, float(item['distance_m']), atol=1e-6)
        assert set(adjacent[0]) == {'sample_id', 'neighbor_id', 'rank', 'distance_m', 'delta_seconds'}
    assert not sets['train'] & sets['validation']
    assert not sets['refit'] & sets['test']
    assert sets['train'] | sets['validation'] == sets['refit']
    assert len(sets['refit']) == 2472
    assert len(sets['test']) == 1649
    assert len(sets['refit'] | sets['test']) == 4121
    assert manifest['protocol'] == 'est626'


def test_real_validation_sequences_are_disjoint():
    directory = manifest_path()
    train = pd.DataFrame(read_csv(directory / 'train.csv'))
    validation = pd.DataFrame(read_csv(directory / 'validation.csv'))
    assert not set(train['sequence']) & set(validation['sequence'])
    assert set(train['label']) == set(validation['label']) == {'0', '1', '2'}


def test_real_spatial_group_sequence_membership():
    frame = pd.DataFrame(read_csv(manifest_path() / 'refit.csv'))
    frame[['longitude', 'latitude']] = frame[['longitude', 'latitude']].astype(float)
    frame['spatial_group'] = spatial_groups(frame)
    assert frame.groupby('sequence')['spatial_group'].nunique().max() == 1


def test_exact_reproduced_est_membership_order_and_seed():
    directory = manifest_path()
    reference = ROOT / '.local-cache' / 'est-reference-4087311'
    for source, target in [('train', 'refit'), ('test', 'test')]:
        path = reference / f'est_{source}_60.csv'
        assert sha256(path) == EST_REFERENCE_SHA256[source]
        expected = read_csv(path)
        actual = read_csv(directory / f'{target}.csv')
        assert [item['ground'] for item in actual] == [f"00_SVI/{item['svi_relative_path']}" for item in expected]
        assert [int(item['label']) for item in actual] == [int(item['label'][0]) for item in expected]
    settings = configuration()
    frame = pd.DataFrame(read_csv(directory / 'refit.csv'))
    frame['label'] = frame['label'].astype(int)
    inner, validation = validation_split(frame, settings['validation_seed'], settings['validation_fraction'])
    for name, expected in [('train', inner), ('validation', validation)]:
        assert [item['sample_id'] for item in read_csv(directory / f'{name}.csv')] == expected['sample_id'].tolist()
    manifest = load_manifest(directory)
    with pytest.raises(ValueError, match='固定划分不一致'):
        check_manifest_settings({**settings, 'split_seed': 42}, manifest)
