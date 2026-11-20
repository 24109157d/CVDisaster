from pathlib import Path
import os

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import f1_score

from analysis import holm, load_predictions, net_benefit, paired_group_comparison, primary_comparisons, subtract_scores
from common import ROOT


def real_predictions():
    source = ROOT.parent / 'combined_experiment_1' / 'results' / '4372858' / 'predictions.csv'
    raw = pd.read_csv(source, dtype={'sample_id': str})
    manifest = Path(os.environ.get('EST_TEST_MANIFEST', ROOT / '.local-cache' / 'manifest-est626'))
    metadata = pd.concat([pd.read_csv(manifest / f'{split}.csv', dtype={'sample_id': str})
                          for split in ('refit', 'test')], ignore_index=True)
    metadata = metadata[['sample_id', 'sequence']].rename(columns={'sequence': 'sequence_i'})
    return raw.merge(metadata, on='sample_id', validate='one_to_one')


def normalized_predictions(tmp_path):
    source = real_predictions()
    frames = []
    for condition in ('paired', 'noise'):
        frame = source[['sample_id', 'true_label', 'sequence_i']].rename(
            columns={'true_label': 'label', 'sequence_i': 'sequence'})
        frame['spatial_group'] = ''
        frame['training_seed'] = 42
        frame['condition_seed'] = 42
        frame['condition'] = condition
        prefix = 'paired_satellite' if condition == 'paired' else 'uniform_noise_satellite'
        frame['prediction'] = source[f'{prefix}_prediction']
        frame['archived_reference_prediction'] = source['uniform_noise_satellite_prediction']
        frames.append(frame)
    path = tmp_path / 'predictions.csv'
    pd.concat(frames, ignore_index=True).to_csv(path, index=False)
    return path


def test_real_noise_effect_and_macro_f1(tmp_path):
    path = normalized_predictions(tmp_path)
    left = load_predictions([path], 'paired', 'all')
    right = load_predictions([path], 'noise', 'all')
    result = paired_group_comparison(left, right, 'sequence', repetitions=100)
    np.testing.assert_allclose(result['gain_pp'], 4 / 1649 * 100, atol=1e-12)
    assert result['per_seed'][0]['corrected_count'] == 4
    assert result['per_seed'][0]['harmed_count'] == 0
    assert result['per_seed'][0]['Nfixed'] == 4
    assert result['per_seed'][0]['Ncorrupted'] == 0
    assert result['per_seed'][0]['delta_N'] == 4
    source = real_predictions()
    expected = f1_score(source['true_label'], source['paired_satellite_prediction'], average='macro') - f1_score(
        source['true_label'], source['uniform_noise_satellite_prediction'], average='macro')
    np.testing.assert_allclose(result['macro_f1_gain'], expected, atol=1e-12)
    reversed_order = paired_group_comparison(left.iloc[::-1], right, 'sequence', repetitions=100)
    np.testing.assert_allclose(result['gain_pp'], reversed_order['gain_pp'])


def test_real_identical_contrast_and_pairing_checks(tmp_path):
    path = normalized_predictions(tmp_path)
    left = load_predictions([path], 'paired', 'all')
    right = load_predictions([path], 'noise', 'all')
    difference = subtract_scores(left, left)
    reference = subtract_scores(right, right)
    result = paired_group_comparison(difference, reference, 'sequence', repetitions=100, contrast=True)
    assert result['gain_pp'] == 0
    assert result['ci95_pp'] == [0, 0]
    assert result['p_value'] == 1
    assert result['per_seed'][0]['Nfixed'] is None
    with pytest.raises(ValueError, match='集合不一致'):
        paired_group_comparison(left.iloc[1:], right, 'sequence', repetitions=100)
    with pytest.raises(ValueError, match='重复评价文件'):
        load_predictions([path, path], 'paired', 'all')


def test_holm_real_noise_comparisons(tmp_path):
    path = normalized_predictions(tmp_path)
    left = load_predictions([path], 'paired', 'all')
    right = load_predictions([path], 'noise', 'all')
    result = paired_group_comparison(left, right, 'sequence', repetitions=100)
    corrected = holm([result['p_value'], 1.0])
    assert corrected[0] >= result['p_value']
    assert corrected[1] == 1


def test_net_benefit_real_predictions():
    source = real_predictions()
    labels = source['true_label'].to_numpy()
    base = source['uniform_noise_satellite_prediction'].to_numpy() == labels
    final = source['paired_satellite_prediction'].to_numpy() == labels
    result = net_benefit(base, final)
    assert result['Nfixed'] == 4
    assert result['Ncorrupted'] == 0
    assert result['delta_N'] == 4
    np.testing.assert_allclose(result['net_benefit_pp'], 400 / 1649)
    reversed_result = net_benefit(final, base)
    assert reversed_result['Ncorrupted'] == 4
    assert reversed_result['delta_N'] == -4
    with pytest.raises(ValueError):
        net_benefit(base[1:], final)


def test_prediction_column_and_single_model_comparisons(tmp_path):
    path = normalized_predictions(tmp_path)
    final = load_predictions([path], 'paired', 'all')
    reference = load_predictions([path], 'paired', 'all', 'archived_reference_prediction')
    result = paired_group_comparison(final, reference, 'sequence', repetitions=100)
    assert result['per_seed'][0]['delta_N'] == 4
    comparisons = primary_comparisons({'full': [str(path)]})
    assert comparisons[0]['right']['prediction_column'] == 'base_prediction'
    assert {item['right']['condition'] for item in comparisons} == {
        'paired', 'wrong', 'noise', 'gaussian', 'zero'}
    with pytest.raises(ValueError):
        primary_comparisons({})
