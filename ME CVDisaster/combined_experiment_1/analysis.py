import numpy as np
from scipy.stats import binomtest
from sklearn.metrics import classification_report, confusion_matrix

from common import CLASS_NAMES


def classification_metrics(labels, predictions):
    labels = np.asarray(labels, dtype=np.int64)
    predictions = np.asarray(predictions, dtype=np.int64)
    if labels.shape != predictions.shape or labels.ndim != 1:
        raise ValueError('标签与预测必须为长度相同的一维数组')
    if not np.isin(labels, [0, 1, 2]).all() or not np.isin(predictions, [0, 1, 2]).all():
        raise ValueError('分类标签或预测超出三个类别')
    if not len(labels):
        return {'count': 0, 'accuracy': None, 'classification_report': None, 'confusion_matrix': None}
    return {'count': len(labels), 'accuracy': float((labels == predictions).mean()),
            'classification_report': classification_report(
                labels, predictions, labels=[0, 1, 2], target_names=CLASS_NAMES,
                output_dict=True, zero_division=0),
            'confusion_matrix': confusion_matrix(labels, predictions, labels=[0, 1, 2]).tolist()}


def paired_effect(labels, paired, combined, statistics):
    labels, paired, combined = (np.asarray(values, dtype=np.int64) for values in [labels, paired, combined])
    result = {'paired': classification_metrics(labels, paired),
              'combined': classification_metrics(labels, combined)}
    first, second = paired == labels, combined == labels
    harmful = int((first & ~second).sum())
    beneficial = int((~first & second).sum())
    count = len(labels)
    result.update(both_correct=int((first & second).sum()), both_incorrect=int((~first & ~second).sum()),
                  correct_to_incorrect=harmful, incorrect_to_correct=beneficial,
                  accuracy_change_percentage_points=None, change_ci_percentage_points=None,
                  harmful_fraction=None, beneficial_fraction=None, mcnemar_exact_two_sided_p=None)
    if count:
        result['accuracy_change_percentage_points'] = 100 * (beneficial - harmful) / count
        result['harmful_fraction'] = harmful / count
        result['beneficial_fraction'] = beneficial / count
        discordant = harmful + beneficial
        result['mcnemar_exact_two_sided_p'] = (
            float(binomtest(harmful, discordant, p=0.5).pvalue) if discordant else 1.0)
        rng = np.random.default_rng(statistics['bootstrap_seed'])
        draws = rng.multinomial(count, [harmful / count, beneficial / count,
                                      (count - discordant) / count],
                                size=statistics['bootstrap_repetitions'])
        differences = 100 * (draws[:, 1] - draws[:, 0]) / count
        tail = (1 - statistics['confidence_level']) / 2
        result['change_ci_percentage_points'] = np.quantile(differences, [tail, 1 - tail]).tolist()
    return result


def summarize(rows, statistics):
    if not rows or len({int(row['sample_id']) for row in rows}) != len(rows):
        raise ValueError('逐图预测不能为空或包含重复ID')
    for row in rows:
        if int(row['geo_correct']) not in [0, 1]:
            raise ValueError('geo_correct必须为0或1')
        if int(row['geo_correct']) and int(row['paired_prediction']) != int(row['combined_prediction']):
            raise ValueError('Geo检索正确的相同图片输入产生了不同Est预测')

    def effect(selected):
        return paired_effect([int(row['true_label']) for row in selected],
                             [int(row['paired_prediction']) for row in selected],
                             [int(row['combined_prediction']) for row in selected], statistics)

    wrong = [row for row in rows if not int(row['geo_correct'])]
    right = [row for row in rows if int(row['geo_correct'])]
    return {'all': effect(rows), 'geo_incorrect': effect(wrong), 'geo_correct': effect(right),
            'geo_incorrect_same_damage_class': effect([
                row for row in wrong if int(row['retrieved_satellite_label']) == int(row['true_label'])]),
            'geo_incorrect_different_damage_class': effect([
                row for row in wrong if int(row['retrieved_satellite_label']) != int(row['true_label'])]),
            'geo_top1_accuracy': len(right) / len(rows),
            'joint_accuracy': sum(int(row['geo_correct']) and int(row['combined_prediction']) == int(row['true_label'])
                                  for row in rows) / len(rows),
            'per_true_class': {name: effect([row for row in rows if int(row['true_label']) == label])
                               for label, name in enumerate(CLASS_NAMES)},
            'statistics': statistics,
            'interval_unit': 'query',
            'primary_comparison': 'geo_incorrect.combined - geo_incorrect.paired'}
