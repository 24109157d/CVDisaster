"""使用现有 TensorFlow/GCViT 环境验证默认缓存命中及真实预训练加载，禁止联网。"""

import argparse
import json
from pathlib import Path
import traceback
from unittest.mock import patch

from prepare_est_cache import EXPECTED_SHA256, SOURCE, prepare, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    output = parser.parse_args().output
    assert not output.exists(), '拒绝覆盖已有验证结果'
    result = {'status': 'running', 'stage': 'check_cache'}

    def save():
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')

    save()
    try:
        result['cache'] = prepare(check_only=True)
        result['stage'] = 'import_existing_tensorflow'
        save()
        print('EST_CACHE_VERIFY_IMPORT_START', flush=True)
        import numpy as np
        import tensorflow as tf
        import keras.utils.data_utils as data_utils
        from gcvit import GCViTTiny
        tf.config.threading.set_intra_op_parallelism_threads(2)
        tf.config.threading.set_inter_op_parallelism_threads(2)
        assert tf.__version__ == '2.10.1'
        assert tf.keras.utils.get_file is data_utils.get_file, '下载拦截入口与实际 Keras 加载器不一致'

        # 捕获实际 Keras 下载入口；只要发生任何下载尝试，验证立即失败。
        with patch.object(data_utils, 'urlretrieve', side_effect=AssertionError('禁止联网下载')) as download:
            result['stage'] = 'keras_get_file_offline'
            save()
            path = tf.keras.utils.get_file('gcvit_tiny_weights.h5',
                'https://github.com/awsaf49/gcvit-tf/releases/download/v1.1.1/gcvit_tiny_weights.h5')
            assert Path(path).samefile(SOURCE)
            print('KERAS_GET_FILE_CACHE_HIT', path, flush=True)
            result['stage'] = 'gcvit_pretrained_load'
            save()
            # 224 像素仅用于 CPU 验证模型工厂实际加载权重，不改正式训练输入尺寸。
            model = GCViTTiny(input_shape=(224, 224, 3), pretrain=True, resize_query=True)
            prediction = model(tf.zeros((1, 224, 224, 3)), training=False).numpy()
            assert prediction.shape == (1, 1000) and np.isfinite(prediction).all()
            assert np.allclose(prediction.sum(axis=1), 1, atol=1e-5)
            assert download.call_count == 0
        assert sha256(SOURCE) == EXPECTED_SHA256, '验证后权重发生变化'
        result.update(status='passed', stage='complete', tensorflow=tf.__version__,
                      keras_cache_path=path, download_attempts=0, prediction_shape=list(prediction.shape))
        print('EST_PRETRAINED_OFFLINE_LOAD_VERIFIED', flush=True)
    except BaseException:
        result.update(status='failed', traceback=traceback.format_exc())
        raise
    finally:
        save()


if __name__ == '__main__':
    main()
