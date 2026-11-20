import hashlib

import numpy as np


def noise_image(sample_id, seed, shape):
    generator = np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed, int(sample_id)])))
    return generator.integers(0, 256, size=shape, dtype=np.uint8).astype(np.float32)


def noise_fingerprint(sample_id, seed, shape):
    value = noise_image(sample_id, seed, shape)
    return hashlib.sha256(value.astype(np.uint8).tobytes()).hexdigest()


def gaussian_image(sample_id, seed, shape):
    generator = np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed, int(sample_id)])))
    return np.clip(generator.normal(127.5, np.sqrt(5461.25), size=shape), 0, 255).astype(np.float32)


def wrong_permutation(count, seed):
    if count < 2:
        raise ValueError('错误卫星置换至少需要两个目标')
    generator = np.random.default_rng(seed)
    # 随机排列形成循环，每个目标均获得其他目标的卫星。
    order = generator.permutation(count)
    permutation = np.empty(count, dtype=np.int64)
    permutation[order] = np.roll(order, 1)
    if np.any(permutation == np.arange(count)):
        raise AssertionError('错误卫星置换包含正确配对')
    return permutation
