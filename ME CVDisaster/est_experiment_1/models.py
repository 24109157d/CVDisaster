import math

import tensorflow as tf
from tensorflow_addons.activations import sparsemax

from common import sha256


VARIANTS = ('full',)


def policy(variant):
    if variant not in VARIANTS:
        raise ValueError(f'未知模型: {variant}')
    return dict(variant=variant, ground=True, satellite=True, neighbors=True, auxiliary=True)


class Encoder(tf.keras.layers.Layer):
    def __init__(self, shape, settings, **kwargs):
        super().__init__(**kwargs)
        self.backbone = tf.keras.applications.ConvNeXtTiny(
            include_top=False, include_preprocessing=True, pooling=None,
            weights=settings['initial_weights'], input_shape=tuple(shape))

    def call(self, inputs, training=None):
        return self.backbone(inputs, training=training)


class Position(tf.keras.layers.Layer):
    def build(self, shape):
        self.position = self.add_weight(
            'position', shape=(1, shape[1], shape[2], shape[3]),
            initializer=tf.keras.initializers.RandomNormal(stddev=0.02))

    def call(self, inputs):
        return inputs + self.position


class SatelliteModulation(tf.keras.layers.Layer):
    def __init__(self, dimension, heads, attention_dimension, strength, **kwargs):
        super().__init__(**kwargs)
        self.strength = strength
        self.norm = tf.keras.layers.LayerNormalization(epsilon=1e-5)
        self.satellite_norm = tf.keras.layers.LayerNormalization(epsilon=1e-5)
        self.reader = tf.keras.layers.MultiHeadAttention(
            heads, attention_dimension // heads, output_shape=dimension)
        self.hidden = tf.keras.layers.Dense(dimension, activation='gelu')
        self.film = tf.keras.layers.Dense(2 * dimension, kernel_initializer='zeros',
                                          bias_initializer='zeros')

    def call(self, inputs, training=None):
        ground, satellite = inputs
        normalized = self.norm(ground)
        context = self.reader(normalized, self.satellite_norm(satellite), training=training)
        gamma, beta = tf.split(tf.tanh(self.film(self.hidden(context))), 2, axis=-1)
        return ground + self.strength * (gamma * normalized + beta)


class Evidence(tf.keras.layers.Layer):
    def __init__(self, dimension, heads, attention_dimension, threshold, **kwargs):
        super().__init__(**kwargs)
        self.heads = heads
        self.width = attention_dimension // heads
        self.threshold = threshold
        self.norm = tf.keras.layers.LayerNormalization(epsilon=1e-5)
        self.query = tf.keras.layers.Dense(attention_dimension, use_bias=False)
        self.key = tf.keras.layers.Dense(attention_dimension, use_bias=False)
        self.value = tf.keras.layers.Dense(attention_dimension, use_bias=False)
        self.relationship = tf.keras.layers.Dense(heads, use_bias=False)
        self.projection = tf.keras.layers.Dense(dimension, use_bias=False)

    def build(self, shape):
        self.null_key = self.add_weight(
            'null_key', shape=(self.heads, self.width),
            initializer=tf.keras.initializers.RandomNormal(stddev=0.02))
        super().build(shape)

    def split_heads(self, value):
        value = tf.reshape(value, [tf.shape(value)[0], tf.shape(value)[1], self.heads, self.width])
        return tf.transpose(value, [0, 2, 1, 3])

    def call(self, inputs):
        target, neighbors, mask, relations = inputs
        tf.debugging.assert_greater_equal(mask, 0.0)
        tf.debugging.assert_less_equal(mask, 1.0)
        query = self.split_heads(self.query(self.norm(target)))
        observations, masses = [], []
        for index in range(int(neighbors.shape[1])):
            current = self.norm(neighbors[:, index])
            key = self.split_heads(self.key(current))
            value = self.split_heads(self.value(current))
            scores = tf.einsum('bhqd,bhkd->bhqk', query, key) / math.sqrt(self.width)
            scores += self.relationship(relations[:, index])[:, :, None, None]
            scores = tf.where(mask[:, index, None, None, None] > 0, scores, -1e9)
            null_score = tf.einsum('bhqd,hd->bhq', query, self.null_key)[..., None] / math.sqrt(self.width)
            weights = sparsemax(tf.concat([scores, null_score], axis=-1), axis=-1)
            tf.debugging.check_numerics(weights, 'Sparsemax 权重出现非有限数值')
            real_weights = weights[..., :-1]
            mass = tf.reduce_sum(real_weights, axis=-1)
            # Null Value 为零；读取局部证据后计算邻图之间的重复程度。
            observed = tf.einsum('bhqk,bhkd->bhqd', real_weights, value)
            observed = tf.math.divide_no_nan(observed, mass[..., None])
            observed = tf.reshape(tf.transpose(observed, [0, 2, 1, 3]),
                                  [tf.shape(target)[0], tf.shape(target)[1], self.heads * self.width])
            observations.append(observed)
            masses.append(tf.reduce_mean(mass, axis=1) * mask[:, index, None])
        values = tf.stack(observations, axis=2)
        mass = tf.stack(masses, axis=2)
        normalized = tf.math.l2_normalize(values, axis=-1)
        similarity = tf.einsum('bqnd,bqmd->bqnm', normalized, normalized)
        similarity = tf.clip_by_value((similarity - self.threshold) / (1.0 - self.threshold), 0.0, 1.0)
        identity = tf.eye(int(neighbors.shape[1]), dtype=values.dtype)[None, None]
        similarity = similarity * (1.0 - identity) + identity
        density = tf.reduce_sum(similarity * mask[:, None, None, :], axis=-1)
        weight = tf.math.divide_no_nan(mass, density)
        weight /= 1.0 + tf.reduce_sum(weight, axis=-1, keepdims=True)
        increment = tf.reduce_mean(tf.reduce_sum(values * weight[..., None], axis=2), axis=1)
        return self.projection(increment)


class UncertaintyGate(tf.keras.layers.Layer):
    def __init__(self, settings, **kwargs):
        super().__init__(**kwargs)
        self.start = settings['gate_start_epoch']
        self.end = settings['gate_full_epoch']
        self.steps_per_epoch = 1

    def build(self, shape):
        self.slope = self.add_weight('slope', shape=(), initializer=tf.keras.initializers.Constant(1.0))
        self.intercept = self.add_weight('intercept', shape=(), initializer=tf.keras.initializers.Constant(-2.0))
        self.progress = self.add_weight('progress', shape=(), initializer='zeros', trainable=False)

    def call(self, probability):
        probability = tf.clip_by_value(probability, 1e-7, 1.0)
        uncertainty = tf.stop_gradient(-tf.reduce_sum(probability * tf.math.log(probability), axis=-1,
                                                     keepdims=True) / math.log(3))
        release = tf.clip_by_value((self.progress - self.start) / (self.end - self.start), 0.0, 1.0)
        return release * tf.sigmoid(tf.nn.softplus(self.slope) * uncertainty + self.intercept)


class ClassificationLoss(tf.keras.losses.Loss):
    def __init__(self, ordinal_weight, boundary_weights):
        super().__init__(name='classification_ordinal')
        self.ordinal_weight = ordinal_weight
        self.boundary_weights = boundary_weights

    def call(self, labels, probability):
        labels = tf.cast(tf.reshape(labels, [-1]), tf.int32)
        ce = tf.keras.losses.sparse_categorical_crossentropy(labels, probability)
        cumulative = tf.cumsum(probability, axis=-1)[:, :2]
        expected = tf.cast(labels[:, None] <= tf.range(2)[None, :], probability.dtype)
        ordinal = 0.5 * tf.reduce_sum(tf.square(cumulative - expected) *
                                     tf.cast(self.boundary_weights, probability.dtype), axis=-1)
        return ce + self.ordinal_weight * ordinal


def classifier(name):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(64), tf.keras.layers.LeakyReLU(),
        tf.keras.layers.Dropout(0.2), tf.keras.layers.Dense(3, activation='softmax')], name=name)


def build_model(settings, variant='full'):
    policy(variant)
    if sha256(settings['initial_weights']) != settings['initial_sha256']:
        raise ValueError('ConvNeXt-Tiny ImageNet 初始权重指纹不一致')
    dimension = settings['dimension']
    inputs = {
        'ground': tf.keras.Input(settings['ground_shape'], name='ground'),
        'satellite': tf.keras.Input(settings['satellite_shape'], name='satellite'),
        'neighbors': tf.keras.Input((settings['neighbors'], *settings['ground_shape']), name='neighbors'),
        'mask': tf.keras.Input((settings['neighbors'],), name='mask'),
        'relations': tf.keras.Input((settings['neighbors'], 2), name='relations')}
    ground_encoder = Encoder(settings['ground_shape'], settings, name='svi')
    satellite_encoder = Encoder(settings['satellite_shape'], settings, name='sat')
    ground_projection = tf.keras.layers.Dense(dimension, name='ground_projection')
    ground_position = Position(name='ground_position')
    satellite_projection = tf.keras.layers.Dense(dimension, name='satellite_projection')
    satellite_parts = []
    satellite = inputs['satellite']
    local = tf.keras.layers.CenterCrop(settings['local_crop_size'], settings['local_crop_size'])(satellite)
    local = tf.keras.layers.Resizing(*settings['satellite_shape'][:2])(local)
    for scale, image in enumerate((satellite, local)):
        feature = satellite_projection(satellite_encoder(image))
        feature = tf.keras.layers.AveragePooling2D(2)(feature)
        feature = Position(name=f'satellite_position_{scale}')(feature)
        satellite_parts.append(tf.keras.layers.Reshape((-1, dimension))(feature))
    satellite_tokens = tf.keras.layers.Concatenate(axis=1)(satellite_parts)
    modulation = SatelliteModulation(dimension, settings['heads'], settings['attention_dimension'],
                                     settings['film_strength'], name='satellite_modulation')

    def encode_ground(image):
        feature = ground_position(ground_projection(ground_encoder(image)))
        feature = tf.keras.layers.Reshape((-1, dimension))(feature)
        return modulation([feature, satellite_tokens])

    target = encode_ground(inputs['ground'])
    neighbors = []
    height, width = [item // 32 for item in settings['ground_shape'][:2]]
    for index in range(settings['neighbors']):
        feature = encode_ground(inputs['neighbors'][:, index])
        feature = tf.keras.layers.Reshape((height, width, dimension))(feature)
        feature = tf.keras.layers.AveragePooling2D(2)(feature)
        neighbors.append(tf.keras.layers.Reshape((-1, dimension))(feature))
    increment = Evidence(dimension, settings['heads'], settings['attention_dimension'],
                         settings['redundancy_threshold'], name='evidence')(
        [target, tf.stack(neighbors, axis=1), inputs['mask'], inputs['relations']])
    base = tf.keras.layers.GlobalAveragePooling1D()(target)
    head = classifier('shared_classifier')
    base_prediction = head(base)
    gate = UncertaintyGate(settings, name='uncertainty_gate')(base_prediction)
    final_prediction = head(base + gate * increment)
    satellite_prediction = classifier('satellite_classifier')(
        tf.keras.layers.GlobalAveragePooling1D()(satellite_tokens))
    outputs = {'final': tf.keras.layers.Activation('linear', name='final')(final_prediction),
               'base': tf.keras.layers.Activation('linear', name='base')(base_prediction),
               'satellite_aux': tf.keras.layers.Activation('linear', name='satellite_aux')(satellite_prediction)}
    return tf.keras.Model(inputs, outputs, name='convnext_tiny_mare')


class WarmUpCosine(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, settings, steps):
        super().__init__()
        self.base = settings['learning_rate']
        self.start = settings['warmup_learning_rate']
        self.warmup = settings['warmup_steps']
        self.total = steps * settings['epochs']
        if self.total <= self.warmup:
            raise ValueError('调度总步数必须超过预热步数')

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        decay = self.base * 0.5 * (1 + tf.cos(math.pi * (step - self.warmup) / (self.total - self.warmup)))
        warmup = self.start + (self.base - self.start) * step / self.warmup
        return tf.where(step > self.total, 0.0, tf.where(step < self.warmup, warmup, decay))


def compile_model(model, settings, steps):
    model.get_layer('uncertainty_gate').steps_per_epoch = steps
    model.compile(
        optimizer=tf.keras.optimizers.SGD(WarmUpCosine(settings, steps)),
        loss={'final': ClassificationLoss(settings['lambda_ordinal'], settings['ordinal_boundary_weights']),
              'base': tf.keras.losses.SparseCategoricalCrossentropy(),
              'satellite_aux': tf.keras.losses.SparseCategoricalCrossentropy()},
        loss_weights={'final': 1.0, 'base': settings['lambda_base'],
                      'satellite_aux': settings['lambda_satellite']},
        metrics={'final': [tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy')],
                 'base': [tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy')]})
