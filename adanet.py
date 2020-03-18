from __future__ import absolute_import, division, print_function
import functools
import os
import shutil

import adanet
import tensorflow.compat.v1 as tf

SEED = 2020
LOG_DIR = '/logs'

(X_train, y_train), (X_test, y_test) = (tf.keras.datasets.boston_housing.load_data())

FEATURES_KEY = "x"


def input_fn(partition, training, batch_size):
    def _input_fn():
        if partition == 'train':
            dataset = tf.data.Dataset.from_tensor_slices(({
                FEATURES_KEY: tf.log1p(X_train)
            }, tf.log1p(y_train)))
        else:
            dataset = tf.data.Dataset.from_tensor_slices(({
                FEATURES_KEY: tf.log1p(X_test)
            }, tf.log1p(y_test)))

        if training:
            dataset = dataset.shuffle(10 * batch_size, seed=SEED).repeat()

        dataset = dataset.batch(batch_size)
        iterator = dataset.make_one_shot_iterator()
        features, labels = iterator.get_next()
        return features, labels

    return _input_fn()


