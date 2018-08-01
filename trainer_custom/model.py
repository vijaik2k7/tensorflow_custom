# Derivative of original work - Copyright 2018 Google Inc. All Rights Reserved. Licensed under the Apache
# License, Version 2.0 (the "License"); you may not use this file except in
# compliance with the License. You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

# File modified for Deloitte - Google ML Specialization under license the Apache
# License, Version 2.0 (the "License");

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import multiprocessing

import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.contrib.learn.python.learn.utils import input_fn_utils
import os
import numpy as np
import re
import time
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from tensorflow.python.feature_column import feature_column

from tensorflow.contrib.learn.python.learn.utils import (
    saved_model_export_utils)

tf.logging.set_verbosity(tf.logging.INFO)

# Data sets
BANK_MKT_TRAINING = 'gs://ml_spec/data/bank-full-v4-train.csv'
BANK_MKT_TEST = 'gs://ml_spec/data/bank-full-v4-test.csv'


COLUMNS = ['x1','x2','x3','x4','x5','x6','x7','x8','x9','x10','x11','x12','x13','x14','x15','x16','x17','x18','x19','x20','x21','x22','x23','x24',\
'x25','x26','x27','x28','x29','x30','y']

CSV_COLUMN_DEFAULTS = [[0.0],[0.0],[0.0],[0.0],[0.0],[0.0],[0.0],[0.0],[0.0],
                        [0.0],[0.0],[0.0],[0.0],[0.0],[0.0],[0.0],[0.0],[0.0],[0.0],[0.0],
                        [0.0],[0.0],[0.0],[0.0],[0.0],[0.0], [0.0],[0.0],[0.0],[0.0],[0]]


LABEL = "y"
FEATURES =  [c for c in COLUMNS if c not in LABEL]

FEATURE_COLUMNS = []
for f in FEATURES:
    FEATURE_COLUMNS += [tf.contrib.layers.real_valued_column(f, normalizer=None)]

def metric_fn(predictions=None, labels=None, weights=None):
    P, update_op1 = tf.contrib.metrics.streaming_precision(predictions, labels)
    R, update_op2 = tf.contrib.metrics.streaming_recall(predictions, labels)
    eps = 1e-5;
    return (2*(P*R)/(P+R+eps), tf.group(update_op1, update_op2))


def model_fn(features, labels, mode, params):
##Model function for an estimator
    tf.set_random_seed(1)
    # Connect the first hidden layer to input layer
    # (features["x"]) with relu activation

    feature_columns = get_input_layer_feature_columns()

    input_layer = tf.feature_column.input_layer(features=features,
                                                feature_columns=feature_columns)

    first_hidden_layer = tf.layers.dense(input_layer, 30, activation=tf.nn.crelu)

    # Connect the second hidden layer to first hidden layer with relu
    second_hidden_layer = tf.layers.dense(first_hidden_layer, 30, activation=tf.nn.crelu)

    # Connect the output layer to second hidden layer (no activation fn)
    # try sigmoid activation
    logits = tf.layers.dense(second_hidden_layer, 2, activation=None)

    # Reshape and retrieve the probably class as prediction
    predictions = tf.argmax(logits, 1)

    # Provide an estimator spec for `ModeKeys.PREDICT`.
    if mode == tf.estimator.ModeKeys.PREDICT:
        tf.set_random_seed(1)
        prediction={
            'class': predictions,
            'prob': tf.nn.softmax(logits)
        }
        return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=prediction, export_outputs = {
            'classes': tf.estimator.export.PredictOutput(prediction)
        })

    # Calculate weights for class imbalance problem to be included in the loss function
    ratio = 0.115847085 #ratio of positive class in training set
    weights = tf.add(tf.convert_to_tensor(ratio, tf.float64) ,tf.scalar_mul((1.0 - ratio - ratio), tf.cast(labels, tf.float64)))

    # Calculate loss using softmax cross entropy
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits, weights=weights)

    # Calculate accuracy as additional eval metric
    eval_metric_ops = {
      'accuracy': tf.metrics.accuracy(
          labels=labels, predictions=predictions)
    }

    eval_metrics_ops_f1 = {
        'f1': metric_fn(labels=labels, predictions=predictions)
    }

    optimizer = tf.train.GradientDescentOptimizer(
      learning_rate=params["learning_rate"])
    train_op = optimizer.minimize(
      loss=loss, global_step=tf.train.get_global_step())

    # Provide an estimator spec for `ModeKeys.EVAL` and `ModeKeys.TRAIN` modes.
    return tf.estimator.EstimatorSpec(
      mode=mode,
      loss=loss,
      train_op=train_op,
      eval_metric_ops=eval_metrics_ops_f1)

def build_estimator(config, LEARNING_RATE=0.0005):
    # Set model params
    model_params = {"learning_rate": LEARNING_RATE}
    # Instantiate Estimator
    nn = tf.estimator.Estimator(model_fn=model_fn, params=model_params, config=config)
    return nn


def parse_label_column(label_string_tensor):
    """Parses a string tensor into the label tensor
    Args:
    label_string_tensor: Tensor of dtype string. Result of parsing the
    CSV column specified by LABEL_COLUMN
    Returns:
    A Tensor of the same shape as label_string_tensor, should return
    an int64 Tensor representing the label index for classification tasks,
    and a float32 Tensor representing the value for a regression task.
    """
    # Build a Hash Table inside the graph
    table = tf.contrib.lookup.index_table_from_tensor(tf.constant(LABELS))

    # Use the hash table to convert string labels to ints and one-hot encode
    return table.lookup(label_string_tensor)

def get_feature_columns():

    columns = {}

    for feature_name in FEATURES:
        columns[feature_name] = tf.feature_column.numeric_column(feature_name)

    feature_columns = {}

    if columns is not None:
        feature_columns.update(columns)

    return feature_columns

def get_input_layer_feature_columns():

    feature_columns = list(get_feature_columns().values())

    dense_columns = list(
        filter(lambda column: isinstance(column, feature_column._NumericColumn),
               feature_columns
        )
    )

    return dense_columns
# ************************************************************************
# YOU NEED NOT MODIFY ANYTHING BELOW HERE TO ADAPT THIS MODEL TO YOUR DATA
# ************************************************************************


def csv_serving_input_fn():
    """Build the serving inputs."""
    csv_row = tf.placeholder(
      shape=[None],
      dtype=tf.string
    )
    features = parse_csv(csv_row)
    features.pop(LABEL_COLUMN)
    return tf.estimator.export.ServingInputReceiver(features, {'csv_row': csv_row})


def example_serving_input_fn():
    """Build the serving inputs."""
    example_bytestring = tf.placeholder(
      shape=[None],
      dtype=tf.string,
    )
    feature_scalars = tf.parse_example(
      example_bytestring,
      tf.feature_column.make_parse_example_spec(INPUT_COLUMNS)
    )
    return tf.estimator.export.ServingInputReceiver(
      features,
      {'example_proto': example_bytestring}
    )

# [START serving-function]
def json_serving_input_fn():
    inputs = {}
    for feat in FEATURE_COLUMNS:
        inputs[feat.name] = tf.placeholder(shape=[None], dtype=feat.dtype)
    return tf.estimator.export.ServingInputReceiver(inputs, inputs)
    # [END serving-function]

SERVING_FUNCTIONS = {
    'JSON': json_serving_input_fn,
    'EXAMPLE': example_serving_input_fn,
    'CSV': csv_serving_input_fn
}


def parse_csv(rows_string_tensor):
    row_columns = tf.expand_dims(rows_string_tensor, -1)
    columns = tf.decode_csv(row_columns, record_defaults=CSV_COLUMN_DEFAULTS)
    features = dict(zip(COLUMNS, columns))
    return features



def input_fn(filenames,
                      num_epochs=None,
                      shuffle=True,
                      skip_header_lines=1,
                      batch_size=200):
    """Generates features and labels for training or evaluation.
    """

    dataset = tf.data.TextLineDataset(filenames).skip(skip_header_lines).map(parse_csv)

    if shuffle:
        dataset = dataset.shuffle(buffer_size=batch_size * 10)
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.batch(batch_size)
    iterator = dataset.make_one_shot_iterator()
    features = iterator.get_next()
    return features, features.pop(LABEL) #parse_label_column(features.pop(LABEL_COLUMN))
