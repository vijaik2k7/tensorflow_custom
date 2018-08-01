#########################################################################
# Version:		Q2_Estimator_API_Bank.py							    #
#																		#
# Description:	Build a customer Estimator API that can                 #
#               predict whether a customer will subscribe to a term		#
#				deposit, based on a number of demographic and account	#
#				attributes.												#
#																		#
# Input:		bank-full-v4-train.csv									#
#				bank-full-v4-test.csv									#
#																		#
# Notes:		
# Source is:http://archive.ics.uci.edu/ml/datasets/Bank+Marketing	#
#																		#
#																		#
# Last revised:	20 Jan 2018						#
#																		#
#																		#
#########################################################################
# coding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import argparse
import sys
import tempfile

# Import urllib
from six.moves import urllib

import numpy as np
import tensorflow as tf
import os
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
FLAGS = None

#Enable Logging
#tf.logging.set_verbosity(tf.logging.INFO)


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
    first_hidden_layer = tf.layers.dense(features["x"], 30, activation=tf.nn.crelu)

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
        return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions={
            'class': predictions,
            'prob': tf.nn.softmax(logits)
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

def main():

    CURR_PATH = os.getcwd()

    BANK_MKT_TRAINING = 'gs://ml_spec/data/bank-full-v4-train.csv' #'../data/bank-full-v4-train.csv'
    BANK_MKT_TEST = 'gs://ml_spec/data/bank-full-v4-test.csv' #'../data/bank-full-v4-test.csv'

    # Load datasets.
    print ('Loading datasets...')
    training_set = tf.contrib.learn.datasets.base.load_csv_with_header(
        filename=BANK_MKT_TRAINING,
        target_dtype=np.int,
        features_dtype=np.float32)

    test_set = tf.contrib.learn.datasets.base.load_csv_with_header(
        filename=BANK_MKT_TEST,
        target_dtype=np.int,
        features_dtype=np.float32)

    #tf.logging.set_verbosity(tf.logging.INFO)
    tf.logging.set_verbosity(tf.logging.ERROR)

    # Learning rate for the model
    LEARNING_RATE = 0.0005

    # Set model params
    model_params = {"learning_rate": LEARNING_RATE}

    # Instantiate Estimator
    nn = tf.estimator.Estimator(model_fn=model_fn, params=model_params)

    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": np.array(training_set.data)},
        y=np.array(training_set.target),
        num_epochs=None,
        shuffle=True)

    # Train
    nn.train(input_fn=train_input_fn, steps=10000)

    # Score accuracy
    test_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": np.array(test_set.data)},
        y=np.array(test_set.target),
        num_epochs=1,
        shuffle=False)

    ev = nn.evaluate(input_fn=test_input_fn)

    # Predict results on the test set
    predict_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": test_set.data},
        num_epochs=1,
        shuffle=False)
    predictions = nn.predict(input_fn=predict_input_fn)

    pred = []
    for i,p in enumerate(predictions):
        pred.append(p['class'])
    print("************************* Performance on Evaluation Set *************************")
    print("F1 Score: ", f1_score(test_set.target.tolist(), pred))
    print("Accuracy: ", accuracy_score(test_set.target.tolist(), pred))
    print("*********************************************************************************")

    #...................................................................
	# Classify two new persons.
	#...................................................................

    new_samples = np.array([[50,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,1,0,7500,1,0,5,5,5,1,-1,0],[28,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,1,0,0,0,1293,1,0,7,5,61,2,-1,0]], dtype=np.float32)

    predict_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x":new_samples}, num_epochs=1, shuffle=False)

    predictions_for_two_new_persons = list(nn.predict(input_fn=predict_input_fn))
    predicted_classes = [p['class'] for p in predictions_for_two_new_persons]

    print("\n\n\n****************************** Test on New Samples ******************************")
    k=0
    for i in predicted_classes:
        print("Sample {0} Class Predictions: {1}".format(k, i))
        print("Prediction Probability: {}".format(predictions_for_two_new_persons[k]['prob'][i]))
        k=k+1;
    print("*********************************************************************************\n")

if __name__ == "__main__":
    main()
