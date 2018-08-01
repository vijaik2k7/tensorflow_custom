# Derivative of original work - Copyright 2018 Google Inc. All Rights Reserved. Licensed under the Apache
# License, Version 2.0 (the "License"); you may not use this file except in
# compliance with the License. You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

# File modified for Deloitte - Google ML Specialization under license the Apache
# License, Version 2.0 (the "License");

import argparse
import os

import trainer_custom.model as model

import tensorflow as tf
from tensorflow.contrib.learn.python.learn.utils import (
    saved_model_export_utils)
from tensorflow.contrib.training.python.training import hparam


def run_experiment(hparams):
    """Run the training and evaluate using the high level API"""

    train_input = lambda: model.input_fn(
      hparams.train_files, shuffle=True
      #num_epochs=hparams.num_epochs,
      #batch_size=hparams.train_batch_size
    )

    # Don't shuffle evaluation data
    eval_input = lambda: model.input_fn(
      hparams.eval_files,
      #batch_size=hparams.eval_batch_size,
      shuffle=False
    )

    train_spec = tf.estimator.TrainSpec(train_input,
                                      max_steps=hparams.train_steps
                                      )

    exporter = tf.estimator.FinalExporter('Bank_Custom_Estimator_Export',model.json_serving_input_fn)
    eval_spec = tf.estimator.EvalSpec(eval_input,
                                    steps=hparams.eval_steps,
                                    exporters=[exporter],
                                    name='bank-custom-eval'
                                    )

    run_config = tf.estimator.RunConfig()
    run_config = run_config.replace(model_dir=hparams.job_dir)
    LEARNING_RATE=0.0005

    print('model dir {}'.format(run_config.model_dir))
    estimator = model.build_estimator(run_config, LEARNING_RATE)
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Input Arguments
    parser.add_argument(
      '--train-files',
      help='GCS or local paths to training data',
      nargs='+',
      required=True
    )
    parser.add_argument(
      '--num-epochs',
      help="""\
      Maximum number of training data epochs on which to train.
      If both --max-steps and --num-epochs are specified,
      the training job will run for --max-steps or --num-epochs,
      whichever occurs first. If unspecified will run for --max-steps.\
      """,
      type=int, default=None
    )
    parser.add_argument(
      '--train-batch-size',
      help='Batch size for training steps',
      type=int,
      default=40
    )
    parser.add_argument(
      '--eval-batch-size',
      help='Batch size for evaluation steps',
      type=int,
      default=40
    )
    parser.add_argument(
      '--eval-files',
      help='GCS or local paths to evaluation data',
      nargs='+',
      required=True
    )
    parser.add_argument(
      '--job-dir',
      help='GCS location to write checkpoints and export models',
      required=True
    )

    # Argument to turn on all logging
    parser.add_argument(
      '--verbosity',
      choices=[
          'DEBUG',
          'ERROR',
          'FATAL',
          'INFO',
          'WARN'
      ],
      default='INFO',
    )
    # Experiment arguments
    parser.add_argument(
      '--train-steps',
      help="""\
      Steps to run the training job for. If --num-epochs is not specified,
      this must be. Otherwise the training job will run indefinitely.\
      """,
      type=int
    )
    parser.add_argument(
      '--eval-steps',
      help='Number of steps to run evalution for at each checkpoint',
      default=100,
      type=int
    )

    args = parser.parse_args()

    # Set python level verbosity
    tf.logging.set_verbosity(args.verbosity)
    # Set C++ Graph Execution level verbosity
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(
      tf.logging.__dict__[args.verbosity] / 10)

    # Run the training job
    hparams=hparam.HParams(**args.__dict__)
    run_experiment(hparams)
