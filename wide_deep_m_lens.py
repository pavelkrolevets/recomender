# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Example code for TensorFlow Wide & Deep Tutorial using tf.estimator API."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import shutil
import sys

import tensorflow as tf


parser = argparse.ArgumentParser()

parser.add_argument(
    '--model_dir', type=str, default='./model',
    help='Base directory for the model.')

parser.add_argument(
    '--model_type', type=str, default='wide_deep',
    help="Valid model types: {'wide', 'deep', 'wide_deep'}.")

parser.add_argument(
    '--train_epochs', type=int, default=200, help='Number of training epochs.')

parser.add_argument(
    '--epochs_per_eval', type=int, default=5,
    help='The number of training epochs to run between evaluations.')

parser.add_argument(
    '--batch_size', type=int, default=100, help='Number of examples per batch.')

parser.add_argument(
    '--train_data', type=str, default='./data/dataset_aligned.csv',
    help='Path to the training data.')

parser.add_argument(
    '--test_data', type=str, default='./data/dataset_aligned.csv',
    help='Path to the test data.')

_NUM_EXAMPLES = {
    'train': 80000,
    'validation': 20000,
}

_CSV_COLUMNS = [
    'movie_id', 'movie_title', 'release_date', 'Action', 'Adventure',
    'Animation', 'Childrens', 'Comedy', 'Crime', 'Documentary',
    'Drama', 'Fantasy', 'Film-Noir', 'Horror','Musical', 'Mystery',
    'Romance','Sci-Fi', 'Thriller', 'War', 'Western', 'user_id', 'rating', 'timestamp', 'age',
    'gender', 'occupation', 'zip_code', 'category_hash']

_CSV_COLUMN_DEFAULTS = [[''], [''], [''], [''], [''], [''], [''], [''], [''],
                        [''], [''], [''], [''], [''], [''], [''], [''], [''],
                        [''], [''], [''], [''], [0],  [''], [0],  [''], [''],
                        [''], ['']]

def build_model_columns():
  """Users"""


  # Continuous columns
  age = tf.feature_column.numeric_column('age')
  # education_num = tf.feature_column.numeric_column('education_num')
  # capital_gain = tf.feature_column.numeric_column('capital_gain')
  # capital_loss = tf.feature_column.numeric_column('capital_loss')
  # hours_per_week = tf.feature_column.numeric_column('hours_per_week')


  occupation = tf.feature_column.categorical_column_with_vocabulary_list(
      'occupation', [ 'technician', 'other' 'writer', 'executive', 'administrator', 'student',
                      'lawyer', 'educator', 'scientist', 'entertainment', 'programmer' ,'librarian',
                      'homemaker' ,'artist' ,'engineer', 'marketing', 'none' ,'healthcare','retired',
                      'salesman', 'doctor'])

  zip = tf.feature_column.categorical_column_with_hash_bucket(
      'zip_code', hash_bucket_size=1000)

  age_buckets = tf.feature_column.bucketized_column(
      age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])

  gender = tf.feature_column.categorical_column_with_vocabulary_list(
      'gender', ['M', 'F'])


  """Items"""
  movie_title = tf.feature_column.categorical_column_with_hash_bucket(
      'movie_title', hash_bucket_size=10000)
  release_date = tf.feature_column.categorical_column_with_hash_bucket(
      'release_date', hash_bucket_size=1000)
  category_hash = tf.feature_column.categorical_column_with_hash_bucket('category_hash', hash_bucket_size=10000)
  # Action = tf.feature_column.categorical_column_with_vocabulary_list(
  #     'Action', ['0', '1'])
  # Adventure = tf.feature_column.categorical_column_with_vocabulary_list(
  #     'Adventure', ['0', '1'])
  # Animation = tf.feature_column.categorical_column_with_vocabulary_list(
  #     'Animation', ['0', '1'])
  # Childrens = tf.feature_column.categorical_column_with_vocabulary_list(
  #     'Childrens', ['0', '1'])
  # Comedy = tf.feature_column.categorical_column_with_vocabulary_list(
  #     'Comedy', ['0', '1'])
  # Crime = tf.feature_column.categorical_column_with_vocabulary_list(
  #     'Crime', ['0', '1'])
  # Documentary = tf.feature_column.categorical_column_with_vocabulary_list(
  #     'Documentary', ['0', '1'])
  # Drama = tf.feature_column.categorical_column_with_vocabulary_list(
  #     'Drama', ['0', '1'])
  # Fantasy = tf.feature_column.categorical_column_with_vocabulary_list(
  #     'Fantasy', ['0', '1'])
  # Film_Noir = tf.feature_column.categorical_column_with_vocabulary_list(
  #     'Film-Noir', ['0', '1'])
  # Horror = tf.feature_column.categorical_column_with_vocabulary_list(
  #     'Horror', ['0', '1'])
  # Musical = tf.feature_column.categorical_column_with_vocabulary_list(
  #     'Musical', ['0', '1'])
  # Mystery = tf.feature_column.categorical_column_with_vocabulary_list(
  #     'Mystery', ['0', '1'])
  # Romance = tf.feature_column.categorical_column_with_vocabulary_list(
  #     'Romance', ['0', '1'])
  # Sci_Fi= tf.feature_column.categorical_column_with_vocabulary_list(
  #     'Sci-Fi', ['0', '1'])
  # Thriller = tf.feature_column.categorical_column_with_vocabulary_list(
  #     'Thriller', ['0', '1'])
  # War = tf.feature_column.categorical_column_with_vocabulary_list(
  #     'War', ['0', '1'])
  # Western = tf.feature_column.categorical_column_with_vocabulary_list(
  #     'Western', ['0', '1'])

  # Wide columns and deep columns.
  base_columns = [gender, age_buckets, occupation, category_hash]

  # crossed_columns = [
  #     tf.feature_column.crossed_column(
  #         ['education', 'occupation'], hash_bucket_size=1000),
  #     tf.feature_column.crossed_column(
  #         [age_buckets, 'education', 'occupation'], hash_bucket_size=1000),
  # ]

  wide_columns = base_columns # + crossed_columns

  deep_columns = [
      age,
      tf.feature_column.embedding_column(movie_title, dimension=32),
      tf.feature_column.embedding_column(category_hash, dimension=32),
      tf.feature_column.embedding_column(gender, dimension=32),
      tf.feature_column.embedding_column(occupation, dimension=32),
      tf.feature_column.embedding_column(zip, dimension=32),
      tf.feature_column.embedding_column(release_date, dimension=32),

      # tf.feature_column.embedding_column(Action, dimension=2),
      # tf.feature_column.embedding_column(Adventure, dimension=2),
      # tf.feature_column.embedding_column(Animation, dimension=2),
      # tf.feature_column.embedding_column(Childrens, dimension=2),
      # tf.feature_column.embedding_column(Comedy, dimension=2),
      # tf.feature_column.embedding_column(Crime, dimension=2),
      # tf.feature_column.embedding_column(Documentary, dimension=2),
      # tf.feature_column.embedding_column(Drama, dimension=2),
      # tf.feature_column.embedding_column(Fantasy, dimension=2),
      # tf.feature_column.embedding_column(Film_Noir, dimension=2),
      # tf.feature_column.embedding_column(Horror, dimension=2),
      # tf.feature_column.embedding_column(Musical, dimension=2),
      # tf.feature_column.embedding_column(Mystery, dimension=2),
      # tf.feature_column.embedding_column(Romance, dimension=2),
      # tf.feature_column.embedding_column(Sci_Fi, dimension=2),
      # tf.feature_column.embedding_column(Thriller, dimension=2),
      # tf.feature_column.embedding_column(War, dimension=2),
      # tf.feature_column.embedding_column(Western, dimension=2),
  ]

  return wide_columns, deep_columns

def build_estimator(model_dir, model_type):
  """Build an estimator appropriate for the given model type."""
  wide_columns, deep_columns = build_model_columns()
  hidden_units = [1024, 512, 256]

  # Create a tf.estimator.RunConfig to ensure the model is run on CPU, which
  # trains faster than GPU for this model.
  run_config = tf.estimator.RunConfig().replace(
      session_config=tf.ConfigProto(device_count={'GPU': 0}))

  if model_type == 'wide':
    return tf.estimator.LinearClassifier(
        model_dir=model_dir,
        feature_columns=wide_columns,
        config=run_config)
  elif model_type == 'deep':
    return tf.estimator.DNNClassifier(
        model_dir=model_dir,
        feature_columns=deep_columns,
        hidden_units=hidden_units,
        config=run_config,
        optimizer=tf.train.ProximalAdagradOptimizer(
            learning_rate=0.1,
            l1_regularization_strength=0.001)

        )
  else:
    return tf.estimator.DNNLinearCombinedClassifier(
        model_dir=model_dir,
        linear_feature_columns=wide_columns,
        dnn_feature_columns=deep_columns,
        dnn_hidden_units=hidden_units,
        config=run_config,
        dnn_optimizer= tf.train.ProximalAdagradOptimizer(learning_rate=0.1,
                                        l1_regularization_strength=0.001)
    )


def input_fn(data_file, num_epochs, shuffle, batch_size):
  """Generate an input function for the Estimator."""
  assert tf.gfile.Exists(data_file), (
      '%s not found. Please make sure you have either run data_download.py or '
      'set both arguments --train_data and --test_data.' % data_file)

  def parse_csv(value):
    print('Parsing', data_file)
    columns = tf.decode_csv(value, record_defaults=_CSV_COLUMN_DEFAULTS)
    features = dict(zip(_CSV_COLUMNS, columns))
    labels = features.pop('rating')
    return features, tf.equal(labels, 1)

  # Extract lines from input files using the Dataset API.
  dataset = tf.data.TextLineDataset(data_file)

  if shuffle:
    dataset = dataset.shuffle(buffer_size=_NUM_EXAMPLES['train'])

  dataset = dataset.map(parse_csv, num_parallel_calls=5)

  # We call repeat after shuffling, rather than before, to prevent separate
  # epochs from blending together.
  dataset = dataset.repeat(num_epochs)
  dataset = dataset.batch(batch_size)
  return dataset


def main(unused_argv):
  # Clean up the model directory if present
  shutil.rmtree(FLAGS.model_dir, ignore_errors=True)
  model = build_estimator(FLAGS.model_dir, FLAGS.model_type)

  # Train and evaluate the model every `FLAGS.epochs_per_eval` epochs.
  for n in range(FLAGS.train_epochs // FLAGS.epochs_per_eval):
    model.train(input_fn=lambda: input_fn(
        FLAGS.train_data, FLAGS.epochs_per_eval, True, FLAGS.batch_size))

    results = model.evaluate(input_fn=lambda: input_fn(
        FLAGS.test_data, 1, False, FLAGS.batch_size))

    # Display evaluation metrics
    print('Results at epoch', (n + 1) * FLAGS.epochs_per_eval)
    print('-' * 60)

    for key in sorted(results):
      print('%s: %s' % (key, results[key]))


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
