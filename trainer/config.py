#!/usr/bin/python
"""
Definition of all the variables
"""

from trainer.secrets import PROJECT_ID, BUCKET
import tensorflow as tf
from tensorflow_transform.tf_metadata import dataset_schema

DATA_DIR = BUCKET + '/data'
TRAIN_INPUT_DATA = DATA_DIR + '/input_data.csv'
TRAIN_OUTPUT_DATA = DATA_DIR + '/output_data.csv'
TFRECORD_DIR = BUCKET + '/tfrecords'
MODEL_DIR = BUCKET + '/model'
BATCH_SIZE = 64

INPUT_SCHEMA = dataset_schema.from_feature_spec({
    'BatchId': tf.io.FixedLenFeature(shape=[], dtype=tf.string),
    'ButterMass': tf.io.FixedLenFeature(shape=[], dtype=tf.float32),
    'ButterTemperature': tf.io.FixedLenFeature(shape=[], dtype=tf.float32),
    'SugarMass': tf.io.FixedLenFeature(shape=[], dtype=tf.float32),
    'SugarHumidity': tf.io.FixedLenFeature(shape=[], dtype=tf.float32),
    'FlourMass': tf.io.FixedLenFeature(shape=[], dtype=tf.float32),
    'FlourHumidity': tf.io.FixedLenFeature(shape=[], dtype=tf.float32),
    'HeatingTime': tf.io.FixedLenFeature(shape=[], dtype=tf.float32),
    'MixingSpeed': tf.io.FixedLenFeature(shape=[], dtype=tf.string),
    'MixingTime': tf.io.FixedLenFeature(shape=[], dtype=tf.float32),
})

OUTPUT_SCHEMA = dataset_schema.from_feature_spec({
    'BatchId': tf.io.FixedLenFeature(shape=[], dtype=tf.string),
    'TotalVolume': tf.io.FixedLenFeature(shape=[], dtype=tf.float32),
    'Density': tf.io.FixedLenFeature(shape=[], dtype=tf.float32),
    'Temperature': tf.io.FixedLenFeature(shape=[], dtype=tf.float32),
    'Humidity': tf.io.FixedLenFeature(shape=[], dtype=tf.float32),
    'Energy': tf.io.FixedLenFeature(shape=[], dtype=tf.float32),
    'Problems': tf.io.FixedLenFeature(shape=[], dtype=tf.string),
})

EXAMPLE_SCHEMA = dataset_schema.from_feature_spec({
    'ButterMass': tf.io.FixedLenFeature(shape=[], dtype=tf.float32),
    'ButterTemperature': tf.io.FixedLenFeature(shape=[], dtype=tf.float32),
    'SugarMass': tf.io.FixedLenFeature(shape=[], dtype=tf.float32),
    'SugarHumidity': tf.io.FixedLenFeature(shape=[], dtype=tf.float32),
    'FlourMass': tf.io.FixedLenFeature(shape=[], dtype=tf.float32),
    'FlourHumidity': tf.io.FixedLenFeature(shape=[], dtype=tf.float32),
    'HeatingTime': tf.io.FixedLenFeature(shape=[], dtype=tf.float32),
    'MixingSpeed': tf.io.FixedLenFeature(shape=[], dtype=tf.string),
    'MixingTime': tf.io.FixedLenFeature(shape=[], dtype=tf.float32),
    'TotalVolume': tf.io.FixedLenFeature(shape=[], dtype=tf.float32),
    'Density': tf.io.FixedLenFeature(shape=[], dtype=tf.float32),
    'Temperature': tf.io.FixedLenFeature(shape=[], dtype=tf.float32),
    'Humidity': tf.io.FixedLenFeature(shape=[], dtype=tf.float32),
    'Energy': tf.io.FixedLenFeature(shape=[], dtype=tf.float32),
    'Problems': tf.io.FixedLenFeature(shape=[], dtype=tf.string),
})
