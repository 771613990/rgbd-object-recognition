"""
TF records help methods.

Author: Daniel Koguciuk
"""
from __future__ import print_function

import os
import argparse
import tensorflow as tf

import utils.tfrecord_utils as tfrecord_utils


def dataset_read_tfrecords(arguments):

    # Filenames
    tfrecord_filenames = [f for f in os.listdir(arguments.tfrecords_path) if '.tfrecord' in f]
    tfrecord_filenames.sort()
    tfrecord_filepaths = [os.path.join(args.tfrecords_path, f) for f in tfrecord_filenames]

    # TFdataset
    tfdataset = tf.data.TFRecordDataset(tfrecord_filepaths)
    tfdataset = tfdataset.map(tfrecord_utils.tfexample_to_paths)
    # tfdataset = tfdataset.map(lambda a, b: tf.py_func(prepare_data_pad,
    #                                                   [a['pcd_path'], a['img_path'], b['name'], b['int']],
    #                                                   [tf.float32, tf.string, tf.string, tf.int64]))
    # tfdataset = tfdataset.map(lambda a, b: tf.py_func(tfrecord_utils.prepare_data_fps,
    #                                                   [a['pcd_path'], a['img_path'], b['name'], b['int']],
    #                                                   [tf.float32, tf.string, tf.string, tf.int64]))
    point_cloud_size = 2048
    tfdataset = tfdataset.map(lambda a, b: tf.py_func(
        tfrecord_utils.load_depth_and_create_point_cloud_data_rnd,
        [a['pcd_path'], a['img_path'], a['loc_path'], b['name'], b['int'], point_cloud_size],
        [tf.float32, tf.string, tf.string, tf.string, tf.int64]))

    # Transformations
    batch_size = 4
    tfdataset = tfdataset.batch(batch_size=batch_size, drop_remainder=True)
    tfdataset = tfdataset.shuffle(buffer_size=batch_size)

    data_iterator = tfdataset.make_one_shot_iterator()
    data_X_pcd, data_X_img, data_X_loc, y_name, y_int = data_iterator.get_next()
    #data_X, data_y = data_iterator.get_next()

    with tf.Session('') as sess:
        #x = sess.run(data_X['pcd_path'])
        x = sess.run(data_X_pcd)
        print(x.shape)
        # data = iterator.get_next()
        # x, y = sess.run(data)
        # print(y)


if __name__ == '__main__':

    # Argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--tfrecords_path', type=str, help='input path where are tfrecords', required=True)
    args = parser.parse_args()

    # Read tf records
    dataset_read_tfrecords(args)
