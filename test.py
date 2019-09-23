import argparse
import numpy as np
import tensorflow as tf
import socket
import importlib
import os
import sys
from tqdm import tqdm
from timeit import default_timer as timer
import utils.tfrecord_utils as tfrecord_utils

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='pointnet_cls_basic', help='Model name: pointnet_cls or pointnet_cls_basic ['
                                                                  'default: pointnet_cls]')
parser.add_argument('--log_dir', default='log', help='Log dir [default: log]')
parser.add_argument('--num_point', type=int, default=1024, help='Point Number [256/512/1024/2048] [default: 1024]')
parser.add_argument('--batch_size', type=int, default=1, help='Batch Size during training [default: 1]')
parser.add_argument('--tfrecords_path', type=str, help='top level input path where are train/test dirs with tfrecords',
                    required=True)
FLAGS = parser.parse_args()

BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
GPU_INDEX = FLAGS.gpu

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'models', FLAGS.model))

LOG_DIR = FLAGS.log_dir

# Import and backup model file
if 'inception' not in FLAGS.model:
    MODEL = importlib.import_module(FLAGS.model)
else:
    import slim.nets.inception_v3 as inception_v3

NUM_CLASSES = 51

HOSTNAME = socket.gethostname()

TFRECORDS_TRAIN_DIRPATH = os.path.join(FLAGS.tfrecords_path, 'train')
TFRECORDS_TEST_DIRPATH = os.path.join(FLAGS.tfrecords_path, 'test')


def train():
    with tf.Graph().as_default():

        #######################################################################
        # TFRecords
        #######################################################################

        # Train filenames
        tfrecord_filenames_test = [f for f in os.listdir(TFRECORDS_TEST_DIRPATH) if '.tfrecord' in f]
        tfrecord_filenames_test.sort()
        tfrecord_filepaths_test = [os.path.join(TFRECORDS_TEST_DIRPATH, f) for f in tfrecord_filenames_test]

        # TFdataset
        tfrecord_filepaths_placeholder = tf.placeholder(tf.string, [None])
        tfdataset = tf.data.TFRecordDataset(tfrecord_filepaths_placeholder)
        tfdataset = tfdataset.shuffle(buffer_size=BATCH_SIZE*10)
        tfdataset = tfdataset.map(tfrecord_utils.tfexample_to_paths, num_parallel_calls=4)

        #######################################################################
        # Unorganized point cloud
        #######################################################################

        # # Load
        # tfdataset = tfdataset.map(lambda a, b: tf.py_func(tfrecord_utils.load_depth_and_create_point_cloud_data_rnd,
        #                                                   [a['pcd_path'], a['img_path'], a['loc_path'], b['name'],
        #                                                    b['int'], NUM_POINT],
        #                                                   [tf.float32, tf.string, tf.string, tf.string, tf.int64]))

        # # Augment
        # tfdataset = tfdataset.map(lambda a, b, c, d, e:
        #                           tf.py_func(tfrecord_utils.augment_point_cloud, [a, b, c, d, e, True, True, False],
        #                                      [tf.float32, tf.string, tf.string, tf.string, tf.int64]))

        #######################################################################
        # Organized point cloud
        #######################################################################

        # Settings
        data_channels = 3
        data_height = 299
        data_width = 299

        # Load data
        tfdataset = tfdataset.map(lambda a, b: tf.py_func(tfrecord_utils.load_depth_and_create_organized_point_cloud,
                                                          [a['pcd_path'], a['img_path'], a['loc_path'], b['name'],
                                                           b['int'], data_height],
                                                          [tf.float32, tf.string, tf.string, tf.string, tf.int64]),
                                  num_parallel_calls=4)

        #######################################################################
        # Depth image
        #######################################################################

        # # Load data
        # data_channels = 1
        # data_height = 299
        # data_width = 299
        # data_scale = 1.0 # max depth from kinect is 10m, so 0.1 gives us range of 0-1
        # tfdataset = tfdataset.map(lambda a, b: tf.py_func(tfrecord_utils.load_depth,
        #                                                   [a['pcd_path'], a['img_path'], a['loc_path'], b['name'],
        #                                                    b['int'], data_height, data_scale],
        #                                                   [tf.float32, tf.string, tf.string, tf.string, tf.int64]),
        #                           num_parallel_calls=4)

        # # Augment
        # tfdataset = tfdataset.map(lambda a, b, c, d, e:
        #                           tf.py_func(tfrecord_utils.augment_depth_image, [a, b, c, d, e],
        #                                      [tf.float32, tf.string, tf.string, tf.string, tf.int64]))

        # Transformations
        tfdataset = tfdataset.batch(batch_size=BATCH_SIZE, drop_remainder=False)
        tfdataset = tfdataset.prefetch(10)

        # Iterator
        data_iterator = tfdataset.make_initializable_iterator()
        data_pcd, _, _, _, data_y_int = data_iterator.get_next()
        data_pcd = tf.reshape(data_pcd, (BATCH_SIZE, data_height, data_width, data_channels))

        #######################################################################
        # Network architecture
        #######################################################################

        with tf.device('/gpu:' + str(GPU_INDEX)):
            is_training_pl = tf.Variable(True, trainable=False, dtype=tf.bool)

            # Note the global_step=batch parameter to minimize.
            # That tells the optimizer to helpfully increment the 'batch' parameter for you every time it trains.
            batch = tf.Variable(0)

            # Get model and loss
            pred, end_points = MODEL.get_model(data_pcd, is_training_pl, num_classes=NUM_CLASSES, bn_decay=None)
            loss = MODEL.get_loss(pred, data_y_int, end_points, num_classes=NUM_CLASSES)
            tf.summary.scalar('loss', loss)

            correct = tf.equal(tf.argmax(pred, 1), tf.to_int64(data_y_int))
            accuracy = tf.reduce_sum(tf.cast(correct, tf.float32)) / float(BATCH_SIZE)
            tf.summary.scalar('accuracy', accuracy)

            # Add ops to save and restore all the variables.
            saver = tf.train.Saver()

        #######################################################################
        # Create session
        #######################################################################

        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        sess = tf.Session(config=config)

        # Add summary writers
        merged = tf.summary.merge_all()
        test_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'test'))

        # Init variables
        sess.run(tf.global_variables_initializer())

        # Restore?
        saver.restore(sess, tf.train.latest_checkpoint(LOG_DIR))

        ops = {'is_training_pl': is_training_pl, 'pred': pred,
               'loss': loss, 'merged': merged, 'step': batch,
               'data_y_int': data_y_int, 'data_pcd': data_pcd,
               }

        eval_one_epoch(sess, ops, test_writer, data_iterator, tfrecord_filepaths_test,
                       tfrecord_filepaths_placeholder)


def eval_one_epoch(sess, ops, test_writer, data_iterator, tfrecord_filepaths_test, tfrecord_filepaths_placeholder):
    """ ops: dict mapping from string to tf ops """
    total_correct = 0.
    total_seen = 0.
    loss_sum = 0.
    total_seen_class = [0. for _ in range(NUM_CLASSES)]
    total_correct_class = [0. for _ in range(NUM_CLASSES)]

    # Reset train data
    sess.run(data_iterator.initializer, feed_dict={tfrecord_filepaths_placeholder: tfrecord_filepaths_test})

    # Unset trainable weights
    sess.run(ops['is_training_pl'].assign(False))

    pbar = tqdm(desc='', unit='tick')
    try:
        while True:

            # Train it
            summary, step, loss_val, pred_val, current_label = sess.run(
                [ops['merged'], ops['step'], ops['loss'], ops['pred'], ops['data_y_int']])

            # Some acc calulation
            pred_val = np.argmax(pred_val, 1)
            correct = np.sum(pred_val == current_label)
            total_correct += correct
            total_seen += BATCH_SIZE
            loss_sum += (loss_val * BATCH_SIZE)
            for i in range(len(current_label)):
                current_class = current_label[i]
                total_seen_class[current_class] += 1
                total_correct_class[current_class] += (pred_val[i] == current_class)

            # Log info
            pbar.set_description('Mean test accuracy: {:.4f} class_accuracy: {:.4f}, and loss {:.4f}'.format(
                total_correct/float(total_seen),
                np.mean(np.array(total_correct_class) / np.array(total_seen_class, dtype=np.float)),
                loss_sum/float(total_seen)))
            pbar.update(1)
            pbar.refresh()

    except tf.errors.OutOfRangeError:
        pass
    pbar.close()

    # Log test accuracy
    summary_log = tf.Summary()
    summary_log.value.add(tag="%stest_accuracy" % "", simple_value=np.sum(total_correct/float(total_seen)))
    test_writer.add_summary(summary_log, step)


if __name__ == "__main__":
    train()
