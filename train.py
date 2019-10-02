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
parser.add_argument('--max_epoch', type=int, default=100, help='Epoch to run [default: 100]')
parser.add_argument('--batch_size', type=int, default=32, help='Batch Size during training [default: 32]')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--decay_step', type=int, default=200000, help='Decay step for lr decay [default: 200000]')
parser.add_argument('--decay_rate', type=float, default=0.8, help='Decay rate for lr decay [default: 0.8]')
parser.add_argument('--tfrecords_path', type=str, help='top level input path where are train/test dirs with tfrecords',
                    required=True)
FLAGS = parser.parse_args()

BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
MAX_EPOCH = FLAGS.max_epoch
BASE_LEARNING_RATE = FLAGS.learning_rate
GPU_INDEX = FLAGS.gpu
MOMENTUM = FLAGS.momentum
OPTIMIZER = FLAGS.optimizer
DECAY_STEP = FLAGS.decay_step
DECAY_RATE = FLAGS.decay_rate

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'models', FLAGS.model))

LOG_DIR = FLAGS.log_dir
if not os.path.exists(LOG_DIR):
    os.mkdir(LOG_DIR)

# Import and backup model file
if 'inception' not in FLAGS.model:
    MODEL = importlib.import_module(FLAGS.model)  # import network module
    MODEL_FILE = os.path.join(BASE_DIR, 'models', FLAGS.model, FLAGS.model + '.py')
    os.system('cp %s %s' % (MODEL_FILE, LOG_DIR))  # bkp of model def
else:
    import slim.nets.inception_v3 as inception_v3

os.system('cp train.py %s' % LOG_DIR)  # bkp of train procedure
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
LOG_FOUT.write(str(FLAGS) + '\n')

NUM_CLASSES = 51

BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99

HOSTNAME = socket.gethostname()

TFRECORDS_TRAIN_DIRPATH = os.path.join(FLAGS.tfrecords_path, 'train')
TFRECORDS_TEST_DIRPATH = os.path.join(FLAGS.tfrecords_path, 'test')


def log_string(out_str):
    LOG_FOUT.write(out_str + '\n')
    LOG_FOUT.flush()
    print(out_str)


def get_learning_rate(batch):
    learning_rate = tf.train.exponential_decay(BASE_LEARNING_RATE,  # Base learning rate.
                                               batch * BATCH_SIZE,  # Current index into the dataset.
                                               DECAY_STEP,  # Decay step.
                                               DECAY_RATE,  # Decay rate.
                                               staircase=True)
    learning_rate = tf.maximum(learning_rate, 0.00001)  # CLIP THE LEARNING RATE!
    return learning_rate


def get_bn_decay(batch):
    bn_momentum = tf.train.exponential_decay(BN_INIT_DECAY, batch * BATCH_SIZE, BN_DECAY_DECAY_STEP,
                                             BN_DECAY_DECAY_RATE, staircase=True)
    bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
    return bn_decay


def train():
    with tf.Graph().as_default():

        #######################################################################
        # TFRecords
        #######################################################################

        # Train filenames
        tfrecord_filenames_train = [f for f in os.listdir(TFRECORDS_TRAIN_DIRPATH) if '.tfrecord' in f]
        tfrecord_filenames_train.sort()
        tfrecord_filepaths_train = [os.path.join(TFRECORDS_TRAIN_DIRPATH, f) for f in tfrecord_filenames_train]

        # Train filenames
        tfrecord_filenames_test = [f for f in os.listdir(TFRECORDS_TEST_DIRPATH) if '.tfrecord' in f]
        tfrecord_filenames_test.sort()
        tfrecord_filepaths_test = [os.path.join(TFRECORDS_TEST_DIRPATH, f) for f in tfrecord_filenames_test]

        # TFdataset
        tfrecord_filepaths_placeholder = tf.placeholder(tf.string, [None])
        tfdataset = tf.data.TFRecordDataset(tfrecord_filepaths_placeholder)
        # tfdataset = tfdataset.shuffle(buffer_size=BATCH_SIZE*10)   Only one tfrecord file -> no action
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

        # # Settings
        # data_channels = 3
        # data_height = 299
        # data_width = 299
        # zero_mean = True
        # unit_ball = True
        #
        # # Load data
        # tfdataset = tfdataset.map(lambda a, b: tf.py_func(tfrecord_utils.load_depth_and_create_organized_point_cloud,
        #                                                   [a['pcd_path'], a['img_path'], a['loc_path'], b['name'],
        #                                                    b['int'], data_height, zero_mean, unit_ball],
        #                                                   [tf.float32, tf.string, tf.string, tf.string, tf.int64]),
        #                           num_parallel_calls=4)

        #######################################################################
        # Depth image
        #######################################################################

        # Load data
        data_channels = 1
        data_height = 299
        data_width = 299
        data_scale = 1.0 # max depth from kinect is 10m, so 0.1 gives us range of 0-1
        data_mean = 775.6092    # None if zero, sample specific if below zero, given value otherwise
        data_std = 499.1676     # None if zero, sample specific if below zero, given value otherwise
        tfdataset = tfdataset.map(lambda a, b: tf.py_func(tfrecord_utils.load_depth,
                                                          [a['pcd_path'], a['img_path'], a['loc_path'], b['name'],
                                                           b['int'], data_height, data_scale, data_mean, data_std],
                                                          [tf.float32, tf.string, tf.string, tf.string, tf.int64]),
                                  num_parallel_calls=4)

        # # Augment
        # tfdataset = tfdataset.map(lambda a, b, c, d, e:
        #                           tf.py_func(tfrecord_utils.augment_depth_image, [a, b, c, d, e],
        #                                      [tf.float32, tf.string, tf.string, tf.string, tf.int64]))

        # Transformations
        tfdataset = tfdataset.shuffle(buffer_size=BATCH_SIZE*2)
        tfdataset = tfdataset.batch(batch_size=BATCH_SIZE, drop_remainder=True)
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
            bn_decay = get_bn_decay(batch)
            tf.summary.scalar('bn_decay', bn_decay)

            # Get model and loss
            if 'inception' not in FLAGS.model:
                pred, end_points = MODEL.get_model(data_pcd, is_training_pl, num_classes=NUM_CLASSES, bn_decay=bn_decay)
                loss = MODEL.get_loss(pred, data_y_int, end_points, num_classes=NUM_CLASSES)
            else:
                pred, end_points = inception_v3.inception_v3(inputs=data_pcd, num_classes=NUM_CLASSES)
                batch_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=data_y_int)
                loss = tf.reduce_mean(batch_loss)

            tf.summary.scalar('loss', loss)
            correct = tf.equal(tf.argmax(pred, 1), tf.to_int64(data_y_int))
            accuracy = tf.reduce_sum(tf.cast(correct, tf.float32)) / float(BATCH_SIZE)
            tf.summary.scalar('accuracy', accuracy)

            # Get training operator
            learning_rate = get_learning_rate(batch)
            tf.summary.scalar('learning_rate', learning_rate)
            if OPTIMIZER == 'momentum':
                optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=MOMENTUM)
            elif OPTIMIZER == 'adam':
                optimizer = tf.train.AdamOptimizer(learning_rate)
            train_op = optimizer.minimize(loss, global_step=batch)

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
        train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'train'), sess.graph)
        test_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'test'))

        # Init variables
        sess.run(tf.global_variables_initializer())

        # Restore?
        # saver.restore(sess, tf.train.latest_checkpoint('log'))

        ops = {'is_training_pl': is_training_pl, 'pred': pred,
               'loss': loss, 'train_op': train_op, 'merged': merged, 'step': batch,
               # 'pointclouds_pl': pointclouds_pl, 'labels_pl': labels_pl,
               'data_y_int': data_y_int, 'data_pcd': data_pcd,
               }

        for epoch in range(MAX_EPOCH):

            log_string('**** EPOCH %03d ****' % epoch)
            sys.stdout.flush()

            train_one_epoch(sess, ops, train_writer, data_iterator, tfrecord_filepaths_train,
                            tfrecord_filepaths_placeholder)
            eval_one_epoch(sess, ops, test_writer, data_iterator, tfrecord_filepaths_test,
                           tfrecord_filepaths_placeholder)

            # Save the variables to disk.
            save_path = saver.save(sess, os.path.join(LOG_DIR, "model.ckpt"))
            log_string("Model saved in file: %s" % save_path)


def train_one_epoch(sess, ops, train_writer, data_iterator, tfrecord_filepaths_train, tfrecord_filepaths_placeholder):
    """ ops: dict mapping from string to tf ops """

    # Iterate over the all datapoints
    total_correct = 0.
    total_seen = 0.
    loss_sum = 0.

    # Reset train data
    sess.run(data_iterator.initializer, feed_dict={tfrecord_filepaths_placeholder: tfrecord_filepaths_train})

    # Set trainable weights
    sess.run(ops['is_training_pl'].assign(True))

    pbar = tqdm(desc='', unit='tick')
    try:
        while True:

            # Train it
            batch_train_start = timer()
            summary, step, _, loss_val, pred_val, current_label = sess.run(
                [ops['merged'], ops['step'], ops['train_op'], ops['loss'], ops['pred'], ops['data_y_int']])
            batch_train_end = timer()

            # Some acc calulation
            train_writer.add_summary(summary, step)
            pred_val = np.argmax(pred_val, 1)
            correct = np.sum(pred_val == current_label)
            total_correct += correct
            total_seen += BATCH_SIZE
            loss_sum += loss_val

            # Log info
            desc = 'Mean train accuracy: {:.4f} loss: {:.4f} batch train accuracy: {:.4f} batch time: {:.4f}'
            pbar.set_description(desc.format(total_correct/float(total_seen), loss_val, correct/float(BATCH_SIZE),
                                             batch_train_end-batch_train_start))
            pbar.update(1)
            pbar.refresh()

    except tf.errors.OutOfRangeError:
        pass
    pbar.close()


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
    LOG_FOUT.close()
