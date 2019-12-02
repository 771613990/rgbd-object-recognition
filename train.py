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
from tensorflow.python.client import timeline


def log_string(out_str):
    #LOG_FOUT.write(out_str + '\n')
    #LOG_FOUT.flush()
    print(out_str)


def get_learning_rate(batch, batch_size,
                      base_learning_rate, learning_rate_decay_rate, learning_rate_decay_step):
    learning_rate = tf.train.exponential_decay(base_learning_rate,  # Base learning rate.
                                               batch * batch_size,  # Current index into the dataset.
                                               learning_rate_decay_step,  # Decay step.
                                               learning_rate_decay_rate,  # Decay rate.
                                               staircase=True)
    learning_rate = tf.maximum(learning_rate, 0.00001)  # CLIP THE LEARNING RATE!
    return learning_rate


def get_bn_decay(batch, batch_size,
                 bn_init_decay, bn_decay_decay_rate,
                 bn_decay_decay_step, bn_decay_clip):
    bn_momentum = tf.train.exponential_decay(bn_init_decay, batch * batch_size, bn_decay_decay_step,
                                             bn_decay_decay_rate, staircase=True)
    bn_decay = tf.minimum(bn_decay_clip, 1 - bn_momentum)
    return bn_decay


def train(tfrecords_path,
          model_name,
          batch_size=16, base_learning_rate=0.001, learning_rate_decay_rate=0.8,
          gpu_index=0, optimizer_name='adam', log_dir='log', epochs=100,
          bn_init_decay=0.5, bn_decay_decay_rate=0.5, bn_decay_decay_step=200000, bn_decay_clip=0.99,
          pretrained_weights_file_path=None):
    """
    Args:
        tfrecords_path (str): Top level input path where are train/test dirs with tfrecords.
        batch_size (int): Batch Size during training [default: 32].
        base_learning_rate (float): Initial learning rate [default: 0.001].
        gpu_index (int): GPU index to use [default: GPU 0].

    """

    # TFRecords paths
    NUM_CLASSES = 51
    TFRECORDS_TRAIN_DIRPATH = os.path.join(tfrecords_path, 'train')
    TFRECORDS_TEST_DIRPATH = os.path.join(tfrecords_path, 'test')

    # Import model module
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(BASE_DIR)
    sys.path.append(os.path.join(BASE_DIR, 'models', model_name))
    MODEL = importlib.import_module(model_name)

    # Log dir
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
        os.mkdir(os.path.join(log_dir, 'timeline'))

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
        tfdataset = tfdataset.shuffle(buffer_size=batch_size*10)   # Only one tfrecord file -> no action
        tfdataset = tfdataset.map(tfrecord_utils.tfexample_to_depth_image, num_parallel_calls=4)

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
        data_height = 224
        data_width = 224
        zero_mean = True
        unit_ball = True

        # # Load data
        # tfdataset = tfdataset.map(lambda a, b: tf.py_func(tfrecord_utils.load_depth_and_create_organized_point_cloud,
        #                                                   [a['pcd_path'], a['img_path'], a['loc_path'], b['name'],
        #                                                    b['int'], data_height, zero_mean, unit_ball],
        #                                                   [tf.float32, tf.string, tf.string, tf.string, tf.int64]),
        #                           num_parallel_calls=4)

        # Load data
        tfdataset = tfdataset.map(lambda a, b: tf.py_func(tfrecord_utils.create_organized_point_cloud,
                                                          [a['depth-image'], a['depth-image-loc'], b['name'], b['int'],
                                                           data_height, zero_mean, unit_ball], [tf.float32, tf.int64,
                                                                                                tf.string, tf.int64]),
                                  num_parallel_calls=4)

        #######################################################################
        # Depth image
        #######################################################################

        # # Load data
        # data_channels = 1
        # data_height = 299
        # data_width = 299
        # data_scale = 1.0  # max depth from kinect is 10m, so 0.1 gives us range of 0-1
        # # data_mean = 775.6092    # None if zero, sample specific if below zero, given value otherwise
        # # data_std = 499.1676     # None if zero, sample specific if below zero, given value otherwise
        # data_mean = 775.6092 - 499.1676  # To be in the range of 0-1
        # data_std = 499.1676 * 2  # To be in the range of 0-1
        # tfdataset = tfdataset.map(lambda a, b: tf.py_func(tfrecord_utils.load_depth,
        #                                                   [a['pcd_path'], a['img_path'], a['loc_path'], b['name'],
        #                                                    b['int'], data_height, data_scale, data_mean, data_std],
        #                                                   [tf.float32, tf.string, tf.string, tf.string, tf.int64]),
        #                           num_parallel_calls=4)

        # # Tile
        # if data_channels == 3:
        #     tfdataset = tfdataset.map(lambda a, b, c, d, e:
        #                               tf.py_func(tfrecord_utils.tile_depth_image, [a, b, c, d, e],
        #                                          [tf.float32, tf.string, tf.string, tf.string, tf.int64]))

        # # Augment
        # tfdataset = tfdataset.map(lambda a, b, c, d, e:
        #                           tf.py_func(tfrecord_utils.augment_depth_image, [a, b, c, d, e],
        #                                      [tf.float32, tf.string, tf.string, tf.string, tf.int64]))

        # Transformations
        tfdataset = tfdataset.shuffle(buffer_size=batch_size * 2)
        tfdataset = tfdataset.batch(batch_size=batch_size, drop_remainder=True)
        tfdataset = tfdataset.prefetch(10)

        # Iterator
        data_iterator = tfdataset.make_initializable_iterator()
        data_pcd, _, _, data_y_int = data_iterator.get_next()
        data_pcd = tf.reshape(data_pcd, (batch_size, data_height, data_width, data_channels))

        #######################################################################
        # Network architecture
        #######################################################################

        with tf.device('/gpu:' + str(gpu_index)):
            is_training_pl = tf.Variable(True, trainable=False, dtype=tf.bool)

            # Note the global_step=batch parameter to minimize.
            # That tells the optimizer to helpfully increment the 'batch' parameter for you every time it trains.
            batch = tf.Variable(0, trainable=False)
            bn_decay = get_bn_decay(batch, batch_size=batch_size,
                                    bn_init_decay=bn_init_decay, bn_decay_decay_rate=bn_decay_decay_rate,
                                    bn_decay_decay_step=bn_decay_decay_step, bn_decay_clip=bn_decay_clip)
            tf.summary.scalar('bn_decay', bn_decay)

            # Get model and loss
            pred, end_points = MODEL.get_model(data_pcd, is_training_pl, num_classes=NUM_CLASSES,
                                               bn_decay=bn_decay, with_bn=False)
            loss = MODEL.get_loss(pred, data_y_int, end_points, num_classes=NUM_CLASSES)

            # # Number of trainable weights
            # trainable_weights_no = np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])
            # print('trainable_weights_no: {}'.format(trainable_weights_no))
            # exit(0)

            tf.summary.scalar('loss', loss)
            correct = tf.equal(tf.argmax(pred, 1), tf.to_int64(data_y_int))
            accuracy = tf.reduce_sum(tf.cast(correct, tf.float32)) / float(batch_size)
            tf.summary.scalar('accuracy', accuracy)

            # Get learning rate
            learning_rate = get_learning_rate(batch, batch_size=batch_size,
                                              base_learning_rate=base_learning_rate,
                                              learning_rate_decay_rate=learning_rate_decay_rate,
                                              learning_rate_decay_step=bn_decay_decay_step)
            tf.summary.scalar('learning_rate', learning_rate)

            # OPTIMIZATION - Also updates batchnorm operations automatically
            with tf.variable_scope('opt') as scope:
                if optimizer_name == 'momentum':
                    optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=MOMENTUM)
                elif optimizer_name == 'adam':
                    optimizer = tf.train.AdamOptimizer(learning_rate)
                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)  # for batchnorm
                with tf.control_dependencies(update_ops):
                    train_op = optimizer.minimize(loss, global_step=batch)

            # Load weghts from checkpoint
            if model_name == 'inception_v3' and pretrained_weights_file_path is not None:
                # Lists of scopes of weights to include/exclude from pretrained snapshot
                pretrained_include = ["InceptionV3"]
                pretrained_exclude = ["InceptionV3/AuxLogits", "InceptionV3/Logits"]

                # PRETRAINED SAVER - For loading pretrained weights on the first run
                pretrained_vars = tf.contrib.framework.get_variables_to_restore(include=pretrained_include,
                                                                                exclude=pretrained_exclude)
                tf_pretrained_saver = tf.train.Saver(pretrained_vars, name="pretrained_saver")

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
        train_writer = tf.summary.FileWriter(os.path.join(log_dir, 'train'), sess.graph)
        test_writer = tf.summary.FileWriter(os.path.join(log_dir, 'test'))

        # Init variables
        sess.run(tf.global_variables_initializer())

        # Profiling
        sess_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        sess_metadata = tf.RunMetadata()

        # print('VARS NO: {}'.format(np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])))
        # exit()

        # Restore?
        # saver.restore(sess, tf.train.latest_checkpoint('log'))

        if model_name == 'inception_v3' and pretrained_weights_file_path is not None:
            sess.run(tf.global_variables_initializer())
            tf_pretrained_saver.restore(sess, pretrained_weights_file_path)

        ops = {'is_training_pl': is_training_pl, 'pred': pred,
               'loss': loss, 'train_op': train_op, 'merged': merged, 'step': batch,
               # 'pointclouds_pl': pointclouds_pl, 'labels_pl': labels_pl,
               'data_y_int': data_y_int, 'data_pcd': data_pcd,
               'batch_size': batch_size, 'num_classes': NUM_CLASSES,
               'sess_options': sess_options, 'sess_metadata': sess_metadata,
               'log_dir': log_dir,
               }

        for epoch in range(epochs):
            log_string('**** EPOCH %03d ****' % epoch)
            sys.stdout.flush()

            train_one_epoch(sess, ops, train_writer, data_iterator, tfrecord_filepaths_train,
                            tfrecord_filepaths_placeholder)
            eval_one_epoch(sess, ops, test_writer, data_iterator, tfrecord_filepaths_test,
                           tfrecord_filepaths_placeholder)

            # Save the variables to disk.
            save_path = saver.save(sess, os.path.join(log_dir, "model.ckpt"))
            log_string("Model saved in file: %s" % save_path)


def train_one_epoch(sess, ops, train_writer, data_iterator, tfrecord_filepaths_train,
                    tfrecord_filepaths_placeholder):
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
                [ops['merged'], ops['step'], ops['train_op'], ops['loss'], ops['pred'], ops['data_y_int']],
                options=ops['sess_options'], run_metadata=ops['sess_metadata'])
            batch_train_end = timer()

            # Profiling
            fetched_timeline = timeline.Timeline(ops['sess_metadata'].step_stats)
            chrome_trace = fetched_timeline.generate_chrome_trace_format()
            with open(os.path.join(ops['log_dir'], 'timeline', 'timeline_02_step_%d.json' % step), 'w') as f:
                f.write(chrome_trace)

            # Print predited value and label
            # print('pred_val: {} curr_lab: {}'.format(pred_val[0], current_label[0]))

            # Some acc calulation
            train_writer.add_summary(summary, step)
            pred_val = np.argmax(pred_val, 1)
            correct = np.sum(pred_val == current_label)
            total_correct += correct
            total_seen += ops['batch_size']
            loss_sum += loss_val

            # Log info
            desc = 'Mean train accuracy: {:.4f} loss: {:.4f} batch train accuracy: {:.4f} batch time: {:.4f}'
            pbar.set_description(desc.format(total_correct / float(total_seen), loss_val, correct / float(ops['batch_size']),
                                             batch_train_end - batch_train_start))
            pbar.update(1)
            pbar.refresh()

    except tf.errors.OutOfRangeError:
        pass
    pbar.close()
    log_string('Mean train accuracy: {:.4f}'.format(total_correct / float(total_seen)))


def eval_one_epoch(sess, ops, test_writer, data_iterator, tfrecord_filepaths_test, tfrecord_filepaths_placeholder):
    """ ops: dict mapping from string to tf ops """
    total_correct = 0.
    total_seen = 0.
    loss_sum = 0.
    total_seen_class = [0. for _ in range(ops['num_classes'])]
    total_correct_class = [0. for _ in range(ops['num_classes'])]

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

            # print('pred_val: {} curr_lab: {}'.format(pred_val[0], current_label[0]))

            # Some acc calulation
            pred_val = np.argmax(pred_val, 1)
            correct = np.sum(pred_val == current_label)
            total_correct += correct
            total_seen += ops['batch_size']
            loss_sum += (loss_val * ops['batch_size'])
            for i in range(len(current_label)):
                current_class = current_label[i]
                total_seen_class[current_class] += 1
                total_correct_class[current_class] += (pred_val[i] == current_class)

            # Log info
            pbar.set_description('Mean test accuracy: {:.4f} class_accuracy: {:.4f}, and loss {:.4f}'.format(
                total_correct / float(total_seen),
                np.mean(np.array(total_correct_class) / np.array(total_seen_class, dtype=np.float)),
                loss_sum / float(total_seen)))
            pbar.update(1)
            pbar.refresh()

    except tf.errors.OutOfRangeError:
        pass
    pbar.close()
    log_string('Mean test accuracy: {:.4f}'.format(total_correct / float(total_seen)))

    # Log test accuracy
    summary_log = tf.Summary()
    summary_log.value.add(tag="%stest_accuracy" % "", simple_value=np.sum(total_correct / float(total_seen)))
    test_writer.add_summary(summary_log, step)


if __name__ == "__main__":

    ###########################################################################
    # Argparse
    ###########################################################################

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
    parser.add_argument('--model', help='Model name', required=True)
    parser.add_argument('--log_dir', default='log', help='Log dir [default: log]')
    parser.add_argument('--num_point', type=int, default=1024, help='Point Number [256/512/1024/2048] [default: 1024]')
    parser.add_argument('--max_epoch', type=int, default=100, help='Epoch to run [default: 100]')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch Size during training [default: 32]')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
    parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
    parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
    parser.add_argument('--decay_step', type=int, default=200000, help='Decay step for lr decay [default: 200000]')
    parser.add_argument('--decay_rate', type=float, default=0.8, help='Decay rate for lr decay [default: 0.8]')
    parser.add_argument('--tfrecords_path', type=str,
                        help='top level input path where are train/test dirs with tfrecords',
                        required=True)
    parser.add_argument('--pretrained_weights', type=str, help='pretrained weights file for inception_v3',
                        required=False)
    args = parser.parse_args()

    ###########################################################################
    # Argparse
    ###########################################################################

    train(tfrecords_path=args.tfrecords_path,
          model_name=args.model,
          batch_size=args.batch_size, base_learning_rate=args.learning_rate,
          gpu_index=args.gpu,
          bn_decay_decay_step=args.decay_step,
          optimizer_name=args.optimizer,
          log_dir=args.log_dir, epochs=args.max_epoch,
          pretrained_weights_file_path=args.pretrained_weights)
