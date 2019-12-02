import tensorflow as tf


def get_model(depth_image, is_training, num_classes, bn_decay=None, with_bn=False):

    ###########################################################################
    # Block #1
    ###########################################################################

    # Convolutional Layer #1    -   14 208 params
    conv1 = tf.layers.conv2d(inputs=depth_image, filters=32, kernel_size=7, strides=2, padding="valid", activation=None)
    # Batch norm #1
    if with_bn:
        conv1 = tf.layers.batch_normalization(inputs=conv1, axis=-1, momentum=0.9, epsilon=0.001, center=True,
                                              scale=True, training=is_training, name='conv1_bn')
    # Activation #1
    conv1 = tf.nn.relu(conv1)
    # Max-pooling #1
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=3, strides=2)

    ###########################################################################
    # Block #2
    ###########################################################################

    # Convolutional Layer #2    -   230 496 params
    conv2 = tf.layers.conv2d(inputs=pool1, filters=64, kernel_size=5, strides=2, padding="same", activation=None)
    # Batch norm #2
    if with_bn:
        conv2 = tf.layers.batch_normalization(inputs=conv2, axis=-1, momentum=0.999, epsilon=0.001, center=True,
                                              scale=True, training=is_training, name='conv2_bn')
    # Activation #2
    conv2 = tf.nn.relu(conv2)
    # Max-pooling #2
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=3, strides=2)

    ###########################################################################
    # Block #3
    ###########################################################################

    # Convolutional Layers #3   -   83 040 params
    conv3 = tf.layers.conv2d(inputs=pool2, filters=96, kernel_size=3, strides=1, padding="same", activation=None)
    # Batch norm #3
    if with_bn:
        conv3 = tf.layers.batch_normalization(inputs=conv3, axis=-1, momentum=0.999, epsilon=0.001, center=True,
                                              scale=True, training=is_training, name='conv3_bn')
    # Activation #3
    conv3 = tf.nn.relu(conv3)

    ###########################################################################
    # Block #4
    ###########################################################################

    # Convolutional Layers #4   -   83 040 params
    conv4 = tf.layers.conv2d(inputs=conv3, filters=96, kernel_size=3, strides=1, padding="same", activation=None)
    # Batch norm #3
    if with_bn:
        conv4 = tf.layers.batch_normalization(inputs=conv4, axis=-1, momentum=0.999, epsilon=0.001, center=True,
                                              scale=True, training=is_training, name='conv4_bn')
    # Activation #3
    conv4 = tf.nn.relu(conv4)

    ###########################################################################
    # Block #5
    ###########################################################################

    # Convolutional Layers #5   -   83 040 params
    conv5 = tf.layers.conv2d(inputs=conv4, filters=64, kernel_size=3, strides=1, padding="same", activation=None)
    # Batch norm #5
    if with_bn:
        conv5 = tf.layers.batch_normalization(inputs=conv5, axis=-1, momentum=0.999, epsilon=0.001, center=True,
                                              scale=True, training=is_training, name='conv5_bn')
    # Activation #5
    conv5 = tf.nn.relu(conv5)
    # Max-pooling #5
    pool5 = tf.layers.max_pooling2d(inputs=conv5, pool_size=3, strides=2)

    ###########################################################################
    # Dense #1
    ###########################################################################

    # Flat layer
    pool5_flat = tf.layers.flatten(pool5)
    # Dense layer #6
    dense6 = tf.layers.dense(inputs=pool5_flat, units=512, activation=None)
    # Batch norm #6
    if with_bn:
        dense6 = tf.layers.batch_normalization(inputs=dense6, axis=-1, momentum=0.999, epsilon=0.001, center=True,
                                              scale=True, training=is_training, name='dense6_bn')
    # Activation #6
    dense6 = tf.nn.relu(dense6)
    # Dropout #6
    dropout6 = tf.layers.dropout(inputs=dense6, rate=0.5)

    ###########################################################################
    # Dense #1
    ###########################################################################

    # Dense layer #7
    dense7 = tf.layers.dense(inputs=dropout6, units=128, activation=None)
    # Batch norm #7
    if with_bn:
        dense7 = tf.layers.batch_normalization(inputs=dense7, axis=-1, momentum=0.999, epsilon=0.001, center=True,
                                              scale=True, training=is_training, name='dense7_bn')
    # Activation #7
    dense7 = tf.nn.relu(dense7)
    # Dropout #7
    dropout7 = tf.layers.dropout(inputs=dense7, rate=0.5)

    ###########################################################################
    # Dense #3
    ###########################################################################

    dense8 = tf.layers.dense(inputs=dropout7, units=num_classes)

    # Return
    return dense8, None


def get_loss(pred, label, end_points, num_classes):
    """ pred: B*NUM_CLASSES,
        label: B, """
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=label)
    classify_loss = tf.reduce_mean(loss)
    tf.summary.scalar('classify loss', classify_loss)
    return classify_loss