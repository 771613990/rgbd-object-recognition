import tensorflow as tf


def get_model(depth_image, is_training, num_classes, bn_decay=None):

    # Variables
    image_size = 224, 224

    # Input Layer
    #input_layer = tf.reshape(point_cloud, [None, None, None, 1]) # Batch * height * width, 1 depth channel

    # Convolutional Layer #1 and Pooling Layer #1
    conv1 = tf.layers.conv2d(inputs=depth_image, filters=96, kernel_size=[7, 7], padding="same", activation=tf.nn.relu)
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    # Convolutional Layer #2 and Pooling Layer #2
    conv2 = tf.layers.conv2d(inputs=pool1, filters=96, kernel_size=[5, 5], padding="same", activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    # Convolutional Layers #3,4,5
    conv3 = tf.layers.conv2d(inputs=pool2, filters=128, kernel_size=[3, 3], strides=2, padding="same", activation=tf.nn.relu)
    conv4 = tf.layers.conv2d(inputs=conv3, filters=128, kernel_size=[3, 3], strides=2, padding="same", activation=tf.nn.relu)
    conv5 = tf.layers.conv2d(inputs=conv4, filters=128, kernel_size=[3, 3], strides=2, padding="same", activation=tf.nn.relu)
    pool5 = tf.layers.max_pooling2d(inputs=conv5, pool_size=[2, 2], strides=2)

    # Dense Layers
    pool5_flat = tf.layers.flatten(pool5)
    dense6 = tf.layers.dense(inputs=pool5_flat, units=1024, activation=tf.nn.relu)
    # dropout = tf.layers.dropout(inputs=dense, rate=0.4)
    dense7 = tf.layers.dense(inputs=dense6, units=512, activation=tf.nn.relu)
    # dropout = tf.layers.dropout(inputs=dense, rate=0.4)
    dense8 = tf.layers.dense(inputs=dense7, units=num_classes)

    # Return
    return dense8, None


def get_loss(pred, label, end_points, num_classes):
    """ pred: B*NUM_CLASSES,
        label: B, """
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=label)
    classify_loss = tf.reduce_mean(loss)
    tf.summary.scalar('classify loss', classify_loss)
    return classify_loss