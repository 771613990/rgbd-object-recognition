import os
import sys
import tensorflow as tf

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
import tf_util
from transform_nets import input_transform_net


def placeholder_inputs(batch_size, num_point):
    pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, 3))
    labels_pl = tf.placeholder(tf.int32, shape=batch_size)
    return pointclouds_pl, labels_pl


def get_model(point_cloud, is_training, num_classes, bn_decay=None):
    """ Classification PointNet, input is BxNx3, output Bx40 """
    batch_size = point_cloud.get_shape()[0].value
    # num_point = point_cloud.get_shape()[1].value
    end_points = {}
    k = 20

    # pairwise distance of the points in the point cloud
    adj_matrix = tf_util.pairwise_distance(point_cloud)

    # get indices of k nearest neighbors
    nn_idx = tf_util.knn(adj_matrix, k=k)

    # edge feature
    edge_feature = tf_util.get_edge_feature(point_cloud, nn_idx=nn_idx, k=k)

    # transform net 1
    with tf.variable_scope('transform_net1') as _:
        transform = input_transform_net(edge_feature, is_training, bn_decay, K=3)

    # point cloud transf
    point_cloud_transformed = tf.matmul(point_cloud, transform)

    # pairwise distance of the points in the point cloud
    adj_matrix = tf_util.pairwise_distance(point_cloud_transformed)

    # get indices of k nearest neighbors
    nn_idx = tf_util.knn(adj_matrix, k=k)

    # I've got neighbors indices and subregion index (0-7)
    edge_feature = tf_util.get_edge_feature(point_cloud_transformed, nn_idx=nn_idx, k=k)

    #  net = tf_util.conv2d_reg(point_cloud_transformed, nn_idx,
    net = tf_util.conv2d(edge_feature,
                         64, [1, 1], padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training,
                         scope='dgcnn1', bn_decay=bn_decay)

    # Maxpool
    net = tf.reduce_max(net, axis=-2, keep_dims=True)
    net_1 = net

    #############################################################################
    # 2nd block
    #############################################################################

    adj_matrix = tf_util.pairwise_distance(net)
    nn_idx = tf_util.knn(adj_matrix, k=k)
    edge_feature = tf_util.get_edge_feature(net, nn_idx=nn_idx, k=k)

    net = tf_util.conv2d(edge_feature, 64, [1, 1],
                         padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training,
                         scope='dgcnn2', bn_decay=bn_decay)
    # net = tf.reduce_max(net, axis=-2, keep_dims=False)
    net = tf.reduce_max(net, axis=-2, keep_dims=True)
    net_2 = net

    #############################################################################
    # 3rd block
    #############################################################################

    adj_matrix = tf_util.pairwise_distance(net)
    nn_idx = tf_util.knn(adj_matrix, k=k)
    edge_feature = tf_util.get_edge_feature(net, nn_idx=nn_idx, k=k)

    net = tf_util.conv2d(edge_feature, 64, [1, 1],
                         padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training,
                         scope='dgcnn3', bn_decay=bn_decay)
    net = tf.reduce_max(net, axis=-2, keep_dims=True)
    net_3 = net

    #############################################################################
    # 4rd block
    #############################################################################

    adj_matrix = tf_util.pairwise_distance(net)
    nn_idx = tf_util.knn(adj_matrix, k=k)
    edge_feature = tf_util.get_edge_feature(net, nn_idx=nn_idx, k=k)

    net = tf_util.conv2d(edge_feature, 128, [1, 1],
                         padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training,
                         scope='dgcnn4', bn_decay=bn_decay)
    net = tf.reduce_max(net, axis=-2, keep_dims=True)
    net_4 = net

    #############################################################################
    # aggregate block
    #############################################################################

    net = tf.concat([net_1, net_2, net_3, net_4], axis=-1)
    net = tf_util.conv2d(net,
                         1024, [1, 1],
                         padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training,
                         scope='agg', bn_decay=bn_decay)
    net = tf.reduce_max(net, axis=1, keep_dims=True)

    # MLP on global point cloud vector
    net = tf.reshape(net, [batch_size, -1])
    net = tf_util.fully_connected(net, 512, bn=True, is_training=is_training,
                                  scope='fc1', bn_decay=bn_decay)
    net = tf_util.dropout(net, keep_prob=0.5, is_training=is_training,
                          scope='dp1')
    net = tf_util.fully_connected(net, 256, bn=True, is_training=is_training,
                                  scope='fc2', bn_decay=bn_decay)
    net = tf_util.dropout(net, keep_prob=0.5, is_training=is_training,
                          scope='dp2')
    net = tf_util.fully_connected(net, num_classes, activation_fn=None, scope='fc3')

    return net, end_points


def get_loss(pred, label, end_points, num_classes):
    """ pred: B*NUM_CLASSES,
      label: B, """
    labels = tf.one_hot(indices=label, depth=num_classes)
    loss = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=pred, label_smoothing=0.2)
    classify_loss = tf.reduce_mean(loss)
    return classify_loss
