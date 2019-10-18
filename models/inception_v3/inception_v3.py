import os
import cv2
import numpy as np
import tensorflow as tf
import slim.nets as nets
import matplotlib.pyplot as plt
import slim.nets.inception_v3 as inception_v3
from slim.nets import inception as inception_arg_scope
from tensorflow.contrib.framework import arg_scope


def get_model(depth_image, is_training, num_classes, bn_decay=None):

    # Define the model - assumes input image has values between 0-1 (roughly)
    with arg_scope(inception_arg_scope.inception_v3_arg_scope()):
        logits, end_points = nets.inception.inception_v3(inputs=depth_image, num_classes=num_classes,
                                                         is_training=is_training, dropout_keep_prob=0.8)

    # Return
    return logits, end_points


def get_loss(logits, labels, end_points, num_classes, l2_reg_weight=0.01):
    """ pred: B*NUM_CLASSES,
        label: B, """

    tf.losses.sparse_softmax_cross_entropy(logits=logits, labels=labels)
    tf_loss = tf.losses.get_total_loss()

    # # classification loss
    # class_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=label))
    # tf.summary.scalar('class_loss', class_loss)
    #
    # # l2 reg
    # vars = tf.trainable_variables()
    # L2_loss = tf.add_n([tf.nn.l2_loss(v) for v in vars if 'bias' not in v.name]) * l2_reg_weight
    # tf.summary.scalar('L2_loss', L2_loss)
    #
    # # final loss
    # final_loss = class_loss + L2_loss
    # tf.summary.scalar('final_loss', final_loss)
    return tf_loss