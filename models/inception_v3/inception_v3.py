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
                                                         is_training=is_training)

    # Return
    return logits, end_points


def get_loss(pred, label, end_points, num_classes):
    """ pred: B*NUM_CLASSES,
        label: B, """
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=label)
    classify_loss = tf.reduce_mean(loss)
    tf.summary.scalar('classify loss', classify_loss)
    return classify_loss