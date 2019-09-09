""" Wrapper functions for TensorFlow layers.

Author: Charles R. Qi
Date: November 2016

Upadted by Yue Wang and Yongbin Sun
"""

import numpy as np
import tensorflow as tf

def _variable_on_cpu(name, shape, initializer, use_fp16=False, trainable=True):
  """Helper to create a Variable stored on CPU memory.
  Args:
    name: name of the variable
    shape: list of ints
    initializer: initializer for Variable
  Returns:
    Variable Tensor
  """
  with tf.device('/cpu:0'):
    dtype = tf.float16 if use_fp16 else tf.float32
    var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype, trainable=trainable)
  return var

def _variable_with_weight_decay(name, shape, stddev, wd, use_xavier=True):
  """Helper to create an initialized Variable with weight decay.

  Note that the Variable is initialized with a truncated normal distribution.
  A weight decay is added only if one is specified.

  Args:
    name: name of the variable
    shape: list of ints
    stddev: standard deviation of a truncated Gaussian
    wd: add L2Loss weight decay multiplied by this float. If None, weight
        decay is not added for this Variable.
    use_xavier: bool, whether to use xavier initializer

  Returns:
    Variable Tensor
  """
  if use_xavier:
    initializer = tf.contrib.layers.xavier_initializer()
  else:
    initializer = tf.truncated_normal_initializer(stddev=stddev)
  var = _variable_on_cpu(name, shape, initializer)
  if wd is not None:
    weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
  return var


def conv1d(inputs,
           num_output_channels,
           kernel_size,
           scope,
           stride=1,
           padding='SAME',
           use_xavier=True,
           stddev=1e-3,
           weight_decay=0.0,
           activation_fn=tf.nn.relu,
           bn=False,
           bn_decay=None,
           is_training=None,
           is_dist=False):
  """ 1D convolution with non-linear operation.

  Args:
    inputs: 3-D tensor variable BxLxC
    num_output_channels: int
    kernel_size: int
    scope: string
    stride: int
    padding: 'SAME' or 'VALID'
    use_xavier: bool, use xavier_initializer if true
    stddev: float, stddev for truncated_normal init
    weight_decay: float
    activation_fn: function
    bn: bool, whether to use batch norm
    bn_decay: float or float tensor variable in [0,1]
    is_training: bool Tensor variable

  Returns:
    Variable tensor
  """
  with tf.variable_scope(scope) as sc:
    num_in_channels = inputs.get_shape()[-1].value
    kernel_shape = [kernel_size,
                    num_in_channels, num_output_channels]
    kernel = _variable_with_weight_decay('weights',
                                         shape=kernel_shape,
                                         use_xavier=use_xavier,
                                         stddev=stddev,
                                         wd=weight_decay)
    outputs = tf.nn.conv1d(inputs, kernel,
                           stride=stride,
                           padding=padding)
    biases = _variable_on_cpu('biases', [num_output_channels],
                              tf.constant_initializer(0.0))
    outputs = tf.nn.bias_add(outputs, biases)

    if bn:
      outputs = batch_norm_for_conv1d(outputs, is_training,
                                      bn_decay=bn_decay, scope='bn', is_dist=is_dist)

    if activation_fn is not None:
      outputs = activation_fn(outputs)
    return outputs




def conv2d(inputs,
           num_output_channels,
           kernel_size,
           scope,
           stride=[1, 1],
           padding='SAME',
           use_xavier=True,
           stddev=1e-3,
           weight_decay=0.0,
           activation_fn=tf.nn.relu,
           bn=False,
           bn_decay=None,
           is_training=None,
           is_dist=False):
  """ 2D convolution with non-linear operation.

  Args:
    inputs: 4-D tensor variable BxHxWxC
    num_output_channels: int
    kernel_size: a list of 2 ints
    scope: string
    stride: a list of 2 ints
    padding: 'SAME' or 'VALID'
    use_xavier: bool, use xavier_initializer if true
    stddev: float, stddev for truncated_normal init
    weight_decay: float
    activation_fn: function
    bn: bool, whether to use batch norm
    bn_decay: float or float tensor variable in [0,1]
    is_training: bool Tensor variable

  Returns:
    Variable tensor
  """
  with tf.variable_scope(scope) as sc:
      kernel_h, kernel_w = kernel_size
      num_in_channels = inputs.get_shape()[-1].value
      kernel_shape = [kernel_h, kernel_w,
                      num_in_channels, num_output_channels]
      kernel = _variable_with_weight_decay('weights',
                                           shape=kernel_shape,
                                           use_xavier=use_xavier,
                                           stddev=stddev,
                                           wd=weight_decay)
      stride_h, stride_w = stride
      outputs = tf.nn.conv2d(inputs, kernel,
                             [1, stride_h, stride_w, 1],
                             padding=padding)
      biases = _variable_on_cpu('biases', [num_output_channels],
                                tf.constant_initializer(0.0))
      outputs = tf.nn.bias_add(outputs, biases)

      if bn:
        outputs = batch_norm_for_conv2d(outputs, is_training,
                                        bn_decay=bn_decay, scope='bn', is_dist=is_dist)

      if activation_fn is not None:
        outputs = activation_fn(outputs)
      return outputs


def conv2d_reg(point_cloud, nn_idx,
              num_output_channels,
              kernel_size,
              scope,
              stride=[1, 1],
              padding='SAME',
              use_xavier=True,
              stddev=1e-3,
              weight_decay=0.0,
              activation_fn=tf.nn.relu,
              bn=False,
              bn_decay=None,
              is_training=None,
              is_dist=False):
  """ 2D convolution with non-linear operation.

  Args:
    point_cloud: 3-D tensor variable BxHxW
    nn_idx: 3-D tensor variable BxHxH
    nn_reg: 4-D tensor variable BxHxK
    num_output_channels: int
    kernel_size: a list of 2 ints
    scope: string
    stride: a list of 2 ints
    padding: 'SAME' or 'VALID'
    use_xavier: bool, use xavier_initializer if true
    stddev: float, stddev for truncated_normal init
    weight_decay: float
    activation_fn: function
    bn: bool, whether to use batch norm
    bn_decay: float or float tensor variable in [0,1]
    is_training: bool Tensor variable

  Returns:
    Variable tensor
  """
  with tf.variable_scope(scope) as sc:

      # Sizes
      point_cloud_shape = point_cloud.get_shape()
      batch_size = point_cloud_shape[0].value				# 32
      num_points = point_cloud_shape[1].value				# 1024
      num_dims = point_cloud_shape[2].value				# 3
      kernel_h, kernel_w = kernel_size					# [1, 1]
      num_in_channels = 2*num_dims					# 2*3 = 6
      kernel_shape = [kernel_h, kernel_w,				# [1, 1, 6, 64]
                      num_in_channels, num_output_channels]
      k = 20
      stride_h, stride_w = stride

      # Neighbor - query
      point_cloud_central = tf.expand_dims(point_cloud, axis=-2)
      point_cloud_central = tf.tile(point_cloud_central, [1, 1, k, 1])

      # idx_ = [ 0  1024  2048  3072  4096  5120  6144  7168  8192  9216 10240 11264 etc.
      idx_ = tf.range(batch_size) * num_points
      idx_ = tf.reshape(idx_, [batch_size, 1, 1]) 

      # flat point pointcloud
      point_cloud_flat = tf.reshape(point_cloud, [-1, num_dims])
      point_cloud_neighbors = tf.gather(point_cloud_flat, nn_idx+idx_)
      point_cloud_diff = point_cloud_neighbors - point_cloud_central
      edge_feature = tf.concat([point_cloud_central, point_cloud_neighbors-point_cloud_central], axis=-1)

      # XYZ
      elems_x = tf.reshape(point_cloud_diff[..., 0], [-1])
      elems_1 = tf.ones((elems_x.get_shape()), dtype=tf.int64)
      elems_0 = tf.zeros((elems_x.get_shape()), dtype=tf.int64)
      point_cloud_diff_x = tf.where(elems_x > 0, elems_1, elems_0)
      elems_y = tf.reshape(point_cloud_diff[..., 1], [-1])
      point_cloud_diff_y = tf.where(elems_y > 0, elems_1, elems_0)
      elems_z = tf.reshape(point_cloud_diff[..., 2], [-1])
      point_cloud_diff_z = tf.where(elems_z > 0, elems_1, elems_0)

      # final subregion
      x_m = tf.constant(4, tf.int64)
      y_m = tf.constant(2, tf.int64)
      z_m = tf.constant(1, tf.int64)
      point_cloud_subregions = point_cloud_diff_x*x_m + point_cloud_diff_y*y_m + point_cloud_diff_z*z_m
      point_cloud_subregions = tf.reshape(point_cloud_subregions, (batch_size, num_points, k))

      # for loop
      outputs = []
      for i in range(8):

        # kernel
        kernel_i = _variable_with_weight_decay('weights_' + str(i),
                                               shape=kernel_shape,
                                               use_xavier=use_xavier,
                                               stddev=stddev,
                                               wd=weight_decay)

        # Take indices where nn_reg is i
        mask_i = tf.map_fn(lambda x: tf.equal(x, i), point_cloud_subregions, dtype=tf.bool)
        mask_i = tf.cast(mask_i, tf.float32)
      
        # Convolve 
        output_i = tf.nn.conv2d(edge_feature, kernel_i, [1, stride_h, stride_w, 1], padding=padding)
        biases_i = _variable_on_cpu('biases_' + str(i), [num_output_channels], tf.constant_initializer(0.0))
        output_i = tf.nn.bias_add(output_i, biases_i)

        #Apply mask
        output_i = tf.multiply(output_i, tf.expand_dims(mask_i, axis=-1))
        output_i = tf.reduce_max(output_i, axis=-2, keep_dims=False)
        outputs.append(output_i)

      outputs = tf.stack(outputs, axis=-2)
      outputs = tf.reduce_sum(outputs, axis=-2, keep_dims=True)

      # bn decay
      if bn:
        outputs = batch_norm_for_conv2d(outputs, is_training,
                                        bn_decay=bn_decay, scope='bn', is_dist=is_dist)

      # Activation
      if activation_fn is not None:
        outputs = activation_fn(outputs)

      # Return
      return outputs

def conv2d_transpose(inputs,
                     num_output_channels,
                     kernel_size,
                     scope,
                     stride=[1, 1],
                     padding='SAME',
                     use_xavier=True,
                     stddev=1e-3,
                     weight_decay=0.0,
                     activation_fn=tf.nn.relu,
                     bn=False,
                     bn_decay=None,
                     is_training=None,
                     is_dist=False):
  """ 2D convolution transpose with non-linear operation.

  Args:
    inputs: 4-D tensor variable BxHxWxC
    num_output_channels: int
    kernel_size: a list of 2 ints
    scope: string
    stride: a list of 2 ints
    padding: 'SAME' or 'VALID'
    use_xavier: bool, use xavier_initializer if true
    stddev: float, stddev for truncated_normal init
    weight_decay: float
    activation_fn: function
    bn: bool, whether to use batch norm
    bn_decay: float or float tensor variable in [0,1]
    is_training: bool Tensor variable

  Returns:
    Variable tensor

  Note: conv2d(conv2d_transpose(a, num_out, ksize, stride), a.shape[-1], ksize, stride) == a
  """
  with tf.variable_scope(scope) as sc:
      kernel_h, kernel_w = kernel_size
      num_in_channels = inputs.get_shape()[-1].value
      kernel_shape = [kernel_h, kernel_w,
                      num_output_channels, num_in_channels] # reversed to conv2d
      kernel = _variable_with_weight_decay('weights',
                                           shape=kernel_shape,
                                           use_xavier=use_xavier,
                                           stddev=stddev,
                                           wd=weight_decay)
      stride_h, stride_w = stride
      
      # from slim.convolution2d_transpose
      def get_deconv_dim(dim_size, stride_size, kernel_size, padding):
          dim_size *= stride_size

          if padding == 'VALID' and dim_size is not None:
            dim_size += max(kernel_size - stride_size, 0)
          return dim_size

      # caculate output shape
      batch_size = inputs.get_shape()[0].value
      height = inputs.get_shape()[1].value
      width = inputs.get_shape()[2].value
      out_height = get_deconv_dim(height, stride_h, kernel_h, padding)
      out_width = get_deconv_dim(width, stride_w, kernel_w, padding)
      output_shape = [batch_size, out_height, out_width, num_output_channels]

      outputs = tf.nn.conv2d_transpose(inputs, kernel, output_shape,
                             [1, stride_h, stride_w, 1],
                             padding=padding)
      biases = _variable_on_cpu('biases', [num_output_channels],
                                tf.constant_initializer(0.0))
      outputs = tf.nn.bias_add(outputs, biases)

      if bn:
        outputs = batch_norm_for_conv2d(outputs, is_training,
                                        bn_decay=bn_decay, scope='bn', is_dist=is_dist)

      if activation_fn is not None:
        outputs = activation_fn(outputs)
      return outputs

   

def conv3d(inputs,
           num_output_channels,
           kernel_size,
           scope,
           stride=[1, 1, 1],
           padding='SAME',
           use_xavier=True,
           stddev=1e-3,
           weight_decay=0.0,
           activation_fn=tf.nn.relu,
           bn=False,
           bn_decay=None,
           is_training=None,
           is_dist=False):
  """ 3D convolution with non-linear operation.

  Args:
    inputs: 5-D tensor variable BxDxHxWxC
    num_output_channels: int
    kernel_size: a list of 3 ints
    scope: string
    stride: a list of 3 ints
    padding: 'SAME' or 'VALID'
    use_xavier: bool, use xavier_initializer if true
    stddev: float, stddev for truncated_normal init
    weight_decay: float
    activation_fn: function
    bn: bool, whether to use batch norm
    bn_decay: float or float tensor variable in [0,1]
    is_training: bool Tensor variable

  Returns:
    Variable tensor
  """
  with tf.variable_scope(scope) as sc:
    kernel_d, kernel_h, kernel_w = kernel_size
    num_in_channels = inputs.get_shape()[-1].value
    kernel_shape = [kernel_d, kernel_h, kernel_w,
                    num_in_channels, num_output_channels]
    kernel = _variable_with_weight_decay('weights',
                                         shape=kernel_shape,
                                         use_xavier=use_xavier,
                                         stddev=stddev,
                                         wd=weight_decay)
    stride_d, stride_h, stride_w = stride
    outputs = tf.nn.conv3d(inputs, kernel,
                           [1, stride_d, stride_h, stride_w, 1],
                           padding=padding)
    biases = _variable_on_cpu('biases', [num_output_channels],
                              tf.constant_initializer(0.0))
    outputs = tf.nn.bias_add(outputs, biases)
    
    if bn:
      outputs = batch_norm_for_conv3d(outputs, is_training,
                                      bn_decay=bn_decay, scope='bn', is_dist=is_dist)

    if activation_fn is not None:
      outputs = activation_fn(outputs)
    return outputs

def fully_connected(inputs,
                    num_outputs,
                    scope,
                    use_xavier=True,
                    stddev=1e-3,
                    weight_decay=0.0,
                    activation_fn=tf.nn.relu,
                    bn=False,
                    bn_decay=None,
                    is_training=None,
                    is_dist=False):
  """ Fully connected layer with non-linear operation.
  
  Args:
    inputs: 2-D tensor BxN
    num_outputs: int
  
  Returns:
    Variable tensor of size B x num_outputs.
  """
  with tf.variable_scope(scope) as sc:
    num_input_units = inputs.get_shape()[-1].value
    weights = _variable_with_weight_decay('weights',
                                          shape=[num_input_units, num_outputs],
                                          use_xavier=use_xavier,
                                          stddev=stddev,
                                          wd=weight_decay)
    outputs = tf.matmul(inputs, weights)
    biases = _variable_on_cpu('biases', [num_outputs],
                             tf.constant_initializer(0.0))
    outputs = tf.nn.bias_add(outputs, biases)
     
    if bn:
      outputs = batch_norm_for_fc(outputs, is_training, bn_decay, 'bn', is_dist=is_dist)

    if activation_fn is not None:
      outputs = activation_fn(outputs)
    return outputs


def max_pool2d(inputs,
               kernel_size,
               scope,
               stride=[2, 2],
               padding='VALID'):
  """ 2D max pooling.

  Args:
    inputs: 4-D tensor BxHxWxC
    kernel_size: a list of 2 ints
    stride: a list of 2 ints
  
  Returns:
    Variable tensor
  """
  with tf.variable_scope(scope) as sc:
    kernel_h, kernel_w = kernel_size
    stride_h, stride_w = stride
    outputs = tf.nn.max_pool(inputs,
                             ksize=[1, kernel_h, kernel_w, 1],
                             strides=[1, stride_h, stride_w, 1],
                             padding=padding,
                             name=sc.name)
    return outputs

def avg_pool2d(inputs,
               kernel_size,
               scope,
               stride=[2, 2],
               padding='VALID'):
  """ 2D avg pooling.

  Args:
    inputs: 4-D tensor BxHxWxC
    kernel_size: a list of 2 ints
    stride: a list of 2 ints
  
  Returns:
    Variable tensor
  """
  with tf.variable_scope(scope) as sc:
    kernel_h, kernel_w = kernel_size
    stride_h, stride_w = stride
    outputs = tf.nn.avg_pool(inputs,
                             ksize=[1, kernel_h, kernel_w, 1],
                             strides=[1, stride_h, stride_w, 1],
                             padding=padding,
                             name=sc.name)
    return outputs


def max_pool3d(inputs,
               kernel_size,
               scope,
               stride=[2, 2, 2],
               padding='VALID'):
  """ 3D max pooling.

  Args:
    inputs: 5-D tensor BxDxHxWxC
    kernel_size: a list of 3 ints
    stride: a list of 3 ints
  
  Returns:
    Variable tensor
  """
  with tf.variable_scope(scope) as sc:
    kernel_d, kernel_h, kernel_w = kernel_size
    stride_d, stride_h, stride_w = stride
    outputs = tf.nn.max_pool3d(inputs,
                               ksize=[1, kernel_d, kernel_h, kernel_w, 1],
                               strides=[1, stride_d, stride_h, stride_w, 1],
                               padding=padding,
                               name=sc.name)
    return outputs

def avg_pool3d(inputs,
               kernel_size,
               scope,
               stride=[2, 2, 2],
               padding='VALID'):
  """ 3D avg pooling.

  Args:
    inputs: 5-D tensor BxDxHxWxC
    kernel_size: a list of 3 ints
    stride: a list of 3 ints
  
  Returns:
    Variable tensor
  """
  with tf.variable_scope(scope) as sc:
    kernel_d, kernel_h, kernel_w = kernel_size
    stride_d, stride_h, stride_w = stride
    outputs = tf.nn.avg_pool3d(inputs,
                               ksize=[1, kernel_d, kernel_h, kernel_w, 1],
                               strides=[1, stride_d, stride_h, stride_w, 1],
                               padding=padding,
                               name=sc.name)
    return outputs





def batch_norm_template(inputs, is_training, scope, moments_dims, bn_decay):
  """ Batch normalization on convolutional maps and beyond...
  Ref.: http://stackoverflow.com/questions/33949786/how-could-i-use-batch-normalization-in-tensorflow
  
  Args:
      inputs:        Tensor, k-D input ... x C could be BC or BHWC or BDHWC
      is_training:   boolean tf.Varialbe, true indicates training phase
      scope:         string, variable scope
      moments_dims:  a list of ints, indicating dimensions for moments calculation
      bn_decay:      float or float tensor variable, controling moving average weight
  Return:
      normed:        batch-normalized maps
  """
  with tf.variable_scope(scope) as sc:
    num_channels = inputs.get_shape()[-1].value
    beta = tf.Variable(tf.constant(0.0, shape=[num_channels]),
                       name='beta', trainable=True)
    gamma = tf.Variable(tf.constant(1.0, shape=[num_channels]),
                        name='gamma', trainable=True)
    batch_mean, batch_var = tf.nn.moments(inputs, moments_dims, name='moments')
    decay = bn_decay if bn_decay is not None else 0.9
    ema = tf.train.ExponentialMovingAverage(decay=decay)
    # Operator that maintains moving averages of variables.
    ema_apply_op = tf.cond(is_training,
                           lambda: ema.apply([batch_mean, batch_var]),
                           lambda: tf.no_op())
    
    # Update moving average and return current batch's avg and var.
    def mean_var_with_update():
      with tf.control_dependencies([ema_apply_op]):
        return tf.identity(batch_mean), tf.identity(batch_var)
    
    # ema.average returns the Variable holding the average of var.
    mean, var = tf.cond(is_training,
                        mean_var_with_update,
                        lambda: (ema.average(batch_mean), ema.average(batch_var)))
    normed = tf.nn.batch_normalization(inputs, mean, var, beta, gamma, 1e-3)
  return normed


def batch_norm_dist_template(inputs, is_training, scope, moments_dims, bn_decay):
  """ The batch normalization for distributed training.
  Args:
      inputs:        Tensor, k-D input ... x C could be BC or BHWC or BDHWC
      is_training:   boolean tf.Varialbe, true indicates training phase
      scope:         string, variable scope
      moments_dims:  a list of ints, indicating dimensions for moments calculation
      bn_decay:      float or float tensor variable, controling moving average weight
  Return:
      normed:        batch-normalized maps
  """
  with tf.variable_scope(scope) as sc:
    num_channels = inputs.get_shape()[-1].value
    beta = _variable_on_cpu('beta', [num_channels], initializer=tf.zeros_initializer())
    gamma = _variable_on_cpu('gamma', [num_channels], initializer=tf.ones_initializer())

    pop_mean = _variable_on_cpu('pop_mean', [num_channels], initializer=tf.zeros_initializer(), trainable=False)
    pop_var = _variable_on_cpu('pop_var', [num_channels], initializer=tf.ones_initializer(), trainable=False)

    def train_bn_op():
      batch_mean, batch_var = tf.nn.moments(inputs, moments_dims, name='moments')
      decay = bn_decay if bn_decay is not None else 0.9
      train_mean = tf.assign(pop_mean, pop_mean * decay + batch_mean * (1 - decay)) 
      train_var = tf.assign(pop_var, pop_var * decay + batch_var * (1 - decay))
      with tf.control_dependencies([train_mean, train_var]):
        return tf.nn.batch_normalization(inputs, batch_mean, batch_var, beta, gamma, 1e-3)

    def test_bn_op():
      return tf.nn.batch_normalization(inputs, pop_mean, pop_var, beta, gamma, 1e-3)

    normed = tf.cond(is_training,
                     train_bn_op,
                     test_bn_op)
    return normed



def batch_norm_for_fc(inputs, is_training, bn_decay, scope, is_dist=False):
  """ Batch normalization on FC data.
  
  Args:
      inputs:      Tensor, 2D BxC input
      is_training: boolean tf.Varialbe, true indicates training phase
      bn_decay:    float or float tensor variable, controling moving average weight
      scope:       string, variable scope
      is_dist:     true indicating distributed training scheme
  Return:
      normed:      batch-normalized maps
  """
  if is_dist:
    return batch_norm_dist_template(inputs, is_training, scope, [0,], bn_decay)
  else:
    return batch_norm_template(inputs, is_training, scope, [0,], bn_decay)


def batch_norm_for_conv1d(inputs, is_training, bn_decay, scope, is_dist=False):
  """ Batch normalization on 1D convolutional maps.
  
  Args:
      inputs:      Tensor, 3D BLC input maps
      is_training: boolean tf.Varialbe, true indicates training phase
      bn_decay:    float or float tensor variable, controling moving average weight
      scope:       string, variable scope
      is_dist:     true indicating distributed training scheme
  Return:
      normed:      batch-normalized maps
  """
  if is_dist:
    return batch_norm_dist_template(inputs, is_training, scope, [0,1], bn_decay)
  else:
    return batch_norm_template(inputs, is_training, scope, [0,1], bn_decay)



  
def batch_norm_for_conv2d(inputs, is_training, bn_decay, scope, is_dist=False):
  """ Batch normalization on 2D convolutional maps.
  
  Args:
      inputs:      Tensor, 4D BHWC input maps
      is_training: boolean tf.Varialbe, true indicates training phase
      bn_decay:    float or float tensor variable, controling moving average weight
      scope:       string, variable scope
      is_dist:     true indicating distributed training scheme
  Return:
      normed:      batch-normalized maps
  """
  if is_dist:
    return batch_norm_dist_template(inputs, is_training, scope, [0,1,2], bn_decay)
  else:
    return batch_norm_template(inputs, is_training, scope, [0,1,2], bn_decay)



def batch_norm_for_conv3d(inputs, is_training, bn_decay, scope, is_dist=False):
  """ Batch normalization on 3D convolutional maps.
  
  Args:
      inputs:      Tensor, 5D BDHWC input maps
      is_training: boolean tf.Varialbe, true indicates training phase
      bn_decay:    float or float tensor variable, controling moving average weight
      scope:       string, variable scope
      is_dist:     true indicating distributed training scheme
  Return:
      normed:      batch-normalized maps
  """
  if is_dist:
    return batch_norm_dist_template(inputs, is_training, scope, [0,1,2,3], bn_decay)
  else:
    return batch_norm_template(inputs, is_training, scope, [0,1,2,3], bn_decay)


def dropout(inputs,
            is_training,
            scope,
            keep_prob=0.5,
            noise_shape=None):
  """ Dropout layer.

  Args:
    inputs: tensor
    is_training: boolean tf.Variable
    scope: string
    keep_prob: float in [0,1]
    noise_shape: list of ints

  Returns:
    tensor variable
  """
  with tf.variable_scope(scope) as sc:
    outputs = tf.cond(is_training,
                      lambda: tf.nn.dropout(inputs, keep_prob, noise_shape),
                      lambda: inputs)
    return outputs


def pairwise_distance(point_cloud):
  """Compute pairwise distance of a point cloud.

  Args:
    point_cloud: tensor (batch_size, num_points, num_dims)

  Returns:
    pairwise distance: (batch_size, num_points, num_points)
  """
  og_batch_size = point_cloud.get_shape().as_list()[0]
  point_cloud = tf.squeeze(point_cloud)
  if og_batch_size == 1:
    point_cloud = tf.expand_dims(point_cloud, 0)
    
  point_cloud_transpose = tf.transpose(point_cloud, perm=[0, 2, 1])
  point_cloud_inner = tf.matmul(point_cloud, point_cloud_transpose)
  point_cloud_inner = -2*point_cloud_inner
  point_cloud_square = tf.reduce_sum(tf.square(point_cloud), axis=-1, keep_dims=True)
  point_cloud_square_tranpose = tf.transpose(point_cloud_square, perm=[0, 2, 1])
  return point_cloud_square + point_cloud_inner + point_cloud_square_tranpose


def knn(adj_matrix, k=20):
  """Get KNN based on the pairwise distance.
  Args:
    pairwise distance: (batch_size, num_points, num_points)
    k: int

  Returns:
    nearest neighbors: (batch_size, num_points, k)
  """
  neg_adj = -adj_matrix
  _, nn_idx = tf.nn.top_k(neg_adj, k=k)
  return nn_idx

def knn_subregions(point_cloud, nn_idx):
  """ Get subregions indices for each neigbor
  Args:
    point_cloud: tensor (batch_size, num_points, num_dims)
    nearest neighbors: tensor (batch_size, num_points, k)

  Returns:
    nearest neighbors subregions: (batch_size, num_points, k)
  """

  # sizes
  point_cloud_shape = point_cloud.get_shape()
  batch_size = point_cloud_shape[0].value
  num_points = point_cloud_shape[1].value
  num_dims = point_cloud_shape[2].value
  k = nn_idx.get_shape()[2].value

  # idx_ = [ 0  1024  2048  3072  4096  5120  6144  7168  8192  9216 10240 11264 etc.
  idx_ = tf.range(batch_size) * num_points
  idx_ = tf.reshape(idx_, [batch_size, 1, 1])

  # flat and neighbors
  point_cloud_flat = tf.reshape(point_cloud, [-1, num_dims])
  point_cloud_neighbors = tf.gather(point_cloud_flat, nn_idx+idx_)  

  # Neighbor - query
  point_cloud_central = tf.expand_dims(point_cloud, axis=-2)
  point_cloud_central = tf.tile(point_cloud_central, [1, 1, k, 1])
  point_cloud_diff = point_cloud_neighbors - point_cloud_central

  # XYZ
  elems_x = tf.reshape(point_cloud_diff[..., 0], [-1])
  point_cloud_diff_x = tf.map_fn(lambda x: tf.greater(x,0), elems_x, dtype=tf.bool)
  elems_y = tf.reshape(point_cloud_diff[..., 1], [-1])
  point_cloud_diff_y = tf.map_fn(lambda x: tf.greater(x,0), elems_y, dtype=tf.bool)
  elems_z = tf.reshape(point_cloud_diff[..., 2], [-1])
  point_cloud_diff_z = tf.map_fn(lambda x: tf.greater(x,0), elems_z, dtype=tf.bool)  

  # final subregion
  point_cloud_diff_x = tf.cast(point_cloud_diff_x, dtype=tf.int64)
  point_cloud_diff_y = tf.cast(point_cloud_diff_y, dtype=tf.int64)
  point_cloud_diff_z = tf.cast(point_cloud_diff_z, dtype=tf.int64)
  elems = tf.stack((point_cloud_diff_x, point_cloud_diff_y, point_cloud_diff_z), axis=-1)
  point_cloud_subregions = tf.map_fn(lambda x: x[0]*4 + x[1]*2 + x[2], elems, dtype=tf.int64)
  point_cloud_subregions = tf.reshape(point_cloud_subregions, (batch_size, num_points, k))
  return point_cloud_subregions

def get_edge_feature(point_cloud, nn_idx, k=20):
  """Construct edge feature for each point
  Args:
    point_cloud: (batch_size, num_points, 1, num_dims)
    nn_idx: (batch_size, num_points, k)
    k: int

  Returns:
    edge features: (batch_size, num_points, k, num_dims)
  """
  og_batch_size = point_cloud.get_shape().as_list()[0]
  point_cloud = tf.squeeze(point_cloud)
  if og_batch_size == 1:
    point_cloud = tf.expand_dims(point_cloud, 0)

  point_cloud_central = point_cloud

  point_cloud_shape = point_cloud.get_shape()
  batch_size = point_cloud_shape[0].value
  num_points = point_cloud_shape[1].value
  num_dims = point_cloud_shape[2].value

  # idx_ = [ 0  1024  2048  3072  4096  5120  6144  7168  8192  9216 10240 11264 etc.
  idx_ = tf.range(batch_size) * num_points
  idx_ = tf.reshape(idx_, [batch_size, 1, 1]) 

  point_cloud_flat = tf.reshape(point_cloud, [-1, num_dims])
  point_cloud_neighbors = tf.gather(point_cloud_flat, nn_idx+idx_)
  point_cloud_central = tf.expand_dims(point_cloud_central, axis=-2)

  point_cloud_central = tf.tile(point_cloud_central, [1, 1, k, 1])

  # Global & local
  edge_feature = tf.concat([point_cloud_central, point_cloud_neighbors-point_cloud_central], axis=-1)

  # Local only
  #edge_feature = point_cloud_neighbors - point_cloud_central

  return edge_feature
