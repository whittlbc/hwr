import tensorflow as tf
import numpy as np


def fully_connected(x, output_units=100, activation=tf.identity, std=1e-3, scope='fc', reuse=False):
  """
  args:
      x, (tf tensor), tensor with shape (batch, width, height, channels)
      std, (float/string), std of weight initialization, 'xavier' for xavier
          initialization
      output_units,(int), number of output units for the layer
      activation, (tf function), tensorflow activation function, e.g. tf.nn.relu
      scope, (string), scope under which to store variables
      reuse, (boolean), whether we want to reuse variables that have already
          been created (i.e. reuse an earilier layer)

  returns:
      a, (tf tensor), the output of the fully_connected layer, has size
          (batch, output_units)
  """
  with tf.variable_scope(scope, reuse=reuse):
    s = x.get_shape().as_list()

    shape = [s[1], output_units]

    if std == 'xavier':
      std = np.sqrt(2.0 / shape[0])

    W = tf.get_variable('W', shape=shape, initializer=tf.random_normal_initializer(0.0, std))
    
    b = tf.get_variable('b', shape=shape[1], initializer=tf.random_normal_initializer(0.0, std))

    h = tf.matmul(x, W) + b
    
    a = activation(h, name='a')
    
    return a