import json
import os
import numpy as np
import tensorflow as tf
from src.helpers.definitions import global_step_path, model_path, model_name, model_dir
from src.helpers import dataset
from src.dl.layers import fully_connected


class Trainer:
  batch_size = 30
  print_every = 10
  save_every = 100
  train_steps = 10000
  learning_rate = 0.01
  
  def __init__(self):
    """
    # Since our images are of size 28x28 and we're using a fully-connected network (which can only take vectors
    # as inputs), our input size of the first layer should be 784 (28x28 with pixels unrolled).

    # Since this is a multi-class classification problem with 10 classes, our final layer
    # should have an output size of 10 and use the softmax activation function. This also means we should
    # use a softmax_cross_entropy loss function.

    # Network Architecture (3 Layers):
    #   Layer 1: (784|300), FCL, ReLU activation
    #   Layer 2: (300|100), FCL, ReLU activation
    #   Layer 3: (100|10), FCL, Softmax activation
    """
    print 'Extracting data...'
    # Get split train/test data
    self.X_train, self.Y_train = dataset.train()
    self.X_test, self.Y_test = dataset.test()

    # Create input/output placeholders
    self.x = tf.placeholder(shape=[None, self.X_train.shape[1]], dtype=tf.float32)
    self.y = tf.placeholder(shape=[None, self.Y_train.shape[1]], dtype=tf.float32)

    # Construct network
    o1 = fully_connected(self.x, output_units=300, activation=tf.nn.relu, scope='fc1')
    o2 = fully_connected(o1, output_units=100, activation=tf.nn.relu, scope='fc2')
    y_hat = fully_connected(o2, output_units=10, activation=tf.nn.softmax, scope='fc3')

    # Define loss and loss optimizer
    self.loss = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(logits=y_hat, labels=self.y))

    opt = tf.train.AdamOptimizer(self.learning_rate)
    self.minimize_loss = opt.minimize(self.loss)

    # Create a new session and initialize globals
    print 'Initializing session...'
    self.sess = tf.Session()
    self.sess.run(tf.global_variables_initializer())
    
    # Create our saver
    self.saver = tf.train.Saver(max_to_keep=200)
    
    # Restore prev model if exists
    if self.model_exists():
      print 'Previous model found. Restoring...'
      self.saver.restore(self.sess, model_path)
    
    # Get stored global step value
    self.global_step = self.get_gstep()

  def get_batch(self):
    inds = list(np.random.choice(range(self.X_train.shape[0]), size=self.batch_size, replace=False))
    inds.sort()
    return self.X_train[inds], self.Y_train[inds]

  def train(self):
    print 'Starting to train. Press Ctrl+C to save and exit.'
    
    try:
      for i in range(self.train_steps)[self.global_step:]:
        print '{}/{}'.format(i, self.train_steps)

        x, y = self.get_batch()
  
        feed_dict = {
          self.x: x,  # input (image)
          self.y: y   # label (truth)
        }
        
        self.sess.run(self.minimize_loss, feed_dict)

        self.global_step += 1
  
        if not self.global_step % self.print_every:
          loss = self.sess.run(self.loss, feed_dict)
          print 'Iteration {}: Training loss = {}'.format(i, loss)
  
        if not self.global_step % self.save_every:
          self.save_session()
          
    except (KeyboardInterrupt, SystemExit):
      print 'Interruption detected, exiting the program...'
    except BaseException, e:
      print 'Unexpected error during training: {}'.format(e)

    self.save_session()
  
  def save_session(self):
    print 'Saving session...'
    self.set_gstep()
    self.saver.save(self.sess, model_path)

  def get_gstep(self):
    # Create global_step.json if not there yet
    if not os.path.exists(global_step_path):
      self.set_gstep()
      return 0
    
    with open(global_step_path) as f:
      return json.load(f).get('val') or 0

  def set_gstep(self):
    if not hasattr(self, 'global_step'):
      self.global_step = 0

    with open(global_step_path, 'w+') as f:
      f.write(json.dumps({'val': self.global_step}, indent=2))

  @staticmethod
  def model_exists():
    return os.path.exists(model_dir) and len([f for f in os.listdir(model_dir) if f.startswith(model_name)]) > 0
