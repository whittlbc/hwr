import tensorflow as tf
import numpy as np

dataset = tf.contrib.learn.datasets.load_dataset('mnist')


def train():
  labels = dataset.train.labels
  return dataset.train.images, np.eye(labels.max()+1)[labels]
  

def test():
  labels = dataset.test.labels
  return dataset.test.images, np.eye(labels.max()+1)[labels]