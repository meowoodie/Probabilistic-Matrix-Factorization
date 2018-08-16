#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Denoising Autoencoders including vanilla version (DAE) and stacked version (SDAE).

References:
- Denoising Autoencoder:
  https://gist.github.com/blackecho/3a6e4d512d3aa8aa6cf9
  https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/variational_autoencoder.py
-
'''

import tensorflow as tf
import numpy as np



class DAE(object):
    '''
    Denoising Autoencoder

    An extension of a classical autoencoder and it was introduced as a building
    block for deep networks in "Extracting and Composing Robust Features with
    Denoising Autoencoders"
    '''

    def __init__(self, n_visible, n_hidden, keep_prob=0.05,
                 lr=0.1, batch_size=64, n_epoches=10, corrupt_lv=0.2):
        self.lr = lr
        self.n_epoches  = n_epoches
        self.batch_size = batch_size

        # initialization of weights
        encode_w = tf.get_variable((n_visible, n_hidden),
            dtype=tf.float32, initializer=tf.random_normal_initializer())
        encode_b = tf.get_variable((n_hidden),
            dtype=tf.float32, initializer=tf.random_normal_initializer())

        decode_w = tf.get_variable((n_hidden, n_visible),
            dtype=tf.float32, initializer=tf.random_normal_initializer())
        decode_b = tf.get_variable((n_visible),
            dtype=tf.float32, initializer=tf.random_normal_initializer())

        # visible input
        self.x = tf.placeholder(tf.float32, (None, self.n_visible))

        # noise mask for corruptting input x
        noise_mask  = np.random.binomial(1, 1 - corrupt_lv, (self.batch_size, self.n_visible))
        corrupted_x = tf.mul(noise_mask , self.x)

        # hidden encode
        z = tf.nn.relu6(tf.add(tf.matmul(corrupted_x, encode_w), encode_b))
        z = tf.nn.dropout(z, keep_prob) # probability to keep units

        # reconstructed input
        x_hat = tf.nn.relu6(tf.add(tf.matmul(z, decode_w), decode_b))

        # define loss and optimizer, minimize the mean squared error
        cost      = tf.reduce_mean(tf.pow(x_hat - x, 2))
        optimizer = tf.train.AdamOptimizer(self.lr).minimize(cost)

        


if __name__ == '__main__':
    # Load MNIST data in a format suited for tensorflow.
    # The script input_data is available under this URL:
    # https://raw.githubusercontent.com/tensorflow/tensorflow/master/tensorflow/examples/tutorials/mnist/input_data.py
    from tensorflow.contrib.learn.python.learn.datasets import mnist
    data      = mnist.read_data_sets('MNIST_data', one_hot=True)
    n_samples = data.train.num_examples
    print(data)
    print(n_samples)
