#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Denoising Autoencoders including vanilla version (dA) and stacked version (SdA).

References:
- Denoising Autoencoder:
  https://gist.github.com/blackecho/3a6e4d512d3aa8aa6cf9
  https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/variational_autoencoder.py
-
'''

import tensorflow as tf
import numpy as np
import arrow
import sys

class dA(object):
    '''
    Denoising Autoencoder class (dA)

    A basic extension of a vanilla autoencoder and it was introduced as a building
    block for deep networks in "Extracting and Composing Robust Features with
    Denoising Autoencoders"
    '''

    def __init__(self, n_visible, n_hidden, x=None, keep_prob=0.05,
                 lr=0.1, batch_size=64, n_epoches=10, corrupt_lv=0.2,
                 w=None, b_vis=None, b_hid=None):
        '''
        Initialize the dA class by specifying the number of visible units, the
        number of hidden units and the corruption level. The constructor also
        receives symbolic variables (tensors) for the input `x`, weights `w` and
        bias `b_vis`, `b_hid` . Such a symbolic variables are useful when, for
        example the input is the result of some computations, or when weights
        are shared between the dA and an MLP layer. When dealing with SdAs this
        always happens.
        '''
        self.lr = lr
        self.n_epoches  = n_epoches
        self.batch_size = batch_size
        self.w     = w     # weights
        self.b_vis = b_vis # bias of visible layer
        self.b_hid = b_hid # bias of hidden layer
        self.z     = None  # hidden encode output
        self.x     = None  # visible input
        self.is_stacked = False

        # initialization of weights
        if not self.w:
            self.w = tf.get_variable(name='encode_w', shape=(n_visible, n_hidden),
                dtype=tf.float32, initializer=tf.random_normal_initializer())
        if not self.b_vis:
            self.b_vis = tf.get_variable(name='encode_b', shape=(n_hidden),
                dtype=tf.float32, initializer=tf.random_normal_initializer())
        if not self.b_hid:
            self.b_hid = tf.get_variable(name='decode_b', shape=(n_visible),
                dtype=tf.float32, initializer=tf.random_normal_initializer())
        self.w_prime = tf.transpose(self.w)

        # visible input
        if x is None:
            self.x = tf.placeholder(tf.float32, (None, n_visible))
        else:
            self.is_stacked = True
            self.x = x

        # noise mask for corruptting input x
        noise_mask  = np.random.binomial(1, 1 - corrupt_lv, (self.batch_size, n_visible)).astype('float32')
        corrupted_x = tf.multiply(noise_mask, self.x)

        # hidden encode
        self.z = tf.nn.sigmoid(tf.add(tf.matmul(corrupted_x, self.w), self.b_vis))
        self.z = tf.nn.dropout(self.z, keep_prob) # probability to keep units

        # reconstructed input
        x_hat = tf.nn.sigmoid(tf.add(tf.matmul(self.z, self.w_prime), self.b_hid))

        # define loss and optimizer, minimize the mean squared error
        self.cost      = tf.reduce_mean(tf.pow(x_hat - self.x, 2))
        self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.cost)

        # initialization of a new sesssion in tensorflow
        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)

    def fit(self, train_x, test_x):
        '''
        Fit a basic two layers autoencoder
        '''
        n_trains  = train_x.shape[0]
        n_tests   = test_x.shape[0]
        n_batches = int(n_trains / self.batch_size)

        assert not n_trains % self.batch_size, "Indivisible batch size."
        assert n_tests > self.batch_size, "Size of test data should be larger than batch size."

        e = 0
        while e < self.n_epoches:
            e += 1
            # shuffle training samples
            shuffled_order = np.arange(n_trains)
            np.random.shuffle(shuffled_order)
            # training iterations over batches
            avg_train_loss = []
            avg_test_loss  = []
            for batch in range(n_batches):
                idx       = np.arange(self.batch_size * batch, self.batch_size * (batch + 1))
                batch_idx = np.mod(idx, n_trains).astype('int32')
                # training entries selected in current batch
                batch_x       = train_x[shuffled_order[batch_idx], :]
                sample_test_x = test_x[np.random.choice(n_tests, self.batch_size), :]
                # Optimize autoencoder
                if self.is_stacked:
                    self.sess.run(self.optimizer)
                else:
                    self.sess.run(self.optimizer, feed_dict={self.x: batch_x})
                # loss for train data and test data
                train_loss = self.sess.run(self.cost, feed_dict={self.x: batch_x})
                test_loss  = self.sess.run(self.cost, feed_dict={self.x: sample_test_x})
                avg_train_loss.append(train_loss)
                avg_test_loss.append(test_loss)
            # training log ouput
            avg_train_loss = np.mean(avg_train_loss) / float(self.batch_size)
            avg_test_loss  = np.mean(avg_test_loss) / float(self.batch_size)
            print('[%s] Epoch %d' % (arrow.now(), e), file=sys.stderr)
            print('[%s] Training loss:\t%f' % (arrow.now(), avg_train_loss), file=sys.stderr)
            print('[%s] Testing loss:\t%f' % (arrow.now(), avg_test_loss), file=sys.stderr)



class SdA(object):
    '''
    Stacked denoising autoencoder class (SdA)

    A stacked denoising autoencoder model is obtained by stacking several dAs.
    The hidden layer of the dA at layer `i` becomes the input of the dA at layer
    `i+1`. The first layer dA gets as input the input of the SdA, and the hidden
    layer of the last dA represents the output. Note that after pretraining, the
    SdA is dealt with as a normal MLP, the dAs are only used to initialize the
    weights.
    '''
    def __init__(self, n_visible, hidden_layers_sizes=[200, 100, 50],
                 keep_prob=0.05, lr=0.1, batch_size=64, n_epoches=10, corrupt_lv=0.2):
        '''
        '''
        self.lr        = lr
        self.dA_layers = []
        self.n_layers  = len(hidden_layers_sizes)
        self.x         = tf.placeholder(tf.float32, (None, n_visible))
        for i in range(self.n_layers):
            if i == 0:
                input_size  = n_visible
                layer_input = self.x
            else:
                input_size  = hidden_layers_sizes[i - 1]
                layer_input = self.hidden_layers[i - 1].z

            output_size = hidden_layers_sizes[i]

            dA_layer = dA(n_visible=input_size, n_hidden=output_size,
                          x=layer_input, keep_prob=keep_prob, lr=lr,
                          batch_size=batch_size, n_epoches=n_epoches,
                          corrupt_lv=corrupt_lv)

            self.dA_layers.append(dA_layer)

    def logistic_output(self):
        '''
        '''
        self.logistic_pred = tf.nn.softmax(self.hidden_layers[-1].z) # Softmax
        self.finetune_cost = tf.reduce_mean(-tf.reduce_sum(self.y * tf.log(logistic_pred), reduction_indices=1))
        self.optimizer     = tf.train.GradientDescentOptimizer(self.lr).minimize(self.finetune_cost)

    def pre_train(self, train_x, test_x):
        '''
        Generates a list of functions, each of them implementing one
        step in trainnig the dA corresponding to the layer with same index.
        The function will require as input the minibatch index, and to train
        a dA you just need to iterate, calling the corresponding function on
        all minibatch indexes.
        '''
        for dA_layer in self.dA_layers:
            cost, updates = dA.get_cost_updates(corruption_level,
                                                learning_rate)

    def fine_tune(self, train_x, train_y, test_x, test_y):
        pass






if __name__ == '__main__':
    # An unittest on MNIST data for Denoising Autoencoder

    # Load MNIST data in a format suited for tensorflow.
    # The script input_data is available under this URL:
    # https://raw.githubusercontent.com/tensorflow/tensorflow/master/tensorflow/examples/tutorials/mnist/input_data.py
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)
    train_x = np.vstack([img.reshape(-1,) for img in mnist.train.images])
    train_y = mnist.train.labels

    test_x  = np.vstack([img.reshape(-1,) for img in mnist.test.images])
    test_y  = mnist.test.labels

    n_feature = train_x.shape[1]
    n_hidden  = 100

    dae = dA(n_feature, n_hidden,
              keep_prob=0.05, lr=0.005, batch_size=55, n_epoches=10, corrupt_lv=0.2)
    dae.fit(train_x, test_x)
