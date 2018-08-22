#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Denoising Autoencoders including vanilla version (dA) and stacked version (SdA).
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

    def __init__(self, model_name, n_visible, n_hidden, x=None,
                 keep_prob=0.05, lr=0.1, batch_size=64, n_epoches=10,
                 corrupt_lv=0.2, w=None, b_vis=None, b_hid=None):
        '''
        Initialize the dA class by specifying the number of visible units, the
        number of hidden units and the corruption level. The constructor also
        receives symbolic variables (tensors) for the input `x`, weights `w` and
        bias `b_vis`, `b_hid` . Such a symbolic variables are useful when, for
        example the input is the result of some computations, or when weights
        are shared between the dA and an MLP layer. When dealing with SdAs this
        always happens.
        '''
        self.model_name = model_name
        self.n_visible  = n_visible
        self.n_hidden   = n_hidden
        self.n_epoches  = n_epoches
        self.batch_size = batch_size
        self.corrupt_lv = corrupt_lv
        self.lr         = lr
        self.w     = w     # weights
        self.b_vis = b_vis # bias of visible layer
        self.b_hid = b_hid # bias of hidden layer
        self.z     = None  # hidden encode output
        self.x     = None  # visible input
        self.is_stacked = False
        # initialization of weights
        if not self.w:
            self.w = tf.get_variable(name='%s_weights' % self.model_name, shape=(n_visible, n_hidden),
                dtype=tf.float32, initializer=tf.random_normal_initializer())
        if not self.b_vis:
            self.b_vis = tf.get_variable(name='%s_visible_bias' % self.model_name, shape=(n_hidden),
                dtype=tf.float32, initializer=tf.random_normal_initializer())
        if not self.b_hid:
            self.b_hid = tf.get_variable(name='%s_hidden_bias' % self.model_name, shape=(n_visible),
                dtype=tf.float32, initializer=tf.random_normal_initializer())
        self.w_prime = tf.transpose(self.w)
        # visible input
        if x is None:
            self.x = tf.placeholder(tf.float32, (None, n_visible))
        else:
            self.x = x
            self.is_stacked = True
        # noise mask for corruptting input x
        noise_mask  = np.random.binomial(1, 1 - corrupt_lv, (self.batch_size, n_visible)).astype('float32')
        corrupted_x = tf.multiply(noise_mask, self.x)
        # hidden encode
        self.z = tf.nn.sigmoid(tf.add(tf.matmul(corrupted_x, self.w), self.b_vis))
        self.z = tf.nn.dropout(self.z, keep_prob) # probability to keep units
        # reconstructed input
        self.x_hat     = tf.nn.sigmoid(tf.add(tf.matmul(self.z, self.w_prime), self.b_hid))
        # define loss and optimizer, minimize the mean squared error
        self.cost      = tf.reduce_mean(tf.pow(self.x_hat - self.x, 2))
        self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.cost)

    def get_hidden_output(self, sess, x):
        '''
        Calculate hidden output (z) given input x if the dA is not stacked.
        '''
        assert not self.is_stacked, 'It is a stacked dA, which is unable to calculate hidden output.'
        return sess.run(self.z, feed_dict={self.x: x})

    def get_reconstructed_x(self, sess, x):
        '''
        Calculate reconstructed x (x_hat) given input x if the dA is not stacked.
        '''
        assert not self.is_stacked, 'It is a stacked dA, which is unable to calculate reconstructed x.'
        return sess.run(self.x_hat, feed_dict={self.x: x})

    def fit(self, sess, train_x, test_x, pretrained=False, input_tensor=None):
        '''
        Fit a basic two layers autoencoder.
        If the dA is stacked, input_tensor has to be specified as a tensor.
        '''
        if not pretrained:
            # initialize the session in tensorflow
            init = tf.global_variables_initializer()
            sess.run(init)
        # number of dataset
        n_trains  = train_x.shape[0]
        n_tests   = test_x.shape[0]
        # number of batches
        n_batches = int(n_trains / self.batch_size)
        # abort the program
        # if the batch size is indivisible w.r.t the size of dataset,
        # or input tensor is undefined when the dA is stacked.
        assert not n_trains % self.batch_size, 'Indivisible batch size.'
        assert n_tests > self.batch_size, 'Size of test data should be larger than batch size.'
        assert (not self.is_stacked) or (input_tensor is not None), 'Input tensor is undefined since the dA is stacked.'
        # train iteratively
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
                # use input_tensor as the input of the model if the model is stacked
                # otherwise use self.x
                if self.is_stacked:
                    placeholder = input_tensor
                else:
                    placeholder = self.x
                # Optimize autoencoder
                sess.run(self.optimizer, feed_dict={placeholder: batch_x})
                # loss for train data and test data
                train_loss = sess.run(self.cost, feed_dict={placeholder: batch_x})
                test_loss  = sess.run(self.cost, feed_dict={placeholder: sample_test_x})
                # append results to list
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
                 keep_prob=0.05, lr=0.1, batch_size=64, n_epoches=10,
                 corrupt_lvs=[0.1, 0.2, 0.3]):
        '''
        '''
        self.lr        = lr
        self.dA_layers = []
        self.n_layers  = len(hidden_layers_sizes)
        self.x         = tf.placeholder(tf.float32, (None, n_visible))
        # construct multiple layers of dAs
        for i in range(self.n_layers):
            # input size and input variable of current dA layer
            if i == 0:
                input_size  = n_visible
                layer_input = self.x
            else:
                input_size  = hidden_layers_sizes[i - 1]
                layer_input = self.dA_layers[i - 1].z
            # output size of current dA layer
            output_size = hidden_layers_sizes[i]
            # initialize current dA layer
            dA_layer = dA(model_name='stacked_layer_%d_dA' % i,
                          n_visible=input_size, n_hidden=output_size,
                          x=layer_input, keep_prob=keep_prob, lr=lr,
                          batch_size=batch_size, n_epoches=n_epoches,
                          corrupt_lv=corrupt_lvs[i])
            # append current dA layer to the list
            self.dA_layers.append(dA_layer)
        # construct supervised layer for fine tuning
        self.supervised_layer()

    def supervised_layer(self):
        '''
        '''
        self.y             = tf.placeholder(tf.float32, (None, 1))
        self.logistic_pred = tf.nn.softmax(self.dA_layers[-1].z) # Softmax
        self.finetune_cost = tf.reduce_mean(-tf.reduce_sum(self.y * tf.log(logistic_pred), reduction_indices=1))
        self.optimizer     = tf.train.GradientDescentOptimizer(self.lr).minimize(self.finetune_cost)

    def pretrain(self, sess, train_x, test_x, pretrained=False):
        '''
        Generates a list of functions, each of them implementing one
        step in trainnig the dA corresponding to the layer with same index.
        The function will require as input the minibatch index, and to train
        a dA you just need to iterate, calling the corresponding function on
        all minibatch indexes.
        '''
        if not pretrained:
            # initialize the session in tensorflow
            init = tf.global_variables_initializer()
            sess.run(init)
        # pre-train each layer of denoising autoencoder
        for dA_layer in self.dA_layers:
            print('[%s] *** Layer (%s) ***' % \
                  (arrow.now(), dA_layer.model_name),
                  file=sys.stderr)
            print('[%s] visible size: %d, hidden size: %d, corruption level: %f' % \
                  (arrow.now(), dA_layer.n_visible, dA_layer.n_hidden, dA_layer.corrupt_lv),
                  file=sys.stderr)
            dA_layer.fit(sess, train_x, test_x, pretrained=True, input_tensor=self.x)

    def fine_tune(self, train_x, train_y, test_x, test_y):
        '''
        '''
        







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

    n_visible = train_x.shape[1]
    n_hidden  = 100

    with tf.Session() as sess:
        # da = dA('test', n_feature, n_hidden,
        #          keep_prob=0.05, lr=0.005, batch_size=55, n_epoches=10, corrupt_lv=0.2)
        # da.fit(sess, train_x, test_x)
        sda = SdA(n_visible=n_visible, hidden_layers_sizes=[200, 100, 50],
                  keep_prob=0.05, lr=0.005, batch_size=55, n_epoches=5,
                  corrupt_lvs=[0.1, 0.2, 0.3])
        sda.pretrain(sess, train_x, test_x)
        sda.
