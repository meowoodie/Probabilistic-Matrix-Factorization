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
        self.z       = None  # hidden encode output
        self.x       = None  # visible input
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
        self.z       = tf.nn.sigmoid(tf.add(tf.matmul(corrupted_x, self.w), self.b_vis))
        self.clean_z = tf.nn.sigmoid(tf.add(tf.matmul(self.x, self.w), self.b_vis))
        # reconstructed input
        self.x_hat       = tf.nn.sigmoid(tf.add(tf.matmul(tf.nn.dropout(self.z, keep_prob), self.w_prime), self.b_hid))
        self.clean_x_hat = tf.nn.sigmoid(tf.add(tf.matmul(self.clean_z, self.w_prime), self.b_hid))
        # define loss and optimizer, minimize the mean squared error
        self.cost      = tf.reduce_mean(tf.pow(self.x_hat - self.x, 2))
        self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.cost, var_list=[self.w, self.b_vis, self.b_hid])

    def get_hidden_output(self, sess, x):
        '''
        Calculate hidden output (z) given input x if the dA is not stacked.
        '''
        assert not self.is_stacked, 'It is a stacked dA, which is unable to calculate hidden output.'
        return sess.run(self.clean_z, feed_dict={self.x: x})

    def get_reconstructed_x(self, sess, x):
        '''
        Calculate reconstructed x (x_hat) given input x if the dA is not stacked.
        '''
        assert not self.is_stacked, 'It is a stacked dA, which is unable to calculate reconstructed x.'
        return sess.run(self.clean_x_hat, feed_dict={self.x: x})

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
                if self.is_stacked and input_tensor is not None:
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
            avg_train_loss = np.mean(avg_train_loss)
            avg_test_loss  = np.mean(avg_test_loss)
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
                 keep_prob=0.05, pretrain_lr=0.005, finetune_lr=0.1,
                 batch_size=64, n_epoches=5, corrupt_lvs=[0.1, 0.2, 0.3]):
        '''
        '''
        self.finetune_lr = finetune_lr
        self.pretrain_lr = pretrain_lr
        self.dA_layers   = []
        self.n_epoches   = n_epoches
        self.batch_size  = batch_size
        self.n_layers    = len(hidden_layers_sizes)
        self.hidden_layers_sizes = hidden_layers_sizes
        self.x           = tf.placeholder(tf.float32, (None, n_visible))
        # construct forward layers of dAs
        for i in range(self.n_layers):
            # input size and input variable of current dA layer
            if i == 0:
                input_size  = n_visible
                layer_input = self.x
            else:
                input_size  = self.hidden_layers_sizes[i - 1]
                layer_input = self.dA_layers[i - 1].clean_z
            # output size of current dA layer
            output_size = self.hidden_layers_sizes[i]
            # initialize current dA layer
            dA_layer = dA(model_name='stacked_layer_%d_dA' % i,
                          n_visible=input_size, n_hidden=output_size,
                          x=layer_input, keep_prob=keep_prob, lr=self.pretrain_lr,
                          batch_size=batch_size, n_epoches=n_epoches,
                          corrupt_lv=corrupt_lvs[i])
            # append current dA layer to the list
            self.dA_layers.append(dA_layer)
        # construct supervised layer for fine tuning
        self.supervised_layer()
        # construct backward layers of dAs
        self.reconstructed_layer()

    def get_reconstructed_x(self, sess, x):
        '''
        Calculate reconstructed x (x_hat) given input x.
        '''
        return sess.run(self.x_hat, feed_dict={self.x: x})

    def get_predicted_y(self, sess, x):
        '''
        Calculate predicted y given input x.
        '''
        return sess.run(self.pred, feed_dict={self.x: x})

    def reconstructed_layer(self):
        '''
        Multiple layers for reconstructing input x. The reconstructed layer
        essentially stacked each reconstructed layer of the dAs and use the last
        output as the reconstructed x.
        '''
        self.x_hat = self.dA_layers[-1].clean_z
        for i in list(reversed(range(self.n_layers))):
            w_prime    = self.dA_layers[i].w_prime
            b_hid      = self.dA_layers[i].b_hid
            self.x_hat = tf.nn.sigmoid(tf.add(tf.matmul(self.x_hat, w_prime), b_hid))

    def supervised_layer(self, n_output=10):
        '''
        A customized supervised layer for fine tuning SdA. This function can be
        highly overrided in accordance with the requirements of the application.
        In this case, the supervised layer is a fully connected softmax layer
        which is used to predict the label of the input x. In general, the
        supervised layer has to include a supervised (response) variable
        `self.y`, a `self.finetune_cost` for monitoring the loss of the
        fine-tuning, and a `self.optimizer` to minimize the difference between
        prediction (depend on `x`) and `y`.
        '''
        # weights and bias for the supervised layer (softmax layer)
        w = tf.get_variable(name='softmax_weights', shape=(self.dA_layers[-1].n_hidden, n_output),
                            dtype=tf.float32, initializer=tf.random_normal_initializer())
        b = tf.get_variable(name='softmax_bias', shape=(n_output),
                            dtype=tf.float32, initializer=tf.random_normal_initializer())
        # label indicating the number of the image
        self.y    = tf.placeholder(tf.float32, (None, 1))
        one_hot_y = tf.one_hot(tf.cast(tf.reshape(self.y, [-1]), tf.int32), 10)
        # prediction made by a softmax layer
        self.pred = tf.nn.softmax(tf.add(tf.matmul(self.dA_layers[-1].clean_z, w), b))
        # cross-entrophy cost for softmax regression
        self.finetune_cost = tf.reduce_mean(-tf.reduce_sum(one_hot_y * tf.log(self.pred + 1e-20), axis=1))
        # accuracy of the predcition
        self.correct_pred  = tf.equal(tf.argmax(self.pred, axis=1), tf.argmax(one_hot_y, axis=1))
        self.accuracy      = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))
        # optimizer of the supervised layer
        self.optimizer     = tf.train.GradientDescentOptimizer(self.finetune_lr).minimize(self.finetune_cost)

    def pretrain(self, sess, train_x, test_x, pretrained=False):
        '''
        Pretrain the SdA only with the input x without supervision.
        '''
        if not pretrained:
            # initialize the session in tensorflow
            init = tf.global_variables_initializer()
            sess.run(init)
        # pre-train each layer of denoising autoencoder
        print('[%s] --- Pre-train Phase ---' % arrow.now(), file=sys.stderr)
        for dA_layer in self.dA_layers:
            print('[%s] *** Layer (%s) ***' % \
                  (arrow.now(), dA_layer.model_name),
                  file=sys.stderr)
            print('[%s] visible size: %d, hidden size: %d, corruption level: %f' % \
                  (arrow.now(), dA_layer.n_visible, dA_layer.n_hidden, dA_layer.corrupt_lv),
                  file=sys.stderr)
            dA_layer.fit(sess, train_x, test_x, pretrained=True, input_tensor=self.x)

    def finetune(self, sess, train_x, train_y, test_x, test_y, pretrained=True):
        '''
        Fine tune the model with labeling information y.
        '''
        if not pretrained:
            # initialize the session in tensorflow
            init = tf.global_variables_initializer()
            sess.run(init)
        print('[%s] --- Fine-tune Phase ---' % arrow.now(), file=sys.stderr)
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
        assert train_x.shape[0] == train_y.shape[0] and test_x.shape[0] == test_y.shape[0], \
               'the size of y and x are inconsistant.'
        # train iteratively
        e = 0
        while e < self.n_epoches:
            e += 1
            # shuffle training samples
            shuffled_order = np.arange(n_trains)
            np.random.shuffle(shuffled_order)
            # training iterations over batches
            avg_train_loss = []
            avg_train_acc  = []
            avg_test_loss  = []
            avg_test_acc   = []
            for batch in range(n_batches):
                idx       = np.arange(self.batch_size * batch, self.batch_size * (batch + 1))
                batch_idx = np.mod(idx, n_trains).astype('int32')
                # training entries selected in current batch
                batch_x       = train_x[shuffled_order[batch_idx], :]
                batch_y       = train_y[shuffled_order[batch_idx], :]
                sample_idx    = np.random.choice(n_tests, self.batch_size)
                sample_test_x = test_x[sample_idx, :]
                sample_test_y = test_y[sample_idx, :]
                # Optimize autoencoder
                sess.run(self.optimizer, feed_dict={self.x: batch_x, self.y: batch_y})
                # loss for train data and test data
                train_loss, train_acc = sess.run(
                    [self.finetune_cost, self.accuracy],
                    feed_dict={self.x: batch_x, self.y: batch_y})
                test_loss, test_acc   = sess.run(
                    [self.finetune_cost, self.accuracy],
                    feed_dict={self.x: sample_test_x, self.y: sample_test_y})
                # append results to list
                avg_train_loss.append(train_loss)
                avg_train_acc.append(train_acc)
                avg_test_loss.append(test_loss)
                avg_test_acc.append(test_acc)
            # training log ouput
            avg_train_loss = np.mean(avg_train_loss)
            avg_train_acc  = np.mean(avg_train_acc)
            avg_test_loss  = np.mean(avg_test_loss)
            avg_test_acc   = np.mean(avg_test_acc)
            print('[%s] Epoch %d' % (arrow.now(), e), file=sys.stderr)
            print('[%s] Training loss:\t%f' % (arrow.now(), avg_train_loss), file=sys.stderr)
            print('[%s] Testing loss:\t%f' % (arrow.now(), avg_test_loss), file=sys.stderr)
            print('[%s] Training accuracy:\t%f' % (arrow.now(), avg_train_acc), file=sys.stderr)
            print('[%s] Testing accuracy:\t%f' % (arrow.now(), avg_test_acc), file=sys.stderr)



if __name__ == '__main__':
    # An unittest on MNIST data for Denoising Autoencoder

    # Load MNIST data in a format suited for tensorflow.
    # The script input_data is available under this URL:
    # https://raw.githubusercontent.com/tensorflow/tensorflow/master/tensorflow/examples/tutorials/mnist/input_data.py
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)
    train_x = np.vstack([img.reshape(-1,) for img in mnist.train.images])
    train_y = np.reshape(mnist.train.labels, (train_x.shape[0], 1))

    print(train_x.shape)
    print(train_y.shape)

    test_x  = np.vstack([img.reshape(-1,) for img in mnist.test.images])
    test_y  = np.reshape(mnist.test.labels, (test_x.shape[0], 1))

    print(test_x.shape)
    print(test_y.shape)

    n_visible = train_x.shape[1]
    n_hidden  = 700

    from utils import show_mnist_images
    with tf.Session() as sess:
        # Single Denoising Autoencoder
        # - This is the best params that I can find for reconstructing the
        #   best quality of images

        # da = dA('test', n_visible, n_hidden,
        #          keep_prob=0.05, lr=0.01, batch_size=1000, n_epoches=25, corrupt_lv=0.1)
        # da.fit(sess, train_x, test_x)
        # reconstructed_x = da.get_reconstructed_x(sess, test_x[0:10])
        # # Plot reconstructed mnist figures
        # show_mnist_images(reconstructed_x)
        # show_mnist_images(test_x[0:10])

        # Stacked Denoising Autoencoder
        sda = SdA(n_visible=n_visible, hidden_layers_sizes=[700, 600],
                  keep_prob=0.05, pretrain_lr=1e-1, finetune_lr=1e-1,
                  batch_size=1000, n_epoches=20, corrupt_lvs=[0.1, 0.1])
        sda.pretrain(sess, train_x, test_x, pretrained=False)
        sda.finetune(sess, train_x, train_y, test_x, test_y, pretrained=True)

        test_sample     = test_x[0:10]
        reconstructed_x = sda.get_reconstructed_x(sess, test_sample)

        # Plot reconstructed mnist figures
        show_mnist_images(reconstructed_x)
        show_mnist_images(test_sample)
