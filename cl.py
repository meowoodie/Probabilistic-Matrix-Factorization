#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Context Learner
'''

import tensorflow as tf
import numpy as np
import arrow
import sys

class CL(object):
    '''
    '''

    def __init__(self, n_hidden, batch_size=64, n_epoches=10):
        self.num_events     = num_events
        self.embedding_size = embedding_size
        self.batch_size     = batch_size
        # self.iters          = iters
        # self.display_step   = display_step

        #TODO: let tfidf of corpus be init embeddings
        self.embeddings = tf.Variable(init_embeddings, dtype=tf.float32)
            # tf.random_uniform(
            #     [self.num_events, self.embedding_size],
            #     -1.0, 1.0))
        self.nce_weights = tf.Variable(
            tf.truncated_normal(
                [self.num_events, self.embedding_size],
                stddev=1.0/math.sqrt(self.embedding_size)))
        self.nce_biases  = tf.Variable(
            tf.zeros([self.num_events]))

        # Placeholders for inputs
        # - target events
        self.train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
        # - context events
        self.train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])

        # Build mapping between embeddings and inputs
        embed = tf.nn.embedding_lookup(self.embeddings, self.train_inputs)

        # Compute the NCE loss, using a sample of the negative labels each time.
        self.loss = tf.reduce_mean(
            tf.nn.nce_loss(weights=self.nce_weights,
                           biases=self.nce_biases,
                           labels=self.train_labels,
                           inputs=embed,
                           num_sampled=num_sampled,
                           num_classes=self.num_events))

        # Optimizer
        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr).minimize(self.loss)

        # Compute the cosine similarity between minibatch examples and all embeddings.
        norm = tf.sqrt(tf.reduce_sum(tf.square(self.embeddings), 1, keep_dims=True))
        self.normalized_embeddings = self.embeddings / norm
