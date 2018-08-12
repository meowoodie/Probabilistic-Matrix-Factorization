#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

"""

import sys
import arrow
import numpy as np
from numpy.linalg import norm
from prec.dataloader import read_ratings

class PMF(object):
    '''
    Probabilistic Matrix Factorization
    '''

    def __init__(self, n_feature, epsilon, lam, momentum, n_epoches, n_batches):
        self.n_feature = n_feature  # number of features
        self.epsilon   = epsilon    # epsilon for leanring rate
        self.lam       = lam        # lambda for L2 regularization
        # self.momentum  = momentum   # momentum for gradient descent
        self.n_epoches = n_epoches  # number of epoches
        self.n_batches = n_batches  # number of batches

        self.V = None # items feature matrix
        self.U = None # users feature matrix

    def loss(self, ratings, U, V):
        '''
        Training Loss Function.

        - ratings: a collection of rating triple (user, item, rating).
        - U:       users feature matrix
        - V:       items feature matrix
        '''
        errors = [
            (r_ij - np.dot(U[i] * V[j].T))**2 + self.lam * norm(U[i]) + self.lam * norm(V[j])
            for i, j, r_ij in ratings ]
        return sum(errors)

    def sgd_update(self, ratings):
        '''
        Update matrix U and V by Stochastic Gradient Descent.
        '''
        for i, j, r_ij in ratings:
            r_ij_hat = np.dot(self.U[i], self.V[j].T)
            grad_U_i = (r_ij_hat - r_ij) * self.V[j] + self.lam * U[i]
            grad_V_j = (r_ij_hat - r_ij) * self.U[i] + self.lam * V[j]
            self.U[i] = self.U[i] - self.epsilon * grad_U_i
            self.V[j] = self.V[j] - self.epsilon * grad_V_j

    def fit(self, train_ratings, test_ratings):
        '''
        Fit PMF model with respect to the ratings. A rating is a triple (user,
        item, rating), in particular, user and item are integers to indicate
        unique ids respectively, and rating is a real value score that associates
        with corresponding user and item. For here, ratings is a numpy array
        with shape (n, 3).

        Params:
        - train_ratings: ratings entries for training purpose
        - test_ratings:  ratings entries for testing purpose
        '''
        # get number of training samples and testing samples
        n_trains = train_ratings.shape[0]
        n_tests  = test_ratings.shape[0]
        # get number of items and number of users
        n_users  = int(max(np.amax(train_ratings[:, 0]), np.amax(test_ratings[:, 0]))) + 1
        n_items  = int(max(np.amax(train_ratings[:, 1]), np.amax(test_ratings[:, 1]))) + 1
        # user and item for testing
        test_users = np.array(list(set(test_ratings[:, 0])), dtype='int32')
        test_items = np.array(list(set(test_ratings[:, 1])), dtype='int32')
        # testing U, V
        test_U = self.U[test_users, :]
        test_V = self.V[test_items, :]
        # Initialization
        if self.V is None or self.U is None:
            self.e = 0
            self.U = 0.1 * np.random.randn(n_users, self.n_feature)
            self.V = 0.1 * np.random.randn(n_items, self.n_feature)
            # delta_U = np.zeros((n_users, self.n_feature))
            # delta_V = np.zeros((n_items, self.n_feature))
        # training iterations over epoches
        while self.e < self.n_epoches:
            self.e += 1
            # shuffle training samples
            shuffled_order = np.arange(n_trains)
            np.random.shuffle(shuffled_order)
            # training iterations over batches
            avg_train_loss = []
            avg_test_loss  = []
            for batch in range(self.n_batches):
                idx       = np.arange(self.batch_size * batch, self.batch_size * (batch + 1))
                batch_idx = np.mod(idx, n_trains)
                # users and items appeared in current batch
                batch_users = np.array(list(set(train_ratings[shuffled_order[batch_idx], 0])), dtype='int32')
                batch_items = np.array(list(set(train_ratings[shuffled_order[batch_idx], 1])), dtype='int32')
                # training U, V and ratings selected in current batch
                batch_U = self.U[batch_users, :]
                batch_V = self.V[batch_items, :]
                batch_ratings = train_ratings[shuffled_order[batch_idx], :]
                # update U and V by sgd in a close-formed gradient
                sgd_update(batch_ratings)
                # loss for training and testing U, V and ratings
                train_loss = loss(self, batch_ratings, batch_U, batch_V)
                test_loss  = loss(self, test_ratings, test_U, test_V)
                avg_train_loss.append(train_loss)
                avg_test_loss.append(test_loss)
            # training log ouput
            avg_train_loss = mean(avg_train_loss)
            avg_test_loss  = mean(avg_test_loss)
            print('[%s] Epoch %d' % (arrow.now(), self.e), file=sys.stderr)
            print('[%s] Training loss:\t%f' % (arrow.now(), avg_train_loss), file=sys.stderr)
            print('[%s] Testing loss:\t%f' % (arrow.now(), avg_test_loss), file=sys.stderr)

if __name__ == '__main__':
