#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Probabilistic Matrix Factorization Model and a simple unittest on Amazon Product
Dataset.

References:
- Collaborative Deep Learning for Recommender Systems
  https://arxiv.org/pdf/1409.2944.pdf
- Probabilistic Matrix Factorization
  https://papers.nips.cc/paper/3208-probabilistic-matrix-factorization.pdf
- Stacked Denoising Autoencoders
  http://www.jmlr.org/papers/volume11/vincent10a/vincent10a.pdf
'''

import sys
import arrow
import numpy as np
from numpy.linalg import norm
from prec.dataloader import read_ratings

class PMF(object):
    '''
    Probabilistic Matrix Factorization
    '''

    def __init__(self, n_feature, epsilon, lam, n_epoches, n_batches):
        self.n_feature = n_feature  # number of features
        self.epsilon   = epsilon    # epsilon for leanring rate
        self.lam       = lam        # lambda for L2 regularization

        self.n_epoches = n_epoches  # number of epoches
        self.n_batches = n_batches  # number of batches

        self.V = None # items feature matrix
        self.U = None # users feature matrix

    def loss(self, ratings):
        '''
        Loss Function for evaluating matrix U and V
        '''
        errors = [
            (float(r_ij) - np.dot(self.U[i], self.V[j].T))**2 + \
            self.lam * norm(self.U[i]) + self.lam * norm(self.V[j])
            for i, j, r_ij in ratings]
        return sum(errors)

    def sgd_update(self, ratings):
        '''
        Update matrix U and V by Stochastic Gradient Descent.
        '''
        for i, j, r_ij in ratings:
            r_ij_hat = np.dot(self.U[i], self.V[j].T)
            grad_U_i = (r_ij_hat - float(r_ij)) * self.V[j] + self.lam * self.U[i]
            grad_V_j = (r_ij_hat - float(r_ij)) * self.U[i] + self.lam * self.V[j]
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
        # Initialization
        if self.V is None or self.U is None:
            self.e = 0
            self.U = 0.1 * np.random.randn(n_users, self.n_feature)
            self.V = 0.1 * np.random.randn(n_items, self.n_feature)
        # training iterations over epoches
        while self.e < self.n_epoches:
            self.e += 1
            # shuffle training samples
            shuffled_order = np.arange(n_trains)
            np.random.shuffle(shuffled_order)
            # training iterations over batches
            avg_train_loss = []
            avg_test_loss  = []
            batch_size     = int(n_trains / self.n_batches)
            for batch in range(self.n_batches):
                idx       = np.arange(batch_size * batch, batch_size * (batch + 1))
                batch_idx = np.mod(idx, n_trains).astype('int32')
                # training ratings selected in current batch
                batch_ratings = train_ratings[shuffled_order[batch_idx], :]
                # test ratings sample with the same size as the training batch
                sample_test_ratings = test_ratings[np.random.choice(len(test_ratings), batch_size), :]
                # update U and V by sgd in a close-formed gradient
                self.sgd_update(batch_ratings)
                # loss for training and testing U, V and ratings
                train_loss = self.loss(batch_ratings)
                test_loss  = self.loss(sample_test_ratings)
                avg_train_loss.append(train_loss)
                avg_test_loss.append(test_loss)
            # training log ouput
            avg_train_loss = np.mean(avg_train_loss) / float(batch_size)
            avg_test_loss  = np.mean(avg_test_loss) / float(batch_size)
            print('[%s] Epoch %d' % (arrow.now(), self.e), file=sys.stderr)
            print('[%s] Training loss:\t%f' % (arrow.now(), avg_train_loss), file=sys.stderr)
            print('[%s] Testing loss:\t%f' % (arrow.now(), avg_test_loss), file=sys.stderr)

if __name__ == '__main__':
    # Load sample data
    ratings        = np.loadtxt("resource/output/ratings_np_mat.txt", delimiter=",").astype('int32')
    # shuffle dataset
    shuffled_order = np.arange(len(ratings))
    np.random.shuffle(shuffled_order)
    ratings        = ratings[shuffled_order]
    # cross validation
    train_ratings  = ratings[0:490000, :]
    test_ratings   = ratings[490000:, :]
    print(train_ratings.shape)
    print(test_ratings.shape)

    # ratings = read_ratings('resource/output/micro_ratings.txt')
    # train_ratings = ratings[0:900, :]
    # test_ratings  = ratings[900:, :]

    # Substantiate PMF
    pmf = PMF(n_feature=100, epsilon=0.1, lam=0.1, n_epoches=10, n_batches=1000)
    # Train PMF
    pmf.fit(train_ratings, test_ratings)
