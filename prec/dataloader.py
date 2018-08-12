#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Data Loader
"""

import sys
import arrow
import numpy as np

def read_ratings(ratings_filename, users_filename=None, items_filename=None):
    with open(ratings_filename, 'r') as rating_fr:
        ratings = [ line.split('\t')[0:3] for line in rating_fr ]
        if users_filename is None or items_filename is None:
            users = list(set([ rating[0] for rating in ratings ]))
            items = list(set([ rating[1] for rating in ratings ]))
        else:
            with open(users_filename, 'r') as users_fr, \
                 open(items_filename, 'r') as items_fr:
                users = [ line for line in users_fr ]
                items = [ line for line in items_fr ]
        # convert user and item to numerical index.
        i = 0
        ratings_matrix = []
        for rating in ratings:
            ratings_matrix.append([ users.index(rating[0]), items.index(rating[1]), float(rating[2]) ])
            if i % 1000 == 0 and i != 0:
                print('[%s] %d ratings have been processed.' % (arrow.now(), i), file=sys.stderr)
            i += 1
        return np.array(ratings_matrix, dtype='int32')

if __name__ == '__main__':
    ratings = read_ratings('resource/output/ratings.txt')
    np.savetxt('resource/output/ratings_np_mat.txt', ratings, delimiter=',')
