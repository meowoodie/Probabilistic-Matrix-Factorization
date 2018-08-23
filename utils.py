#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Scipts for various of helper functions.
'''

import numpy as np
import matplotlib.pyplot as plt

def show_mnist_images(x):
    assert x.shape[1] == 784, '(%d, %d) is an invalid shape of x.' % x.shape
    x = x.reshape([x.shape[0], 28, 28])

    plt.figure()
    plt.gray()
    plt.imshow(x[0])
    plt.show()
