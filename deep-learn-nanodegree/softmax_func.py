#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 12:41:08 2018

@author: manib
"""

import numpy as np

# Write a function that takes as input a list of numbers, and returns
# the list of values given by the softmax function.
def softmax(L):
    exp_values = np.exp(L)
    total = np.sum(exp_values)
    result = exp_values / total
    return result.tolist()

# Write a function that takes as input two lists Y, P,
# and returns the float corresponding to their cross-entropy.
def cross_entropy(Y, P):
    Yarr = np.array(Y)
    Parr = np.array(P)
    cross_entr = np.sum(-Yarr*np.log(Parr) - (1-Yarr)*np.log(1 - Parr))
    return cross_entr