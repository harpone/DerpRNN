from __future__ import division, print_function
import numpy as np
import matplotlib.pyplot as plt
import theano
import os
import sys
import pickle

from theano import tensor as T

from models.deeprnn import DeepRNN, InvariantDeepRNN
from utils.utils import *

from ctools.nonpytools import *

plot = plt.plot

print('Theano version: {}'.format(theano.version.full_version))


###########################################
## MUSIC ##################################
###########################################

#dataset = np.load('selected.npz')['arr_0'].tolist()
dataset = np.load('classics.npz')['arr_0'].tolist()

new_dataset = []

for n, song in enumerate(dataset):
    #print(len(song))
    if len(song) > 300:
        new_dataset.append(song)

dataset = new_dataset
dataset = dataset[0:50]

state = 'all_noninvariant_5_5'

model = DeepRNN(depth=5,
                 width=5,
                 readout_width=1,
                 readin_layer='tanh',
                 rnn_layer='gru',
                 readout_layer='rbm',
                 n_visible=88,
                 optimizer='adadelta',
                 state_from_file=state,
                 sparsity=.5,
                 readin_input_scale=8.,
                 recurrence_spectral_radius=1.2,
                 recurrence_input_scale=2.,
                 readout_scale=4.9,
                 dropout=0.,
                 input_dropout=0.,
                 input_noise=0.005)

# Phase 1
model.train(dataset,
            lr=.1,                     # .1 or 1. !!!
            min_batch_size=200,
            max_batch_size=200,
            num_epochs=1000,
            save_as=state,
            gamma=.99)



