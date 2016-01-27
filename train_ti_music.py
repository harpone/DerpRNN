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
dataset = dataset[0:5]

state = 'all_invariant_10_5_constrained'

while True:

    try:
        model = InvariantDeepRNN(depth=10,
                         width=5,
                         readout_width=1,
                         readin_layer='tanh',
                         rnn_layer='gru',
                         readout_layer='rbm',
                         n_visible=88,
                         optimizer='adadelta',
                         state_from_file=state,  ### !!!!!
                         sparsity=1.,
                         readin_input_scale=8.,
                         recurrence_spectral_radius=4.,
                         recurrence_input_scale=4.,
                         readout_scale=4.9,
                         dropout=0.,
                         input_dropout=0.,
                         input_noise=0.,
                         constrain_eigvals=True)

        # # Phase 1
        # model.train(dataset,
        #             lr=1.,                     # .1 or 1. !!!
        #             min_batch_size=25,
        #             max_batch_size=50,
        #             num_epochs=10,
        #             save_as=state,
        #             gamma=.9)

        # Phase 2
        model.train(dataset,
                    lr=.1,                     # .1 or 1. !!!
                    min_batch_size=200,
                    max_batch_size=200,
                    num_epochs=100,
                    save_as=state,
                    gamma=.9)

    except ValueError:
        print('\nStarting over...')
        sys.stdout.flush()
        continue

    except KeyboardInterrupt:
        print('\nInterrupted by user... hit Ctrl+C one more time to break!')
        break



