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

#######################################
#### TEXT #############################
#######################################
from keras.datasets.data_utils import get_file

path = get_file('nietzsche.txt', origin="https://s3.amazonaws.com/text-datasets/nietzsche.txt")
text = open(path).read().lower()
print('corpus length:', len(text))

chars = set(text)
print('total chars:', len(chars))

stripped_chars = {'\n',' ','!','"',"'",'(',')',',','-','.','0','1','2',
                  '3','4','5','6','7','8','9',':',';','?','a','b','c','d',
                  'e','f','g','h','i','j','k','l','m','n','o','p','q','r',
                  's','t','u','v','w','x','y','z'}

char_indices = dict((c, i) for i, c in enumerate(stripped_chars))
indices_char = dict((i, c) for i, c in enumerate(stripped_chars))


print('Vectorization...')
dataset = np.zeros((len(text), len(stripped_chars)), dtype=theano.config.floatX)
for t, char in enumerate(text):
    if char in stripped_chars:
        dataset[t, char_indices[char]] = 1

text_recon = ''
for row in dataset:
    idx = np.argmax(row)
    text_recon += indices_char[idx]

text_file = open("nietzsche_recon.txt", "w")
text_file.write(text_recon)
text_file.close()

load_state = 'saved_model_parameters/text_5_10_m0.016'
state = 'text_5_10_new'
n_visible = dataset.shape[1]

model = DeepRNN(depth=5,
                 width=10,
                 readout_width=1,
                 readin_layer='tanh',
                 rnn_layer='gru',
                 readout_layer='softmax',
                 n_visible=n_visible,
                 optimizer='adadelta',
                 state_from_file=load_state,
                 sparsity=1.,
                 readin_input_scale=2.,
                 recurrence_spectral_radius=1.2,
                 recurrence_input_scale=1.,
                 readout_scale=1.,
                 dropout=0.,
                 input_dropout=0.,
                 input_noise=0.)


# # Phase 2
# model.train(dataset,
#             lr=10.,                     # .1 or 1. !!!
#             min_batch_size=100,
#             max_batch_size=100,
#             num_epochs=100,
#             save_as=state,
#             gamma=.95)                 # .9 or .99 !!!


# Phase 3
model.train(dataset,
            lr=.1,                     # .1 or 1. !!!
            min_batch_size=200,
            max_batch_size=200,
            num_epochs=100,
            save_as=state,
            gamma=.99)                 # .9 or .99 !!!