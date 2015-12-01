__author__ = 'heka'


from __future__ import division, print_function

__author__ = 'Heikki Arponen (heka)'
#TODO:
# - validation error
# - save best (val. error) parameters
# - MIDI: out activation
# - time shift invariant spectrum?? (spec rad seems to decrease often...)

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import theano
import os
from theano import tensor as T
from theano.tensor.signal.conv import conv2d
from collections import OrderedDict

# local imports:
from utils.utils import *


plot = plt.plot

# Theano helper functions:
vec1 = T.dmatrix()
vec2 = T.dmatrix()
veclen = vec1.shape[1]

conv1d_expr = conv2d(vec1, vec2, image_shape=(1, veclen), border_mode='valid')

conv1d = theano.function([vec1, vec2], outputs=conv1d_expr)




class RNNModel(object):

    def __init__(self, nh, ni,
                 out_activation=None,
                 simulation_rounded=False,
                 round_scale=1.,
                 decay_rate=.99,
                 in_hid_scale=.1,
                 hid_out_scale=.1,
                 connections=15,
                 spectral_scale=1.1):
        """
        - Uses multiple 1-step predictions in cost function
        nh :: dimension of the hidden layer
        ni :: dimension of input data
        time  :: initialization/ burn-in period

        -Use this for e.g. music (input vector = notes and has multiple 1's) or
        text (input is a char vector and has one 1).
        - Hidden activation is tanh
        - Output layer activation is sigmoid or something else
        - Output dim = input dim
        - Input data is boolean valued vectors


        """

        assert connections < nh, "Error: 'connections' must be smaller than # of hidden units!"
        self.connections = connections
        self.spectral_scale = spectral_scale

        if out_activation is 'tanh':
            self.out_activation = T.tanh
        else:
            self.out_activation = T.nnet.sigmoid

        self.simulation_rounded = simulation_rounded
        self.round_scale = round_scale

        # parameters of the model
        self.in_hid_scale = in_hid_scale
        self.hid_out_scale = hid_out_scale
        self.ni = ni
        self.nh = nh
        # Initialize parameters:
        self.params, self.names, self.caches = self.initialize_params()

        # Input as matrix:
        x = T.matrix('Input')  # x.shape = (# timesteps, ni)
        # -- time is always the first dimension!
        # -- x_t.shape = (ni,), h_t.shape = (nh,)

        ####################################
        ### Recurrence in training mode: ###
        ####################################
        [h_trn, x_trn], _ = theano.scan(fn=self.recurrence,
                                        sequences=x,
                                        outputs_info=[self.h0, None])

        # cost, gradients and learning rate
        lr = T.scalar('Learn rate')
        burn_in = T.iscalar('Burn-in')  # removed for now
        #burn_in = 0
        cost = T.mean(((x[burn_in + 1:] - x_trn[burn_in:-1, 0, :]) ** 2))
        gradients = T.grad(cost, self.params)

        # Gradient updates:
        updates = OrderedDict((p, p - lr * g / T.sqrt(ch + 1e-8))
                              for p, g, ch in zip(self.params, gradients, self.caches))
        # Keep hidden state for sequential learning:
        updates.update({self.h0: h_trn[-1, :, :]})
        # update caches:
        cache_updates = OrderedDict((ch, decay_rate * ch + (1 - decay_rate) * g ** 2)
                                   for ch, g in zip(self.caches, gradients))
        updates.update(cache_updates)

        # theano functions
        self.cost_value = theano.function(inputs=[x, burn_in],
                                          outputs=cost)
        self.train_on_batch = theano.function(inputs=[x, lr, burn_in],
                                              outputs=cost,
                                              updates=updates)

        ######################################
        ### Recurrence in generative mode: ###
        ######################################
        timesteps = T.iscalar('Timesteps')
        # Initial data:
        x0 = T.fmatrix('Initial x')
        #x0.tag.test_value = np.random.rand(9, 1).astype(np.float32)
        # pass through initial data:
        [h_gen, _], _ = theano.scan(fn=self.recurrence,
                                   sequences=x0,
                                   outputs_info=[self.h0, None])
        # assign initial h and x:
        h_init = h_gen[-2, :, :]  # shape (times, 1, nh)
        x_init = T.reshape(x0[-1], (1, ni))

        [h_gen, x_gen], _ = theano.scan(fn=self.recurrence_sim,
                                        outputs_info=[h_init, T.unbroadcast(x_init, 0)],
                                        n_steps=timesteps)
        # about the T.unbroadcast: https://github.com/Theano/Theano/issues/2985

        # theano functions
        self.simulate = theano.function(inputs=[x0, timesteps],
                                        outputs=[h_gen, x_gen],
                                        updates=[(self.h0, h_gen[-1, :, :])])

    def initialize_params(self):

        ni = self.ni
        nh = self.nh
        in_hid_scale = self.in_hid_scale
        hid_out_scale = self.hid_out_scale

        # get initial values:
        W_hid_val = get_random_sparse(nh, self.connections)
        rho = get_spectral_radius(W_hid_val)
        W_hid_val *= self.spectral_scale / rho
        W_in_val = np.random.randn(nh, ni) * in_hid_scale
        W_out_val = np.random.randn(ni, nh) * hid_out_scale

        self.W_hid = theano.shared(W_hid_val.astype(theano.config.floatX))
        self.W_in = theano.shared(in_hid_scale * W_in_val.astype(theano.config.floatX))
        self.W_out = theano.shared(hid_out_scale * W_out_val.astype(theano.config.floatX))
        self.b_hid = theano.shared(np.zeros((1, nh), dtype=theano.config.floatX))
        self.b_out = theano.shared(np.zeros((1, ni), dtype=theano.config.floatX))
        self.h0 = theano.shared(np.zeros((1, nh), dtype=theano.config.floatX))  # init h

        # group parameters into list:
        params = [self.W_hid, self.W_in, self.W_out, self.b_hid, self.b_out]
        names = ['Hidden weights', 'Input weights',
                 'Output weights', 'Hidden bias', 'Output bias']

        # initial caches for RMSProp:
        self.W_hid_cache  = theano.shared(np.ones((nh, nh)).astype(theano.config.floatX))
        self.W_in_cache   = theano.shared(np.ones((nh, ni)).astype(theano.config.floatX))
        self.W_out_cache   = theano.shared(np.ones((ni, nh)).astype(theano.config.floatX))
        self.b_hid_cache  = theano.shared(np.ones((1, nh), dtype=theano.config.floatX))
        self.b_out_cache   = theano.shared(np.ones((1, ni), dtype=theano.config.floatX))
        caches = [self.W_hid_cache, self.W_in_cache, self.W_out_cache, self.b_hid_cache, self.b_out_cache]

        return params, names, caches


    def recurrence(self, x_t, h_t):
        h_tp1 = T.tanh(T.dot(h_t, self.W_hid.T) + T.dot(x_t, self.W_in.T) + self.b_hid)
        xhat_tp1 = self.out_activation(T.dot(h_tp1, self.W_out.T) + self.b_out)
        return [h_tp1, xhat_tp1]

    def recurrence_sim(self, h_t, x_t):  # inputs in different order... :/
        scale = self.round_scale
        h_tp1 = T.tanh(T.dot(h_t, self.W_hid.T) + T.dot(x_t, self.W_in.T) + self.b_hid)
        if self.simulation_rounded:
            xhat_tp1 = T.round(scale * (self.out_activation(T.dot(h_tp1, self.W_out.T) + self.b_out)))
        else:
            xhat_tp1 = self.out_activation(T.dot(h_tp1, self.W_out.T) + self.b_out)
        return [h_tp1, xhat_tp1]

    def save(self, folder):
        for param, name in zip(self.params, self.names):
            np.save(os.path.join(folder, name + '.npy'), param.get_value())


class InvariantRNNModel(RNNModel):
    """
    Translation invariant version of the RNNModel (mostly for MIDI data)
    """
    # def __init__(self, nh, ni,
    #              out_activation=None,
    #              decay_rate=.99,
    #              in_hid_scale=.1,
    #              hid_out_scale=.1,
    #              connections=15,
    #              spectral_scale=1.1):
    #
    #     # insert 1d conv definition here??
    #
    #     # get parent initializations:
    #     super(InvariantRNNModel, self).__init__(nh, ni,
    #              out_activation=None,
    #              decay_rate=.99,
    #              in_hid_scale=.1,
    #              hid_out_scale=.1,
    #              connections=15,
    #              spectral_scale=1.1)


    def initialize_params(self):
        """
        Now the parameter *vectors* need to be initialized s.t.
        the hidden W *matrix* has fixed spectral radius and is
        sparsely connected!
        :return:
        """
        ni = self.ni
        nh = self.nh
        in_hid_scale = self.in_hid_scale
        hid_out_scale = self.hid_out_scale

        # get initial values:
        w_hid_val, W_hid_val = get_random_sparse_ti(nh, self.connections)
        rho = get_spectral_radius(W_hid_val)
        w_hid_val *= self.spectral_scale / rho
        u_in_val = np.random.randn(1, ni + nh - 1) * in_hid_scale
        v_out_val = np.random.randn(1, ni + nh - 1) * hid_out_scale

        # def parameters:
        self.w_hid = theano.shared(w_hid_val.astype(theano.config.floatX))
        self.u_in = theano.shared(in_hid_scale * u_in_val.astype(theano.config.floatX))
        self.v_out = theano.shared(hid_out_scale * v_out_val.astype(theano.config.floatX))
        self.b_hid = theano.shared(np.float32(0.))
        self.b_out = theano.shared(np.float32(0.))
        self.h0 = theano.shared(np.zeros((1, nh), dtype=theano.config.floatX))  # init h

        # group parameters into list:
        params = [self.w_hid, self.u_in, self.v_out, self.b_hid, self.b_out]
        names = ['Hidden weights', 'Input weights',
                 'Output weights', 'Hidden bias', 'Output bias']

         # initial caches for RMSProp:
        self.w_hid_cache  = theano.shared(np.ones((1, 2 * nh - 1)).astype(theano.config.floatX))
        self.u_in_cache   = theano.shared(np.ones((1, ni + nh - 1)).astype(theano.config.floatX))
        self.v_out_cache   = theano.shared(np.ones((1, ni + nh - 1)).astype(theano.config.floatX))
        self.b_hid_cache  = theano.shared(np.float32(1.0))
        self.b_out_cache   = theano.shared(np.float32(1.0))

        caches = [self.w_hid_cache, self.u_in_cache, self.v_out_cache, self.b_hid_cache, self.b_out_cache]

        return params, names, caches

    # rewrite with convolution:
    def recurrence(self, x_t, h_t):
        w = self.w_hid
        u = self.u_in
        v = self.v_out
        x_t = T.reshape(x_t, (1, self.ni), ndim=2)
        # def convolutions:
        w_dot_h = conv2d(w, h_t, image_shape=(1, w.shape[1]), border_mode='valid')
        u_dot_x = conv2d(u, x_t, image_shape=(1, u.shape[1]), border_mode='valid')
        # def first recurrence:
        h_tp1 = T.tanh(w_dot_h + u_dot_x + self.b_hid)
        # def new convolution:
        v_dot_htp1 = conv2d(v, h_tp1, image_shape=(1, v.shape[1]), border_mode='valid')
        # ... and recurrence for x:
        xhat_tp1 = self.out_activation(v_dot_htp1 + self.b_out)

        return [h_tp1, xhat_tp1]

    def recurrence_sim(self, h_t, x_t):
        """
        - Different order of inputs
        - x_t not reshaped because no unbroadcast in scan
        """
        scale = self.round_scale
        w = self.w_hid
        u = self.u_in
        v = self.v_out

        # def convolutions:
        w_dot_h = conv2d(w, h_t, image_shape=(1, w.shape[1]), border_mode='valid')
        u_dot_x = conv2d(u, x_t, image_shape=(1, u.shape[1]), border_mode='valid')
        # def first recurrence:
        h_tp1 = T.tanh(w_dot_h + u_dot_x + self.b_hid)
        # def new convolution:
        v_dot_htp1 = conv2d(v, h_tp1, image_shape=(1, v.shape[1]), border_mode='valid')
        # ... and recurrence for x:
        if self.simulation_rounded:
            xhat_tp1 = T.round(self.out_activation(scale * (v_dot_htp1 + self.b_out)))
        else:
            xhat_tp1 = self.out_activation(scale * (v_dot_htp1 + self.b_out))

        return [h_tp1, xhat_tp1]







from __future__ import division, print_function
# Author: Nicolas Boulanger-Lewandowski
# University of Montreal (2012)
# RNN-RBM deep learning tutorial
# More information at http://deeplearning.net/tutorial/rnnrbm.html
#
# Modified by Heikki Arponen (2015)
# heikki@quantmechanics.com

__version__ = "v 0.2"

import sys
import time

import matplotlib.pyplot as plt
import theano
from theano import tensor as T
from theano.tensor.signal.conv import conv2d as conv_signal
#from theano.tensor.shared_randomstreams import RandomStreams
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from theano.compile.debugmode import DebugMode

# Local imports:
from utils.utils import *

#Don't use a python long as this don't work on 32 bits computers.
np.random.seed(0xbeef)
rng = RandomStreams(seed=np.random.randint(1 << 30))
theano.config.warn.subtensor_merge_bug = False
theano.config.exception_verbosity = 'low'
theano.config.compute_test_value = 'off'


# Debug mode:
#mode = 'DebugMode'
mode = None
#mode ='FAST_COMPILE'
theano.config.warn.signal_conv2d_interface = False

#convolution = conv_nnet  # won't work...
convolution = conv_signal

# global parameters:
min_notes_in_batch = 3  # require this many nonzero notes in batch


def build_rbm(v, W, bv, bh, k):
    """Construct a k-step Gibbs chain starting at v for an RBM.

    v : Theano vector or matrix
        If a matrix, multiple chains will be run in parallel (batch).
    W : Theano matrix
        Weight matrix of the RBM.
    bv : Theano vector
        Visible bias vector of the RBM.
    bh : Theano vector
        Hidden bias vector of the RBM.
    k : scalar or Theano scalar
        Length of the Gibbs chain.

    Return a (v_sample, cost, monitor, updates) tuple:

    v_sample : Theano vector or matrix with the same shape as `v`
        Corresponds to the generated sample(s).
    cost : Theano scalar
        Expression whose gradient with respect to W, bv, bh is the CD-k
        approximation to the log-likelihood of `v` (training example) under the
        RBM. The cost is averaged in the batch case.
    monitor: Theano scalar
        Pseudo log-likelihood (also averaged in the batch case).
    updates: dictionary of Theano variable -> Theano variable
        The `updates` object returned by scan."""

    def gibbs_step(v):
        mean_h = T.nnet.sigmoid(T.dot(v, W) + bh)  # shape (timesteps, n_hidden)
        h = rng.binomial(size=mean_h.shape, n=1, p=mean_h,
                         dtype=theano.config.floatX)
        mean_v = T.nnet.sigmoid(T.dot(h, W.T) + bv)
        v = rng.binomial(size=mean_v.shape, n=1, p=mean_v,
                         dtype=theano.config.floatX)
        return mean_v, v

    chain, updates = theano.scan(lambda v: gibbs_step(v)[1], outputs_info=[v],
                                 n_steps=k)
    v_sample = chain[-1]

    mean_v = gibbs_step(v_sample)[0]
    monitor = -T.xlogx.xlogy0(v, mean_v) - T.xlogx.xlogy0(1 - v, 1 - mean_v)
    #monitor = T.nnet.binary_crossentropy(v, mean_v)  # note the sign!
    monitor = monitor.sum() / v.shape[0]

    def free_energy(v):
        return -(v * bv).sum() - T.log(1 + T.exp(T.dot(v, W) + bh)).sum()

    cost = (free_energy(v) - free_energy(v_sample)) / v.shape[0]

    return v_sample, cost, monitor, updates


class RnnRbm(object):
    """Simple class to train an RNN-RBM from MIDI files and to generate sample
    sequences."""
    def __init__(self,
                 recurrent_layer='gru',
                 n_hidden=150,
                 n_hidden_recurrent=100,
                 r=(21, 109),
                 dt=0.3,
                 optimizer='adadelta',
                 recurrent_connected_pct=.15,
                 spectral_radius=1.1,
                 state_from_file=None):
        """

        :param recurrent_layer: 'standard_rnn' or 'gru'
        :param n_hidden:  integer
            Number of hidden units of the conditional RBMs.
        :param n_hidden_recurrent: integer
            Number of hidden units of the RNN.
        :param r: (integer, integer) tuple
            Specifies the pitch range of the piano-roll in MIDI note numbers,
            including r[0] but not r[1], such that r[1]-r[0] is the number of
            visible units of the RBM at a given time step. The default (21,
            109) corresponds to the full range of piano (88 notes).
        :param dt: float
            Sampling period when converting the MIDI files into piano-rolls, or
            equivalently the time difference between consecutive time steps.
        :param optimizer: 'sgd' or 'adadelta'
        :param recurrent_connected_pct: float
            percent of initial nonzero values in the initialization of the
            recurrent hidden weight matrix.
        :param spectral_radius: float
            spectral radius of the recurrent hidden weight matrix.
        :param state_from_file: str
            filename of parameters/ optimzer state, which were saved during
            training

        :return:
        """

        if recurrent_layer is 'gru':
            build_network = self.build_grurbm
        elif recurrent_layer is 'rnn':
            build_network = self.build_rnnrbm

        self.epsilon = 1e-8
        self.dataset = None
        self.costs = None
        self.monitors = None
        self.r = r
        self.dt = dt
        self.timesteps = T.iscalar('Song length')
        self.recurrent_connections = int(n_hidden_recurrent * recurrent_connected_pct)
        self.spectral_radius = spectral_radius
        self.filename = state_from_file
        self.recurrent_layer = recurrent_layer
        self.training_steps_done = 0
        self.steps = None
        self.initial_state = None
        self.initial_visible = None
        self.ext = theano.shared(np.zeros((1, self.n_visible), dtype=theano.config.floatX))
        self.n_visible = r[1] - r[0]
        width = 1
        self.n_hidden = width * self.n_visible

        lr = T.scalar('Learn rate')
        gamma = T.scalar('Adadelta gamma (slowness)')
        beta1 = T.scalar('Adam par1')
        beta2 = T.scalar('Adam par2')

        (v, v_sample, cost, monitor, params,
             updates_train, v_t, updates_generate,
             u_t_trn, hidden_inits) = build_network(r[1] - r[0],
                                                    n_hidden,
                                                    n_hidden_recurrent)
        #except Exception as e:  # for soru in base class
        #    print('{}: not implemented!'.format(e))

        self.parameters = params
        self.optimizer = optimizer

        gradient = T.grad(cost, params, consider_constant=[v_sample])
        #updates_train.update(
        #    ((p, p - lr * g) for p, g in zip(params, gradient))
        #)

        # select optimizer function:
        if optimizer is 'adadelta':
            _optimizer = self.adadelta
            self.gamma = gamma
            optimizer_args = (params, gradient, lr, gamma)
            train_function_ins = [v, lr, gamma]
        elif optimizer is 'adam':
            _optimizer = self.adam
            self.gamma = gamma
            optimizer_args = (params, gradient, lr, beta1, beta2)
            train_function_ins = [v, lr, beta1, beta2]
        else:  # use sgd
            _optimizer = self.sgd
            optimizer_args = (params, gradient, lr)
            train_function_ins = [v, lr]

        updates = _optimizer(*optimizer_args)
        updates_train.update(updates)

        # Compile theano functions:
        self.train_function = theano.function(
            train_function_ins,
            [monitor, cost],
            updates=updates_train,
            on_unused_input='warn',
            mode=mode,
            name='train'
        )
        self.generate_function = theano.function(
            [self.timesteps] + hidden_inits,
            v_t,
            updates=updates_generate,
            mode=mode,
            name='generate'
        )

        self.generate_hidden_states = theano.function(
            [v],
            u_t_trn,
            on_unused_input='warn',
            mode=mode,
            name='generate hidden states'
        )


    def build_rnnrbm(self, n_visible, n_hidden, n_hidden_recurrent):
        """Construct a symbolic RNN-RBM and initialize parameters.

        n_visible : integer
            Number of visible units.
        n_hidden : integer
            Number of hidden units of the conditional RBMs.
        n_hidden_recurrent : integer
            Number of hidden units of the RNN.

        Return a (v, v_sample, cost, monitor, params, updates_train, v_t,
        updates_generate) tuple:

        v : Theano matrix
            Symbolic variable holding an input sequence (used during training)
        v_sample : Theano matrix
            Symbolic variable holding the negative particles for CD log-likelihood
            gradient estimation (used during training)
        cost : Theano scalar
            Expression whose gradient (considering v_sample constant) corresponds
            to the LL gradient of the RNN-RBM (used during training)
        monitor : Theano scalar
            Frame-level pseudo-likelihood (useful for monitoring during training)
        params : tuple of Theano shared variables
            The parameters of the model to be optimized during training.
        updates_train : dictionary of Theano variable -> Theano variable
            Update object that should be passed to theano.function when compiling
            the training function.
        v_t : Theano matrix
            Symbolic variable holding a generated sequence (used during sampling)
        updates_generate : dictionary of Theano variable -> Theano variable
            Update object that should be passed to theano.function when compiling
            the generation function."""

        (W, bv, bh, Wuh, Wuv, Wvu, Wuu, bu) = self.get_inits(n_visible,
                                                             n_hidden,
                                                             n_hidden_recurrent)
        params = [W, bv, bh, Wuh, Wuv, Wvu, Wuu, bu]

        u0 = T.zeros((n_hidden_recurrent,))  # initial value for the RNN hidden

        v = T.matrix()  # a training sequence

        # If `v_t` is given, deterministic recurrence to compute the variable
        # biases bv_t, bh_t at each time step. If `v_t` is None, same recurrence
        # but with a separate Gibbs chain at each time step to sample (generate)
        # from the RNN-RBM. The resulting sample v_t is returned in order to be
        # passed down to the sequence history.
        def recurrence(v_t, u_tm1):
            bv_t = bv + T.dot(u_tm1, Wuv)
            bh_t = bh + T.dot(u_tm1, Wuh)
            generate = v_t is None
            if generate:
                v_t, _, _, updates = build_rbm(T.zeros((n_visible,)), W, bv_t,
                                               bh_t, k=25)
            u_t = T.tanh(bu + T.dot(v_t, Wvu) + T.dot(u_tm1, Wuu))
            return ([v_t, u_t], updates) if generate else [u_t, bv_t, bh_t]

        # For training, the deterministic recurrence is used to compute all the
        # {bv_t, bh_t, 1 <= t <= T} given v. Conditional RBMs can then be trained
        # in batches using those parameters.
        (u_t_trn, bv_t, bh_t), updates_train = theano.scan(lambda v_t, u_tm1, *_: recurrence(v_t, u_tm1),
                                                           sequences=v,
                                                           outputs_info=[u0, None, None],
                                                           non_sequences=params)

        v_sample, cost, monitor, updates_rbm = build_rbm(v, W, bv_t[:], bh_t[:],
                                                         k=15)
        updates_train.update(updates_rbm)

        # symbolic loop for sequence generation
        (v_t, u_t), updates_generate = theano.scan(
            lambda u_tm1, *_: recurrence(None, u_tm1),
            outputs_info=[None, u0], non_sequences=params, n_steps=self.timesteps)

        return (v, v_sample, cost, monitor, params, updates_train, v_t,
                updates_generate, u_t_trn)

    def build_grurbm(self, n_visible, n_hidden, n_hidden_recurrent):
        """Construct a symbolic RNN-RBM and initialize parameters.

        n_visible : integer
            Number of visible units.
        n_hidden : integer
            Number of hidden units of the conditional RBMs.
        n_hidden_recurrent : integer
            Number of hidden units of the RNN.

        Return a (v, v_sample, cost, monitor, params, updates_train, v_t,
        updates_generate) tuple:

        v : Theano matrix
            Symbolic variable holding an input sequence (used during training)
        v_sample : Theano matrix
            Symbolic variable holding the negative particles for CD log-likelihood
            gradient estimation (used during training)
        cost : Theano scalar
            Expression whose gradient (considering v_sample constant) corresponds
            to the LL gradient of the RNN-RBM (used during training)
        monitor : Theano scalar
            Frame-level pseudo-likelihood (useful for monitoring during training)
        params : tuple of Theano shared variables
            The parameters of the model to be optimized during training.
        updates_train : dictionary of Theano variable -> Theano variable
            Update object that should be passed to theano.function when compiling
            the training function.
        v_t : Theano matrix
            Symbolic variable holding a generated sequence (used during sampling)
        updates_generate : dictionary of Theano variable -> Theano variable
            Update object that should be passed to theano.function when compiling
            the generation function."""

        (W, bv, bh, Wuh, Wuv, U, Uz, Ur, W0, Wz, Wr) = self.get_inits(n_visible,
                                                                      n_hidden,
                                                                      n_hidden_recurrent)
        params = [W, bv, bh, Wuh, Wuv, U, Uz, Ur, W0, Wz, Wr]

        u0 = T.zeros((n_hidden_recurrent,))
        z0 = T.zeros((n_hidden_recurrent,))
        r0 = T.zeros((n_hidden_recurrent,))

        self.trn_initial_state = [u0, z0, r0]
        self.initial_visible = T.zeros((n_visible,))

        # random inits for generative mode:
        #u0_gen = theano.shared(np.random.uniform(-1, 1, (n_hidden_recurrent,)).astype(np.float32))
        #z0_gen = theano.shared(np.random.uniform(0, 1, (n_hidden_recurrent,)).astype(np.float32))
        #r0_gen = theano.shared(np.random.uniform(0, 1, (n_hidden_recurrent,)).astype(np.float32))

        u0_gen = T.vector()
        z0_gen = T.vector()
        r0_gen = T.vector()

        hidden_inits = [u0_gen, z0_gen, r0_gen]
        #self.initial_state = hidden_inits
        #self.hidden_inits = hidden_inits

        v = T.matrix()  # a training sequence

        # If `v_t` is given, deterministic recurrence to compute the variable
        # biases bv_t, bh_t at each time step. If `v_t` is None, same recurrence
        # but with a separate Gibbs chain at each time step to sample (generate)
        # from the RNN-RBM. The resulting sample v_t is returned in order to be
        # passed down to the sequence history.

        def recurrence(v_t, u_tm1, z_t, r_t):  # v is input, u is neuron
            bv_t = bv + T.dot(u_tm1, Wuv)
            bh_t = bh + T.dot(u_tm1, Wuh)
            generate = v_t is None
            if generate:
                v_t, _, _, updates = build_rbm(self.initial_visible, W, bv_t,
                                               bh_t, k=25)
            u_t = (1 - z_t) * u_tm1 + z_t * T.tanh(T.dot(r_t * u_tm1, U) + T.dot(v_t, W0))
            z_t = T.nnet.sigmoid(T.dot(u_tm1, Uz) + T.dot(v_t, Wz))
            r_t = T.nnet.sigmoid(T.dot(u_tm1, Ur) + T.dot(v_t, Wr))
            return ([v_t, u_t, z_t, r_t], updates) if generate else [u_t, z_t, r_t, bv_t, bh_t]

        # For training, the deterministic recurrence is used to compute all the
        # {bv_t, bh_t, 1 <= t <= T} given v. Conditional RBMs can then be trained
        # in batches using those parameters.
        (u_t, z_t, r_t, bv_t, bh_t), updates_train = theano.scan(
            lambda v_t, u_tm1, z_t, r_t, *_: recurrence(v_t, u_tm1, z_t, r_t),
            sequences=v, outputs_info=[u0, z0, r0, None, None], non_sequences=params)

        u_t_trn = [u_t, z_t, r_t]

        v_sample, cost, monitor, updates_rbm = build_rbm(v, W, bv_t[:], bh_t[:],
                                                         k=15)
        updates_train.update(updates_rbm)

        # symbolic loop for sequence generation
        (v_t, u_t, z_t, r_t), updates_generate = theano.scan(
            lambda u_tm1, z_t, r_t, *_: recurrence(None, u_tm1, z_t, r_t),
            outputs_info=[None, u0_gen, z0_gen, r0_gen], non_sequences=params, n_steps=self.timesteps)

        return (v, v_sample, cost, monitor, params, updates_train, v_t,
                updates_generate, u_t_trn, hidden_inits)

    def get_inits(self, n_visible, n_hidden, n_hidden_recurrent):

        #RBM parameters:
        W = shared_normal(n_visible, n_hidden, 0.01)
        bv = shared_zeros(n_visible)
        bh = shared_zeros(n_hidden)
        Wuh = shared_normal(n_hidden_recurrent, n_hidden, 0.0001)
        Wuv = shared_normal(n_hidden_recurrent, n_visible, 0.0001)

        rbm_parameters = (W, bv, bh, Wuh, Wuv)

        if self.recurrent_layer is not 'gru':
            Wvu = shared_normal(n_visible, n_hidden_recurrent, 0.0001)
            Wuu = get_random_sparse(n_hidden_recurrent, self.recurrent_connections)
            rho = get_spectral_radius(Wuu)
            Wuu *= self.spectral_radius / rho  # normalized to have specified spectral radius
            bu = shared_zeros(n_hidden_recurrent)

            rnn_parameters = (Wvu, Wuu, bu)

        else:
            U_val = get_random_sparse(n_hidden_recurrent, self.recurrent_connections)
            rho = get_spectral_radius(U_val)
            U_val *= self.spectral_radius / rho  # normalized to have specified spectral radius
            U = theano.shared(U_val)
            #Uz = shared_normal(n_hidden_recurrent, n_hidden_recurrent, 0.0001)
            Uz_val = get_random_sparse(n_hidden_recurrent, self.recurrent_connections)
            rho = get_spectral_radius(Uz_val)
            Uz_val *= self.spectral_radius / rho  # normalized to have specified spectral radius
            Uz = theano.shared(Uz_val)
            #Ur = shared_normal(n_hidden_recurrent, n_hidden_recurrent, 0.0001)
            Ur_val = get_random_sparse(n_hidden_recurrent, self.recurrent_connections)
            rho = get_spectral_radius(Ur_val)
            Ur_val *= self.spectral_radius / rho  # normalized to have specified spectral radius
            Ur = theano.shared(Ur_val)
            W0 = shared_normal(n_visible, n_hidden_recurrent, 0.0001)
            Wz = shared_normal(n_visible, n_hidden_recurrent, 0.0001)
            Wr = shared_normal(n_visible, n_hidden_recurrent, 0.0001)

            rnn_parameters = (U, Uz, Ur, W0, Wz, Wr)

        parameters = rbm_parameters + rnn_parameters

        if self.filename is not None:  # override above with loaded values
            loaded_params = np.load(self.filename + '.npz')
            num_params = len(loaded_params['param_list'])
            for n in range(num_params):
                try:
                    parameters[n].set_value(loaded_params['param_list'][n])
                except Exception as e:
                    print("{}: Something went wrong... parameter values not set!".format(e))
                    pass


        return parameters

    def adadelta(self, params, gradient, lr, gamma):
        """Adadelta optimizer function.

        - gamma = 1 is SGD

        :param params: params list
        :return:
        """
        eps = self.epsilon

        # initialize state:
        gtm1 = [shared_zeros(*param.get_value().shape) for param in params]
        stm1 = [shared_zeros(*param.get_value().shape) for param in params]

        if self.filename is not None:  # override with saved state values
            loaded_params = np.load(self.filename + '.npz')
            num_params = len(loaded_params['param_list'])
            for n in range(num_params):
                gtm1[n].set_value(loaded_params['gtm1_list'][n])
                stm1[n].set_value(loaded_params['stm1_list'][n])

        # for storing the full state of optimizer:
        self.gtm1 = gtm1
        self.stm1 = stm1
        self.gradient = gradient

        gt = [(1 - gamma) * grad_i ** 2 + gamma * gtm1_i
              for grad_i, gtm1_i in zip(gradient, gtm1)]

        dparams = [T.sqrt((stm1_i + eps) / (gt_i + eps)) *
                   theano.gradient.grad_clip(grad_i, -10, 10)
                   for stm1_i, gt_i, grad_i in
                   zip(stm1, gt, gradient)]

        st = [(1 - gamma) * dpar_i ** 2 + gamma * stm1_i
              for dpar_i, stm1_i in
              zip(dparams, stm1)]

        param_updates = [(p, p - lr * dp) for p, dp in zip(params, dparams)]
        gt_updates = zip(gtm1, gt)
        st_updates = zip(stm1, st)

        return param_updates + gt_updates + st_updates

    def adam(self, params, gradient, lr=0.001, beta1=0.9, beta2=0.999):
        """Adam optimizer function.

        - Note that 'st' is now the first order gradient's MA
        while gt is the second order one's

        :param params: params list
        :return:
        """
        eps = self.epsilon

        # initialize state:
        gtm1 = [shared_zeros(*param.get_value().shape) for param in params]
        stm1 = [shared_zeros(*param.get_value().shape) for param in params]

        if self.filename is not None:  # override with saved state values
            loaded_params = np.load(self.filename + '.npz')
            num_params = len(loaded_params['param_list'])
            for n in range(num_params):
                gtm1[n].set_value(loaded_params['gtm1_list'][n])
                stm1[n].set_value(loaded_params['stm1_list'][n])

        # for storing the full state of optimizer:
        self.gtm1 = gtm1
        self.stm1 = stm1
        self.gradient = gradient

        # track timesteps for bias correction:
        steps = theano.shared(np.float32(1.))
        if self.filename is not None:  # fetch steps from saved state
            try:
                loaded_params = np.load(self.filename + '.npz')
                adam_steps = loaded_params['adam_steps']
                steps.set_value(adam_steps)
            except Exception as e:
                print(e)
        self.steps = steps

        steps_new = steps + 1

        # st = m_t in paper
        st = [(beta1 * stm1_i + (1 - beta1) * grad_i) / (1 - beta1 ** steps)
              for stm1_i, grad_i in zip(stm1, gradient)]

        # gt = v_t in paper
        gt = [(beta2 * gtm1_i + (1 - beta2) * grad_i ** 2) / (1 - beta2 ** steps)
              for gtm1_i, grad_i in zip(gtm1, gradient)]

        dparams = [st_i / (T.sqrt(gt_i) + eps)
                   for st_i, gt_i in zip(st, gt)]

        param_updates = [(p, p - lr * dp) for p, dp in zip(params, dparams)]
        gt_updates = zip(gtm1, gt)
        st_updates = zip(stm1, st)
        step_update = [(steps, steps_new)]

        return param_updates + gt_updates + st_updates + step_update

    def sgd(self, params, gradient, lr):

        return [(p, p - lr * g) for p, g in zip(params, gradient)]

    def train(self, dataset, lr=1., gamma=.9, beta1=0.9, beta2=0.999,
              min_batch_size=100, max_batch_size=None, num_epochs=200,
              save_as=None, noise_probability=0., early_stopping=0.):
        """Train the RNN-RBM via stochastic gradient descent (SGD) using MIDI
        files converted to piano-rolls.

        dataset : list of numpy arrays
        batch_size : integer
            Training sequences will be split into subsequences of at most this
            size before applying the SGD updates.
        num_epochs : integer
            Number of epochs (pass over the training set) performed. The user
            can safely interrupt training with Ctrl+C at any time."""

        if self.optimizer is 'adadelta':
            hyperparams = (lr, gamma)
        elif self.optimizer is 'adam':
            hyperparams = (lr, beta1, beta2)
        else:
            hyperparams = (lr, )

        self.dataset = dataset

        # flatten all parameters into an array for FLANN NN computation:
        # TODO: not sure if this is very useful, since the parameter space
        # is very high dimensional... basically parameters can bounce around
        # minimum and NN will not converge to zero in a long time... take e.g.
        # large dim. arrays with random 0, 1's and compute nn after generating
        # a new one each time! But hmm I think it should be steadily decreasing...
        # anyway think about it
        import pyflann
        flann = pyflann.FLANN()
        pyflann.set_distance_type('euclidean')

        param_vec = self.param_vec
        if param_vec is None:
            param_vec = np.array([])
            for param in self.parameters:
                param_vec = np.concatenate((param_vec, param.get_value().flatten()))
            param_vec = param_vec[None, :]

        # get song probabilities:
        song_lenghts = np.array([len(song) for song in dataset])
        dataset_size = song_lenghts.sum()
        song_probs = song_lenghts / float(dataset_size)

        if max_batch_size is None:
            max_batch_size = min_batch_size
            mean_batch_size = min_batch_size
        else:
            mean_batch_size = int((min_batch_size + max_batch_size) / 2)

        num_batches_in_epoch = int(dataset_size / mean_batch_size)

        best_monitor = 100.
        done = False
        try:
            for epoch in xrange(num_epochs):
                if done:
                    break
                start_time = time.time()

                costs = []
                monitors = []

                batch_number = 0
                while batch_number < num_batches_in_epoch:
                    monitor = 100.
                    batch_number += 1
                    # choose random song from the dataset:
                    try:
                        song = np.random.choice(dataset, p=song_probs)
                    except ValueError:
                        song = dataset[0]
                    # choose random batch from song:
                    song_length = song.shape[0]
                    start_idx = np.random.randint(0, song_length - max_batch_size)
                    end_idx = start_idx + np.random.randint(min_batch_size, max_batch_size + 1)
                    batch = song[start_idx:end_idx]
                    if np.sum(batch) < min_notes_in_batch:
                        continue
                    # add noise:
                    if noise_probability > 0.:
                        noise = np.random.binomial(1, noise_probability, batch.shape).astype(np.float32)
                        # noise does 0 -> 1...
                        batch = batch + noise
                        # ... and 1 -> 0:
                        batch[batch > 1] = 0
                    if batch.shape[0] > min_batch_size / 2:  # discard if batch size too small
                        monitor, cost = self.train_function(batch, *hyperparams)
                        costs.append(cost)
                        monitors.append(monitor)
                        print('\rSample {:4}/{} -- Cost={:6.3f} -- Monitor={:6.3f}'.format(batch_number,
                                                                               num_batches_in_epoch,
                                                                               float(cost),
                                                                               float(monitor)), end='')
                    if monitor < early_stopping:
                        print('\nEarly stop.')
                        done = True
                        break

                costs = np.asarray(costs)
                monitors = np.asarray(monitors)
                costs[costs < -100] = 0.  # getting rid of infs
                monitors[monitors > 100] = 0.  # getting rid of infs
                self.costs = costs
                self.monitors = monitors
                avg_cost = np.round(np.mean(costs), 3)
                std_cost = np.round(np.std(costs), 3)
                avg_monitor = np.round(np.mean(monitors), 3)
                std_monitor = np.round(np.std(monitors), 3)
                time_elapsed = time.time() - start_time

                # Nearest neighbors in parameter space:
                param_vec_next = np.array([])
                for param in self.parameters:
                    param_vec_next = np.concatenate((param_vec_next, param.get_value().flatten()))

                #flann_params = flann.build_index(param_vec, target_precision=.9)
                nn_dist = np.sqrt(flann.nn(param_vec, param_vec_next, 1)[1][0])

                # add to previous parameter vectors:
                param_vec = np.vstack((param_vec, param_vec_next))

                print('\rEpoch {:4}/{} | Cost mean={:6.3f}, std={:6.3f} | '
                      'Monitor mean={:6.3f}, std={:6.3f} | '
                      'NN dist={:6.3f} | Time={} s\n'.format(epoch + 1,
                                                               num_epochs,
                                                               avg_cost,
                                                               std_cost,
                                                               avg_monitor,
                                                               std_monitor,
                                                               nn_dist,
                                                               np.round(time_elapsed, 0)), end='')
                sys.stdout.flush()
                if save_as is not None and avg_monitor < best_monitor:
                    #print('Saving results...')
                    best_monitor = avg_monitor
                    # save full state, not just parameters:
                    param_list = []
                    gtm1_list = []
                    stm1_list = []
                    for n in range(len(self.parameters)):
                        param_list.append(self.parameters[n].get_value())
                        gtm1_list.append(self.gtm1[n].get_value())
                        stm1_list.append(self.stm1[n].get_value())

                    np.savez(save_as,
                             param_list=param_list,
                             gtm1_list=gtm1_list,
                             stm1_list=stm1_list)



        except KeyboardInterrupt:
            self.costs = costs
            self.monitors = monitors
            print('\nInterrupted by user.')

    def generate(self,
                 filename=None,
                 show=True,
                 timesteps=200,
                 initial_data=None,
                 ext_magnitude=.1,
                 ext_regularization=.5,
                 ext_overemphasize=1.):
        """Generate a sample sequence, plot the resulting piano-roll and save
        it as a MIDI file.

        - Uses pythonmidi and Hexahedria's noteStateMatrixToMidi

        filename : string, None
            A MIDI file will be created at this location. If filename=None,
            will not create a MIDI file but just returns the piano roll.
        show : boolean
            If True, a piano-roll of the generated sequence will be shown."""

        # if initial_data is not None:
        #     h_states = self.generate_hidden_states(initial_data)  # shape (depth, n_hidden)
        #
        #     initial_states = []
        #     for h_state in h_states:
        #         initial_states.append(h_state[-1])
        #
        #     #for h_state, statevec in zip(h_states, self.initial_state):
        #         #statevec.set_value(h_state[-1])
        #         #print(statevec.get_value())
        # else:
        #     initial_states = []
        #     for state in self.trn_initial_state:
        #         initial_states.append(np.zeros(state.eval().shape, dtype=np.float32))

        # set to generate mode:
        self.training_mode = False

        # get initial hidden state from seed data:
        initial_state = self.generate_hidden_states(initial_data, 1., .9)[:, -1, :]

        # construct external field:
        n_vis = self.n_visible
        x = np.arange(n_vis)
        ext = -ext_magnitude * ((n_vis / 2 + ext_regularization) ** 2 /
                               (x + ext_regularization) /
                               (n_vis - x + ext_regularization) -
                               ext_overemphasize)
        ext = ext[None, :].astype(theano.config.floatX)
        self.ext.set_value(ext)

        piano_roll = self.generate_function(timesteps, initial_state).astype(np.int64)
        if initial_data is not None:
            self.piano_roll = np.concatenate((initial_data, piano_roll), axis=0).astype(np.int64)
        else:
            self.piano_roll = piano_roll

        if filename is not None:

            statematrix = statemat_from_pianoroll(self.piano_roll)
            noteStateMatrixToMidi(statematrix, name=filename)

            if show:
                #extent = (0, self.dt * len(self.generated_data)) + self.r
                plt.figure()
                #plt.imshow(self.generated_data.T, origin='lower', aspect='auto',
                #             interpolation='nearest', cmap=plt.cm.gray_r,
                #             extent=extent)
                plt.pcolor(self.piano_roll.T, cmap='Greens')
                if initial_data is not None:
                    plt.vlines(initial_data.shape[0], 0, 88, colors='red')
                plt.xlabel('timestep')
                plt.ylabel('MIDI note number')
                plt.title('Seed data and generated piano-roll')
        else:
            return self.piano_roll

        # revert to training mode:
        self.training_mode = True

        # reset external field, just in case:
        self.ext.set_value(np.zeros_like(ext, dtype=theano.config.floatX))

    def generate_rnn_activity(self, input):
        """Generate rnn hidden state activity using given input

        :param input:
            sample of input data
        :return:
            sample of hidden state data, same length as input data
        """
        simulated = self.generate_hidden_states(input, 0., 1.)
        return simulated


    def load_parameters(self, params_file):
        loaded_params = np.load(params_file + '.npz')
        n = 0
        while True:
            try:
                self.parameters[n].set_value(loaded_params['param_list'][n])
                n += 1
            except Exception as e:
                print("{}".format(e))
                break


####################################
### Translation invariant models ###
####################################


def build_rbm_ti(v, wvec, wuwa_t, wuwb_t, bv_t, bh_t, k):
    """Construct a k-step Gibbs chain starting at v for a
    translation invariant RBM.

    v : Theano matrix of shape (timesteps, n_visible)
    wvec : Theano *vector* of shape(
        Weight vector of the RBM to be used in the convolution
    wuwa_t: vector of shape (timesteps, n_hidden)
    bv_t : Theano row vector of shape (timesteps, n_visible)
        Visible bias scalar of the RBM as a f() of time
    bh_t : Theano row vector of shape (timesteps, n_hidden)
        Hidden bias scalar of the RBM as a f() of time
    k : scalar or Theano scalar
        Length of the Gibbs chain.

    Return a (v_sample, cost, monitor, updates) tuple:

    v_sample :
    cost : Theano scalar
        Expression whose gradient with respect to W, bv, bh is the CD-k
        approximation to the log-likelihood of `v` (training example) under the
        RBM. The cost is averaged in the batch case.
    monitor: Theano scalar
        Pseudo log-likelihood (also averaged in the batch case).
    updates: dictionary of Theano variable -> Theano variable
        The `updates` object returned by scan."""

    def gibbs_step(v_in):  # input shape (timesteps, n_visible); no scan slicing!
        v_dot_W0 = convolution(wvec, v_in[:, None, :])[:, 0, :]  # get shape (timesteps, n_hidden)
        #TODO: check the dims below!! [0] really needed??
        wuwb_t_dot_v = T.sum(wuwb_t * v_in, axis=1)[0]  # shape (timesteps, ) (??? check!!!)
        v_dot_W = v_dot_W0 + wuwb_t_dot_v * wuwa_t   # broadcasting?
        mean_h = T.nnet.sigmoid(v_dot_W + bh_t)
        h = rng.binomial(size=mean_h.shape, n=1, p=mean_h,
                         dtype=theano.config.floatX)
        wvecT = wvec[:, ::-1] #  reverse w corresponds to the transpose of the matrix W
        h_dot_W0T = convolution(wvecT, h[:, None, :])[:, 0, :]  # get shape (timesteps, n_visible)
        wuwa_t_dot_u = T.sum(wuwa_t * h, axis=1)[0]  # shape (timesteps, )
        h_dot_WT = h_dot_W0T + wuwa_t_dot_u * wuwb_t
        mean_v = T.nnet.sigmoid(h_dot_WT + bv_t)
        v_in = rng.binomial(size=mean_v.shape, n=1, p=mean_v,
                         dtype=theano.config.floatX)  # shape (timesteps, n_visible)
        return mean_v, v_in  # v should be shape (n_visible, )

    chain, updates = theano.scan(lambda v_in: gibbs_step(v_in)[1], outputs_info=[T.unbroadcast(v, 0)],
                                 n_steps=k)
    v_sample = chain[-1]

    mean_v = gibbs_step(v_sample)[0]
    monitor = -T.xlogx.xlogy0(v, mean_v) - T.xlogx.xlogy0(1 - v, 1 - mean_v)  # note sign
    monitor = monitor.sum() / v.shape[0]

    #TODO: make sure this works as well as original!!
    def free_energy(v_in):  # input shape (timesteps, n_visible)
        v_dot_W0 = convolution(wvec, v_in[:, None, :])[:, 0, :]  # get shape (timesteps, n_hidden)
        wuwb_t_dot_v = T.sum(wuwb_t * v_in, axis=1)[0]  # shape (timesteps, )
        v_dot_W = v_dot_W0 + wuwb_t_dot_v * wuwa_t
        return -(v_in * bv_t).sum() - T.log(1 + T.exp(v_dot_W + bh_t)).sum()
    # v_t predicts v_sample_tp1:
    cost = (free_energy(v) - free_energy(v_sample)) / v.shape[0]

    # def free_energy(v_in):  # input shape (timesteps, n_visible)
    #     v_dot_W0 = convolution(wvec, v_in[:, None, :])[:, 0, :]  # get shape (timesteps, n_hidden)
    #     wuwb_t_dot_v = T.sum(wuwb_t * v_in, axis=1)[0]  # shape (timesteps, )
    #     v_dot_W = v_dot_W0 + wuwb_t_dot_v * wuwa_t
    #     return -(v_in * bv_t).sum() - T.log(1 + T.exp(v_dot_W + bh_t)).sum()
    # # v_t predicts v_sample_tp1:
    # cost = (free_energy(v) - free_energy(v_sample)) / v.shape[0]

    return v_sample, cost, monitor, updates


class RnnRbmTI(RnnRbm):
    """Translation invariant version of the RnnRbm."""


    def build_grurbm(self, n_visible, n_hidden, n_hidden_recurrent):
        """Construct a symbolic RNN-RBM and initialize parameters.

        **This is now a translation invariant model**

        wvec: vector -> RBM weight matrix
        bv: visible bias
        bh: hidden bias
        wuh: from recurrent hidden to bias matrix
        wuv: from recurrent hidden to visible matrix
        wuwa: from recurrent hidden to RBM weight, hidden
        wuwb: from recurrent hidden to RBM weight, visible (or vice versa??)
        uvec: recurrent u-u interaction
        uz: recurrent u-z interaction
        ur: recurrent u-r interaction
        w0: recurrent input-u interaction
        wz: recurrent input-z interaction
        wr: recurrent input-r interaction

        """

        (wvec, bv, bh, wuh, wuv, wuwa, wuwb, uvec, uz, ur, w0, wz, wr) = self.get_inits(n_visible,
                                                                            n_hidden,
                                                                            n_hidden_recurrent)
        params = [wvec, bv, bh, wuh, wuv, wuwa, wuwb, uvec, uz, ur, w0, wz, wr]

        u0 = T.zeros((n_hidden_recurrent,))
        z0 = T.zeros((n_hidden_recurrent,))
        r0 = T.zeros((n_hidden_recurrent,))

        self.trn_initial_state = [u0, z0, r0]
        self.initial_visible = T.zeros((1, n_visible))

        # # random inits for generative mode:
        # u0_gen = theano.shared(np.random.uniform(-1, 1, (n_hidden_recurrent,)).astype(np.float32))
        # z0_gen = theano.shared(np.random.uniform(0, 1, (n_hidden_recurrent,)).astype(np.float32))
        # r0_gen = theano.shared(np.random.uniform(0, 1, (n_hidden_recurrent,)).astype(np.float32))
        #
        # self.initial_state = [u0_gen, z0_gen, r0_gen]

        u0_gen = T.vector()
        z0_gen = T.vector()
        r0_gen = T.vector()

        hidden_inits = [u0_gen, z0_gen, r0_gen]

        v = T.matrix()  # a training sequence
        # do the reshaping inside convolutions

        # If `v_t` is given, deterministic recurrence to compute the variable
        # biases bv_t, bh_t at each time step. If `v_t` is None, same recurrence
        # but with a separate Gibbs chain at each time step to sample (generate)
        # from the RNN-RBM. The resulting sample v_t is returned in order to be
        # passed down to the sequence history.

        def recurrence(v_t, u_tm1, z_t, r_t):  # v is input, u is hidden unit
            # v_t shape is (n_visible, ) because scan takes a slice

            # update RBM parameters:
            utm1_dot_wuv = convolution(wuv, u_tm1[None, :])[0]
            bv_t = bv + utm1_dot_wuv
            utm1_dot_wuh = convolution(wuh, u_tm1[None, :])[0]
            bh_t = bh + utm1_dot_wuh
            #utm1_dot_WuW = convolution(wuw, u_tm1[None, :])[0]
            #wvec_t = wvec[0] + utm1_dot_WuW
            wuwa_t = convolution(wuwa, u_tm1[None, :])[0]  # now vector shape (n_hidden, )
            wuwb_t = convolution(wuwb, u_tm1[None, :])[0]  # now vector shape (n_visible, )

            generate = v_t is None
            if generate:
                v_t, _, _, updates = build_rbm_ti(self.initial_visible,
                                                  wvec,
                                                  wuwa_t[None, :],
                                                  wuwb_t[None, :],
                                                  bv_t,
                                                  bh_t,
                                                  k=25)
                v_t = v_t[0]

            # all dot products = convolutions that are needed below:
            r_t_u_tm1_dot_U = convolution(uvec, u_tm1[None, :] * r_t[None, :])[0]
            v_t_dot_W0 = convolution(w0, v_t[None, None, :])[0, 0, :]
            u_tm1_dot_Uz = convolution(uz, u_tm1[None, :])[0]
            v_t_dot_Wz = convolution(wz, v_t[None, :])[0]
            u_tm1_dot_Ur = convolution(ur, u_tm1[None, :])[0]
            v_t_dot_Wr = convolution(wr, v_t[None, :])[0]

            #u_t = (1 - z_t) * u_tm1 + z_t * T.tanh(T.dot(r_t * u_tm1, U) + T.dot(v_t, W0))
            u_t = (1 - z_t) * u_tm1 + z_t * T.tanh(r_t_u_tm1_dot_U + v_t_dot_W0)
            #z_t = T.nnet.sigmoid(T.dot(u_tm1, Uz) + T.dot(v_t, Wz))
            z_t = T.nnet.sigmoid(u_tm1_dot_Uz + v_t_dot_Wz)
            #r_t = T.nnet.sigmoid(T.dot(u_tm1, Ur) + T.dot(v_t, Wr))
            r_t = T.nnet.sigmoid(u_tm1_dot_Ur + v_t_dot_Wr)
            return ([v_t, u_t, z_t, r_t], updates) if generate else [u_t, z_t, r_t, bv_t,
                                                                     bh_t, wuwa_t, wuwb_t]

        # For training, the deterministic recurrence is used to compute all the
        # {bv_t, bh_t, 1 <= t <= T} given v. Conditional RBMs can then be trained
        # in batches using those parameters.
        (u_t, z_t, r_t, bv_t, bh_t, wuwa_t, wuwb_t), updates_train = theano.scan(
            lambda v_t, u_tm1, z_t, r_t, *_: recurrence(v_t, u_tm1, z_t, r_t),
                                                        sequences=v,
                                                        outputs_info=[u0, z0, r0, None, None, None, None],
                                                        non_sequences=params
                                                                )
        u_t_trn = [u_t, z_t, r_t]

        # TODO: see if increasing k results in better simulations!
        v_sample, cost, monitor, updates_rbm = build_rbm_ti(v, wvec, wuwa_t, wuwb_t, bv_t, bh_t,
                                                         k=15)
        updates_train.update(updates_rbm)

        # symbolic loop for sequence generation
        (v_t, u_t, z_t, r_t), updates_generate = theano.scan(
            lambda u_tm1, z_t, r_t, *_: recurrence(None, u_tm1, z_t, r_t),
            outputs_info=[None, u0_gen, z0_gen, r0_gen], non_sequences=params, n_steps=self.timesteps)

        return (v, v_sample, cost, monitor, params, updates_train, v_t,
                updates_generate, u_t_trn, hidden_inits)


    def get_inits(self, n_visible, n_hidden, n_hidden_recurrent):

        #RBM parameters:
        wvec = shared_normal(1, n_visible + n_hidden - 1, 0.01)
        bv = shared_zeros()
        bh = shared_zeros()
        wuh = shared_normal(1, n_hidden_recurrent + n_hidden - 1, 0.0001)  # hidden_rec to hidden
        wuv = shared_normal(1, n_hidden_recurrent + n_visible - 1, 0.0001)  # hidden_rec to visible
        wuwa = shared_normal(1, n_hidden_recurrent + n_hidden - 1, 0.0001)  # hidden_rec to hidden side of W
        wuwb = shared_normal(1, n_hidden_recurrent + n_visible - 1, 0.0001)  # hidden_rec to visible side of W

        rbm_parameters = (wvec, bv, bh, wuh, wuv, wuwa, wuwb)

        uvec = shared_normal(1, 2 * n_hidden_recurrent - 1, .0001)
        uz = shared_normal(1, 2 * n_hidden_recurrent - 1, .0001)
        ur = shared_normal(1, 2 * n_hidden_recurrent - 1, .0001)
        w0 = shared_normal(1, n_visible + n_hidden_recurrent - 1, .0001)
        wz = shared_normal(1, n_visible + n_hidden_recurrent - 1, .0001)
        wr = shared_normal(1, n_visible + n_hidden_recurrent - 1, .0001)

        rnn_parameters = (uvec, uz, ur, w0, wz, wr)

        parameters = rbm_parameters + rnn_parameters

        if self.filename is not None:  # override above with loaded values
            loaded_params = np.load(self.filename + '.npz')
            num_params = len(loaded_params['param_list'])
            for n in range(num_params):
                try:
                    parameters[n].set_value(loaded_params['param_list'][n])
                except Exception as e:
                    print("{}: Parameter values not set!".format(e))
                    pass

        return parameters






def build_rbm_tip_old(v, Wmat, wuwa_t, wuwb_t, bv_t, bh_t, k):
    """Construct a k-step Gibbs chain starting at v for a
    translation invariant RBM.

    v : Theano matrix of shape (timesteps, n_visible)
    Wmat : Theano matrix of the RBM
    wuwa_t: matrix of shape (timesteps, n_hidden)
    wuwb_t: matrix of shape (timesteps, n_visible)
    bv_t : Theano row vector of shape (timesteps, n_visible)
        Visible bias scalar of the RBM as a f() of time
    bh_t : Theano row vector of shape (timesteps, n_hidden)
        Hidden bias scalar of the RBM as a f() of time
    k : scalar or Theano scalar
        Length of the Gibbs chain.

    Return a (v_sample, cost, monitor, updates) tuple:

    v_sample :
    cost : Theano scalar
        Expression whose gradient with respect to W, bv, bh is the CD-k
        approximation to the log-likelihood of `v` (training example) under the
        RBM. The cost is averaged in the batch case.
    monitor: Theano scalar
        Pseudo log-likelihood (also averaged in the batch case).
    updates: dictionary of Theano variable -> Theano variable
        The `updates` object returned by scan."""

    def gibbs_step(v_in):  # input shape (timesteps, n_visible); no scan slicing!
        v_dot_W0 = T.dot(v_in, Wmat)  # get shape (timesteps, n_hidden)
        wuwb_t_dot_v = T.sum(wuwb_t * v_in, axis=1)[:, None] # shape (timesteps, 1)
        v_dot_W = v_dot_W0 + wuwb_t_dot_v * wuwa_t   # broadcasting??
        mean_h = T.nnet.sigmoid(v_dot_W + bh_t)
        h = rng.binomial(size=mean_h.shape, n=1, p=mean_h,
                         dtype=theano.config.floatX)  # shape (timesteps, n_hidden)
        #wvecT = wvec[:, ::-1] #  reverse w corresponds to the transpose of the matrix W
        #h_dot_W0T = convolution(wvecT, h[:, None, :])[:, 0, :]  # get shape (timesteps, n_visible)
        h_dot_W0T = T.dot(h, Wmat.T)
        wuwa_t_dot_u = T.sum(wuwa_t * h, axis=1)[:, None]  # shape (timesteps, )
        h_dot_WT = h_dot_W0T + wuwa_t_dot_u * wuwb_t
        mean_v = T.nnet.sigmoid(h_dot_WT + bv_t)
        v_in = rng.binomial(size=mean_v.shape, n=1, p=mean_v,
                         dtype=theano.config.floatX)  # shape (timesteps, n_visible)
        return mean_v, v_in  # v should be shape (n_visible, )

    chain, updates = theano.scan(lambda v_in: gibbs_step(v_in)[1], outputs_info=[T.unbroadcast(v, 0)],
                                 n_steps=k)
    v_sample = chain[-1]

    mean_v = gibbs_step(v_sample)[0]
    monitor = -T.xlogx.xlogy0(v, mean_v) - T.xlogx.xlogy0(1 - v, 1 - mean_v)  # note sign
    monitor = monitor.sum() / v.shape[0]

    def free_energy(v_in):  # input shape (timesteps, n_visible)
        v_dot_W0 = T.dot(v_in, Wmat)  # get shape (timesteps, n_hidden)
        wuwb_t_dot_v = T.sum(wuwb_t * v_in, axis=1)[:, None]
        v_dot_W = v_dot_W0 + wuwb_t_dot_v * wuwa_t
        return -(v_in * bv_t).sum() - T.log(1 + T.exp(v_dot_W + bh_t)).sum()
    # v_t predicts v_sample_tp1:
    cost = (free_energy(v) - free_energy(v_sample)) / v.shape[0]

    # def free_energy(v_in):  # input shape (timesteps, n_visible)
    #     v_dot_W0 = convolution(wvec, v_in[:, None, :])[:, 0, :]  # get shape (timesteps, n_hidden)
    #     wuwb_t_dot_v = T.sum(wuwb_t * v_in, axis=1)[0]  # shape (timesteps, )
    #     v_dot_W = v_dot_W0 + wuwb_t_dot_v * wuwa_t
    #     return -(v_in * bv_t).sum() - T.log(1 + T.exp(v_dot_W + bh_t)).sum()
    # # v_t predicts v_sample_tp1:
    # cost = (free_energy(v) - free_energy(v_sample)) / v.shape[0]

    return v_sample, cost, monitor, updates

def tip_from_vec(vector, k, l):
    """Constructs a symbolic translation invariant
    and periodic (TIP) matrix from a vector input.

    k: multiple of in dimension
    l: multiple of out dimension
    """
    vec_len = vector.shape[0]
    tip_mat, _ = theano.scan(lambda vec: T.roll(vec, 1),
                         outputs_info=[vector],
                         n_steps=vec_len)
    if k > 1 or l > 1:
        tip_mat = T.tile(tip_mat, (k, l), ndim=2)

    return tip_mat

class RnnRbmTIP_old(RnnRbm):

    """Translation invariant *and* periodic version of the RnnRbm.
    This one has dynamic RBM weight mat... may be the cause of
    instability and learning problems??
    """



    def __init__(self, *args, **kwargs):
        """
        kwargs:
        width -- int; multiplier of input-to-rnn dimension.
            Will determine the total number of independent parameters
            as (5 * k + 6) * n_visible (for GRU).
        """
        self.level = kwargs.pop('width')

        # set hidden dimensions and add to kwargs:
        n_visible = kwargs.pop('n_visible')
        kwargs['n_hidden'] = self.level * n_visible
        kwargs['n_hidden_recurrent'] = self.level * n_visible

        # call other parent class inits:
        super(RnnRbmTIP, self).__init__(*args, **kwargs)

    def build_grurbm(self, n_visible, n_hidden, n_hidden_recurrent):
        """Construct a symbolic RNN-RBM and initialize parameters.

        **This is now a translation invariant and periodic model**

        """

        (wvec, bv, bh, wuh, wuv, wuwa, wuwb, uvec, uz, ur, w0, wz, wr) = self.get_inits(n_visible,
                                                                            n_hidden,
                                                                            n_hidden_recurrent)
        params = [wvec, bv, bh, wuh, wuv, wuwa, wuwb, uvec, uz, ur, w0, wz, wr]

        k = self.level
        # Create the TIP matrices from parameters:
        # RBM mats:
        Wmat = tip_from_vec(wvec, 1, k)  # RBM weight mat, non-dynamic term
        Wuh = tip_from_vec(wuh, 1, 1)  # for hidden bias
        Wuv = tip_from_vec(wuv, k, 1)  # for visible bias; note no .T!!
        Wuwa = tip_from_vec(wuwa, 1, 1)  # for RBM weight mat, dynamic term
        Wuwb = tip_from_vec(wuwb, k, 1)  # for RBM weight mat, dynamic term
        # GRU mats:
        Umat = tip_from_vec(uvec, 1, 1)
        Uz = tip_from_vec(uz, 1, 1)
        Ur = tip_from_vec(ur, 1, 1)
        W0 = tip_from_vec(w0, 1, k)
        Wz = tip_from_vec(wz, 1, k)
        Wr = tip_from_vec(wr, 1, k)

        # Training mode inits:
        u0 = T.zeros((k * n_visible,))
        z0 = T.zeros((k * n_visible,))
        r0 = T.zeros((k * n_visible,))

        self.trn_initial_state = [u0, z0, r0]

        # Generating mode init variables:
        u0_gen = T.vector()
        z0_gen = T.vector()
        r0_gen = T.vector()

        hidden_inits = [u0_gen, z0_gen, r0_gen]

        v = T.matrix()  # a training sequence
        # do the reshaping inside convolutions


        def recurrence(v_t, u_tm1, z_t, r_t):  # v is input, u is hidden unit
            # v_t shape is (n_visible, ) because scan takes a slice
            # update RBM parameters:
            bv_t = bv + T.dot(u_tm1, Wuv)
            bh_t = bh + T.dot(u_tm1, Wuh)
            #wuwa_t = convolution(wuwa, u_tm1[None, :])[0]  # now vector shape (n_hidden, )
            wuwa_t = T.dot(u_tm1, Wuwa)
            #wuwb_t = convolution(wuwb, u_tm1[None, :])[0]  # now vector shape (n_visible, )
            wuwb_t = T.dot(u_tm1, Wuwb)
            generate = v_t is None
            if generate:
                #TODO: do build_rbm_tip
                v_t, _, _, updates = build_rbm_tip(T.zeros((1, n_visible)),
                                                  Wmat,
                                                  wuwa_t[None, :],
                                                  wuwb_t[None, :],
                                                  bv_t,
                                                  bh_t,
                                                  k=25)
                # TODO: see if increasing k results in better simulations!
                v_t = v_t[0]

            # all dot products:
            r_t_u_tm1_dot_U = T.dot(u_tm1 * r_t, Umat)
            v_t_dot_W0 = T.dot(v_t, W0)
            u_tm1_dot_Uz = T.dot(u_tm1, Uz)
            v_t_dot_Wz = T.dot(v_t, Wz)
            u_tm1_dot_Ur = T.dot(u_tm1, Ur)
            v_t_dot_Wr = T.dot(v_t, Wr)

            u_t = (1 - z_t) * u_tm1 + z_t * T.tanh(r_t_u_tm1_dot_U + v_t_dot_W0)
            z_t = T.nnet.sigmoid(u_tm1_dot_Uz + v_t_dot_Wz)
            r_t = T.nnet.sigmoid(u_tm1_dot_Ur + v_t_dot_Wr)
            return ([v_t, u_t, z_t, r_t], updates) if generate else [u_t, z_t, r_t, bv_t,
                                                                     bh_t, wuwa_t, wuwb_t]

        non_sequences = [Wmat, bv, bh, Wuh, Wuv, Wuwa, Wuwb, Umat, Uz, Ur, W0, Wz, Wr]

        # For training, the deterministic recurrence is used to compute all the
        # {bv_t, bh_t, 1 <= t <= T} given v. Conditional RBMs can then be trained
        # in batches using those parameters.
        (u_t, z_t, r_t, bv_t, bh_t, wuwa_t, wuwb_t), updates_train = theano.scan(
            lambda v_t, u_tm1, z_t, r_t, *_: recurrence(v_t, u_tm1, z_t, r_t),
                                                        sequences=v,
                                                        outputs_info=[u0, z0, r0, None, None, None, None],
                                                        non_sequences=non_sequences
                                                                                )
        u_t_trn = [u_t, z_t, r_t]

        v_sample, cost, monitor, updates_rbm = build_rbm_tip(v, Wmat, wuwa_t, wuwb_t,
                                                            bv_t, bh_t, k=15)
        updates_train.update(updates_rbm)

        # symbolic loop for sequence generation
        (v_t, u_t, z_t, r_t), updates_generate = theano.scan(
            lambda u_tm1, z_t, r_t, *_: recurrence(None, u_tm1, z_t, r_t),
            outputs_info=[None, u0_gen, z0_gen, r0_gen], non_sequences=params, n_steps=self.timesteps)

        return (v, v_sample, cost, monitor, params, updates_train, v_t,
                updates_generate, u_t_trn, hidden_inits)

    def get_inits(self, n_visible, n_hidden, n_hidden_recurrent):

        k = self.level
        #RBM parameters:
        # to be stacked into TIP matrices
        wvec = shared_normal(0, n_visible, 0.01, 'wvec')  # stack n_visible times; RBM global weight
        bv = shared_zeros()
        bh = shared_zeros()
        wuh = shared_normal(0, k * n_visible, 0.0001, 'wuh')  # hidden_rec to hidden; stack n_hidden_recurrent times
        wuv = shared_normal(0, n_visible, 0.0001, 'wuv')  # hidden_rec to visible
        # stack these n_hidden_recurrent times:
        wuwa = shared_normal(0, k * n_visible, 0.0001, 'wuwa')  # hidden_rec to hidden side of W
        wuwb = shared_normal(0, n_visible, 0.0001, 'wuwb')  # hidden_rec to visible side of W

        rbm_parameters = (wvec, bv, bh, wuh, wuv, wuwa, wuwb)

        #GRU parameters:
        uvec = shared_normal(0, k * n_visible, .0001, 'uvec')  # square
        uz = shared_normal(0, k * n_visible, .0001, 'uz')  # square
        ur = shared_normal(0, k * n_visible, .0001, 'ur')  # square
        w0 = shared_normal(0, n_visible, .0001, 'w0')
        wz = shared_normal(0, n_visible, .0001, 'wz')
        wr = shared_normal(0, n_visible, .0001, 'wr')

        rnn_parameters = (uvec, uz, ur, w0, wz, wr)

        parameters = rbm_parameters + rnn_parameters

        if self.filename is not None:  # override above with loaded values
            loaded_params = np.load(self.filename + '.npz')
            num_params = len(loaded_params['param_list'])
            for n in range(num_params):
                try:
                    parameters[n].set_value(loaded_params['param_list'][n])
                except Exception as e:
                    print("{}: Parameter values not set!".format(e))
                    pass

        return parameters






def build_rbm_tip_old2(v, Wmat, bv_t, bh_t, k):
    """Construct a k-step Gibbs chain starting at v for a
    translation invariant RBM.

    v : Theano matrix of shape (timesteps, n_visible)
    Wmat : Theano matrix of the RBM
    bv_t : Theano row vector of shape (timesteps, n_visible)
        Visible bias scalar of the RBM as a f() of time
    bh_t : Theano row vector of shape (timesteps, n_hidden)
        Hidden bias scalar of the RBM as a f() of time
    k : scalar or Theano scalar
        Length of the Gibbs chain.

    Return a (v_sample, cost, monitor, updates) tuple:

    v_sample :
    cost : Theano scalar
        Expression whose gradient with respect to W, bv, bh is the CD-k
        approximation to the log-likelihood of `v` (training example) under the
        RBM. The cost is averaged in the batch case.
    monitor: Theano scalar
        Pseudo log-likelihood (also averaged in the batch case).
    updates: dictionary of Theano variable -> Theano variable
        The `updates` object returned by scan."""

    def gibbs_step(v_in):  # input shape (timesteps, n_visible); no scan slicing!
        v_dot_W0 = T.dot(v_in, Wmat)  # get shape (timesteps, n_hidden)
        mean_h = T.nnet.sigmoid(v_dot_W0 + bh_t)
        h = rng.binomial(size=mean_h.shape, n=1, p=mean_h,
                         dtype=theano.config.floatX)  # shape (timesteps, n_hidden)
        h_dot_W0T = T.dot(h, Wmat.T)
        mean_v = T.nnet.sigmoid(h_dot_W0T + bv_t)
        v_in = rng.binomial(size=mean_v.shape, n=1, p=mean_v,
                         dtype=theano.config.floatX)  # shape (timesteps, n_visible)
        return mean_v, v_in  # v should be shape (n_visible, )

    chain, updates = theano.scan(lambda v_in: gibbs_step(v_in)[1], outputs_info=[T.unbroadcast(v, 0)],
                                 n_steps=k)
    v_sample = chain[-1]

    mean_v = gibbs_step(v_sample)[0]
    monitor = -T.xlogx.xlogy0(v, mean_v) - T.xlogx.xlogy0(1 - v, 1 - mean_v)  # note sign
    monitor = monitor.sum() / v.shape[0]

    def free_energy(v_in):  # input shape (timesteps, n_visible)
        v_dot_W0 = T.dot(v_in, Wmat)  # get shape (timesteps, n_hidden)
        return -(v_in * bv_t).sum() - T.log(1 + T.exp(v_dot_W0 + bh_t)).sum()
    # v_t predicts v_sample_tp1:
    cost = (free_energy(v) - free_energy(v_sample)) / v.shape[0]

    # def free_energy(v_in):  # input shape (timesteps, n_visible)
    #     v_dot_W0 = convolution(wvec, v_in[:, None, :])[:, 0, :]  # get shape (timesteps, n_hidden)
    #     wuwb_t_dot_v = T.sum(wuwb_t * v_in, axis=1)[0]  # shape (timesteps, )
    #     v_dot_W = v_dot_W0 + wuwb_t_dot_v * wuwa_t
    #     return -(v_in * bv_t).sum() - T.log(1 + T.exp(v_dot_W + bh_t)).sum()
    # # v_t predicts v_sample_tp1:
    # cost = (free_energy(v) - free_energy(v_sample)) / v.shape[0]

    return v_sample, cost, monitor, updates

class RnnRbmTIP_old2(RnnRbm):

    """Translation invariant *and* periodic version of the RnnRbm.
    Dynamic weight mat dropped, so now only dynamic biases...
    """

    def __init__(self, *args, **kwargs):
        """
        kwargs:
        width -- int; multiplier of input-to-rnn dimension.
            Will determine the total number of independent parameters
            as (5 * k + 6) * n_visible (for GRU).
        """
        self.level = kwargs.pop('width')

        # set hidden dimensions and add to kwargs:
        n_visible = kwargs.pop('n_visible')
        kwargs['n_hidden'] = self.level * n_visible
        kwargs['n_hidden_recurrent'] = self.level * n_visible

        # call other parent class inits:
        super(RnnRbmTIP, self).__init__(*args, **kwargs)

    def build_grurbm(self, n_visible, n_hidden, n_hidden_recurrent):
        """Construct a symbolic RNN-RBM and initialize parameters.

        **This is now a translation invariant and periodic model**

        """

        (wvec, bv, bh, wuh, wuv, uvec, uz, ur, w0, wz, wr) = self.get_inits(n_visible,
                                                                            n_hidden,
                                                                            n_hidden_recurrent)
        params = [wvec, bv, bh, wuh, wuv, uvec, uz, ur, w0, wz, wr]

        k = self.level
        # Create the TIP matrices from parameters:
        # RBM mats:
        Wmat = tip_from_vec(wvec, 1, k)  # RBM weight mat
        Wuh = tip_from_vec(wuh, 1, 1)  # for hidden bias
        Wuv = tip_from_vec(wuv, k, 1)  # for visible bias; note no .T!!
        # GRU mats:
        Umat = tip_from_vec(uvec, 1, 1)
        Uz = tip_from_vec(uz, 1, 1)
        Ur = tip_from_vec(ur, 1, 1)
        W0 = tip_from_vec(w0, 1, k)
        Wz = tip_from_vec(wz, 1, k)
        Wr = tip_from_vec(wr, 1, k)

        # Training mode inits:
        u0 = T.zeros((k * n_visible,))
        z0 = T.zeros((k * n_visible,))
        r0 = T.zeros((k * n_visible,))

        self.trn_initial_state = [u0, z0, r0]

        # Generating mode init variables:
        u0_gen = T.vector()
        z0_gen = T.vector()
        r0_gen = T.vector()

        hidden_inits = [u0_gen, z0_gen, r0_gen]

        v = T.matrix()  # a training sequence
        # do the reshaping inside convolutions


        def recurrence(v_t, u_tm1, z_t, r_t):  # v is input, u is hidden unit
            # v_t shape is (n_visible, ) because scan takes a slice
            # update RBM parameters:
            bv_t = bv + T.dot(u_tm1, Wuv)
            bh_t = bh + T.dot(u_tm1, Wuh)
            generate = v_t is None
            if generate:
                #TODO: do build_rbm_tip
                v_t, _, _, updates = build_rbm_tip(T.zeros((1, n_visible)),
                                                  Wmat,
                                                  bv_t,
                                                  bh_t,
                                                  k=25)
                # TODO: see if increasing k results in better simulations!
                v_t = v_t[0]

            # all dot products:
            r_t_u_tm1_dot_U = T.dot(u_tm1 * r_t, Umat)
            v_t_dot_W0 = T.dot(v_t, W0)
            u_tm1_dot_Uz = T.dot(u_tm1, Uz)
            v_t_dot_Wz = T.dot(v_t, Wz)
            u_tm1_dot_Ur = T.dot(u_tm1, Ur)
            v_t_dot_Wr = T.dot(v_t, Wr)

            u_t = (1 - z_t) * u_tm1 + z_t * T.tanh(r_t_u_tm1_dot_U + v_t_dot_W0)
            z_t = T.nnet.sigmoid(u_tm1_dot_Uz + v_t_dot_Wz)
            r_t = T.nnet.sigmoid(u_tm1_dot_Ur + v_t_dot_Wr)
            return ([v_t, u_t, z_t, r_t], updates) if generate else [u_t, z_t, r_t, bv_t, bh_t]

        non_sequences = [Wmat, bv, bh, Wuh, Wuv, Umat, Uz, Ur, W0, Wz, Wr]

        # For training, the deterministic recurrence is used to compute all the
        # {bv_t, bh_t, 1 <= t <= T} given v. Conditional RBMs can then be trained
        # in batches using those parameters.
        (u_t, z_t, r_t, bv_t, bh_t), updates_train = theano.scan(
            lambda v_t, u_tm1, z_t, r_t, *_: recurrence(v_t, u_tm1, z_t, r_t),
                                                        sequences=v,
                                                        outputs_info=[u0, z0, r0, None, None],
                                                        non_sequences=non_sequences
                                                                                )
        u_t_trn = [u_t, z_t, r_t]

        v_sample, cost, monitor, updates_rbm = build_rbm_tip(v, Wmat, bv_t, bh_t, k=15)
        updates_train.update(updates_rbm)

        # symbolic loop for sequence generation
        (v_t, u_t, z_t, r_t), updates_generate = theano.scan(
            lambda u_tm1, z_t, r_t, *_: recurrence(None, u_tm1, z_t, r_t),
            outputs_info=[None, u0_gen, z0_gen, r0_gen], non_sequences=params, n_steps=self.timesteps)

        return (v, v_sample, cost, monitor, params, updates_train, v_t,
                updates_generate, u_t_trn, hidden_inits)

    def get_inits(self, n_visible, n_hidden, n_hidden_recurrent):

        k = self.level
        #RBM parameters:
        # to be stacked into TIP matrices
        wvec = shared_normal(0, n_visible, 0.01, 'wvec')  # stack n_visible times; RBM global weight
        bv = shared_zeros()
        bh = shared_zeros()
        wuh = shared_normal(0, k * n_visible, 0.0001, 'wuh')  # hidden_rec to hidden; stack n_hidden_recurrent times
        wuv = shared_normal(0, n_visible, 0.0001, 'wuv')  # hidden_rec to visible

        rbm_parameters = (wvec, bv, bh, wuh, wuv)

        #GRU parameters:
        uvec = shared_normal(0, k * n_visible, .0001, 'uvec')  # square
        uz = shared_normal(0, k * n_visible, .0001, 'uz')  # square
        ur = shared_normal(0, k * n_visible, .0001, 'ur')  # square
        w0 = shared_normal(0, n_visible, .0001, 'w0')
        wz = shared_normal(0, n_visible, .0001, 'wz')
        wr = shared_normal(0, n_visible, .0001, 'wr')

        rnn_parameters = (uvec, uz, ur, w0, wz, wr)

        parameters = rbm_parameters + rnn_parameters

        if self.filename is not None:  # override above with loaded values
            loaded_params = np.load(self.filename + '.npz')
            num_params = len(loaded_params['param_list'])
            for n in range(num_params):
                try:
                    parameters[n].set_value(loaded_params['param_list'][n])
                except Exception as e:
                    print("{}: Parameter values not set!".format(e))
                    pass

        return parameters


def build_rbm_ti(v, wvec, wuwa_t, wuwb_t, bv_t, bh_t, k):


    def gibbs_step(v_in):  # input shape (timesteps, n_visible); no scan slicing!
        v_dot_W0 = convolution(wvec, v_in[:, None, :])[:, 0, :]  # get shape (timesteps, n_hidden)
        #TODO: check the dims below!! [0] really needed??
        wuwb_t_dot_v = T.sum(wuwb_t * v_in, axis=1)[0]  # shape (timesteps, ) (??? check!!!)
        v_dot_W = v_dot_W0 + wuwb_t_dot_v * wuwa_t   # broadcasting?
        mean_h = T.nnet.sigmoid(v_dot_W + bh_t)
        h = rng.binomial(size=mean_h.shape, n=1, p=mean_h,
                         dtype=theano.config.floatX)
        wvecT = wvec[:, ::-1] #  reverse w corresponds to the transpose of the matrix W
        h_dot_W0T = convolution(wvecT, h[:, None, :])[:, 0, :]  # get shape (timesteps, n_visible)
        wuwa_t_dot_u = T.sum(wuwa_t * h, axis=1)[0]  # shape (timesteps, )
        h_dot_WT = h_dot_W0T + wuwa_t_dot_u * wuwb_t
        mean_v = T.nnet.sigmoid(h_dot_WT + bv_t)
        v_in = rng.binomial(size=mean_v.shape, n=1, p=mean_v,
                         dtype=theano.config.floatX)  # shape (timesteps, n_visible)
        return mean_v, v_in  # v should be shape (n_visible, )

    chain, updates = theano.scan(lambda v_in: gibbs_step(v_in)[1], outputs_info=[T.unbroadcast(v, 0)],
                                 n_steps=k)
    v_sample = chain[-1]

    mean_v = gibbs_step(v_sample)[0]
    monitor = -T.xlogx.xlogy0(v, mean_v) - T.xlogx.xlogy0(1 - v, 1 - mean_v)  # note sign
    monitor = monitor.sum() / v.shape[0]

    #TODO: make sure this works as well as original!!
    def free_energy(v_in):  # input shape (timesteps, n_visible)
        v_dot_W0 = convolution(wvec, v_in[:, None, :])[:, 0, :]  # get shape (timesteps, n_hidden)
        wuwb_t_dot_v = T.sum(wuwb_t * v_in, axis=1)[0]  # shape (timesteps, )
        v_dot_W = v_dot_W0 + wuwb_t_dot_v * wuwa_t
        return -(v_in * bv_t).sum() - T.log(1 + T.exp(v_dot_W + bh_t)).sum()
    # v_t predicts v_sample_tp1:
    cost = (free_energy(v) - free_energy(v_sample)) / v.shape[0]

    # def free_energy(v_in):  # input shape (timesteps, n_visible)
    #     v_dot_W0 = convolution(wvec, v_in[:, None, :])[:, 0, :]  # get shape (timesteps, n_hidden)
    #     wuwb_t_dot_v = T.sum(wuwb_t * v_in, axis=1)[0]  # shape (timesteps, )
    #     v_dot_W = v_dot_W0 + wuwb_t_dot_v * wuwa_t
    #     return -(v_in * bv_t).sum() - T.log(1 + T.exp(v_dot_W + bh_t)).sum()
    # # v_t predicts v_sample_tp1:
    # cost = (free_energy(v) - free_energy(v_sample)) / v.shape[0]

    return v_sample, cost, monitor, updates



def build_rbm_tip(v, wvec_c, bv_t, bh_t, k):
    """Construct a k-step Gibbs chain starting at v for a
    translation invariant RBM.

    v : Theano matrix of shape (timesteps, n_visible)
    wvec_c : Theano RBM weight vector, TIP processed!!
    bv_t : Theano row vector of shape (timesteps, n_visible)
        Visible bias scalar of the RBM as a f() of time
    bh_t : Theano row vector of shape (timesteps, n_hidden)
        Hidden bias scalar of the RBM as a f() of time
    k : scalar or Theano scalar
        Length of the Gibbs chain.

    Return a (v_sample, cost, monitor, updates) tuple:

    v_sample :
    cost : Theano scalar
        Expression whose gradient with respect to W, bv, bh is the CD-k
        approximation to the log-likelihood of `v` (training example) under the
        RBM. The cost is averaged in the batch case.
    monitor: Theano scalar
        Pseudo log-likelihood (also averaged in the batch case).
    updates: dictionary of Theano variable -> Theano variable
        The `updates` object returned by scan."""

    wvec_cT = wvec_c[:, ::-1]
    def gibbs_step(v_in, bv, bh):  # input shape (timesteps, n_visible); no scan slicing!
        v_dot_W0 = convolution(wvec_c, v_in[:, None, :])[:, 0, :]  # get shape (timesteps, n_hidden)
        mean_h = T.nnet.sigmoid(v_dot_W0 + bh)
        h = rng.binomial(size=mean_h.shape, n=1, p=mean_h,
                         dtype=theano.config.floatX)  # shape (timesteps, n_hidden)
        h_dot_W0T = convolution(wvec_cT, h[:, None, :])[:, 0, :]
        mean_v = T.nnet.sigmoid(h_dot_W0T + bv)
        v_in = rng.binomial(size=mean_v.shape, n=1, p=mean_v,
                            dtype=theano.config.floatX)  # shape (timesteps, n_visible)
        return mean_v, v_in  # shape (timesteps, n_visible)

    chain, updates_rbm = theano.scan(lambda v_in, bv, bh: gibbs_step(v_in, bv, bh)[1],
                                     outputs_info=[T.unbroadcast(v, 0)],
                                     non_sequences=[bv_t, bh_t],
                                     n_steps=k,
                                     name='RBM sampling scan')
    v_sample = chain[-1]  # shape (timesteps, n_visible)

    mean_v = gibbs_step(v_sample, bv_t, bh_t)[0][:-1]  # shape (timesteps - 1, n_visible)
    monitor = -T.xlogx.xlogy0(v[1:], mean_v) - T.xlogx.xlogy0(1 - v[1:], 1 - mean_v)  # note sign
    monitor = monitor.sum() / v.shape[0]

    def free_energy(v_in):  # input shape (timesteps, n_visible)
        v_dot_W0 = convolution(wvec_c, v_in[:, None, :])[:, 0, :]
        return -(v_in * bv_t[:-1]).sum() - T.log(1 + T.exp(v_dot_W0 + bh_t[:-1])).sum()
    # v_t predicts v_sample_tp1:
    cost = (free_energy(v[1:]) - free_energy(v_sample[:-1])) / v.shape[0]

    return v_sample, cost, monitor, updates_rbm


class RnnRbmTIP(RnnRbm):

    """Translation invariant *and* periodic version of the RnnRbm.
        Dynamic weight mat dropped, so now only dynamic biases...
    """

    def __init__(self,
                 depth=1,
                 width=1,
                 r=(21, 109),
                 optimizer='adadelta',
                 state_from_file=None,
                 connected_weights=.15,
                 recurrent_spectral_radius=1.1,
                 rbm_spectral_radius=.5,
                 input_spectral_radius=.1,
                 dropout=0.,
                 input_dropout=0.,
                 input_noise=0.):
        """
        :param depth: uint; number of rnn layers
        :param width: uint; hidden and hidden recurrent sizes are
            size width * n_visible
        :param r: (integer, integer) tuple
            Specifies the pitch range of the piano-roll in MIDI note numbers,
            including r[0] but not r[1], such that r[1]-r[0] is the number of
            visible units of the RBM at a given time step. The default (21,
            109) corresponds to the full range of piano (88 notes).
        :param optimizer: 'sgd' or 'adadelta'
        :param connected_weights: float
            percent of initial nonzero values in the initialization of the
            recurrent hidden weight matrix.
        :param rnn_spectral_radius: float
            spectral radius of the recurrent hidden weight matrix.
        :param state_from_file: str
            filename of parameters/ optimzer state, which were saved during
            training
        :param dropout: 0 < float < 1
            dropout=.2 means 20% chance of neuron dropped out of graph
        :param input_dropout: dropout just for input to first layer
        :param input_noise: 0 < float < 1
            Instead of just dropping notes, may also add "error" notes
            Should not be used together with dropout!

        :return:
        """

        build_network = self.build_grurbm

        self.epsilon = 1e-8  # epsilon parameter for Adadelta
        self.dataset = None
        self.costs = None
        self.monitors = None
        self.r = r
        self.depth = depth
        self.width = width
        self.filename = state_from_file
        self.training_steps_done = 0
        self.initial_state = None
        self.initial_visible = None
        self.connected_weights = connected_weights
        self.recurrent_spectral_radius = recurrent_spectral_radius
        self.input_spectral_radius = input_spectral_radius
        self.rbm_spectral_radius = rbm_spectral_radius
        self.dropout = dropout
        self.input_dropout = input_dropout
        self.input_noise = input_noise

        # train or generate mode?
        self.training_mode = True

        lr = T.scalar('Learn rate')
        lr.tag.test_value = 1.
        gamma = T.scalar('Adadelta gamma (slowness)')
        gamma.tag.test_value = .9
        self.timesteps = T.iscalar('Song length')
        self.timesteps.tag.test_value = 19
        self.n_visible = r[1] - r[0]
        self.n_hidden = width * self.n_visible
        # Initialize external field to zeros:
        self.ext = theano.shared(np.zeros((1, self.n_visible), dtype=theano.config.floatX))

        self.param_vec = None


        (v, v_sample, cost, monitor, params,
             updates_train, v_gen, updates_generate,
             u_t_trn, u_dynamics, u0_gen) = build_network(self.n_visible,
                                                    self.n_hidden)

        self.parameters = params
        self.optimizer = optimizer

        gradient = T.grad(cost, params, consider_constant=[v_sample])
        #updates_train.update(
        #    ((p, p - lr * g) for p, g in zip(params, gradient))
        #)

        # select optimizer function:
        if optimizer is 'adadelta':
            _optimizer = self.adadelta
            self.gamma = gamma
            optimizer_args = (params, gradient, lr, gamma)
            train_function_ins = [v, lr, gamma]
        else:  # use sgd
            _optimizer = self.sgd
            optimizer_args = (params, gradient, lr)
            train_function_ins = [v, lr]

        updates = _optimizer(*optimizer_args)
        updates_train.update(updates)

        # Compile theano functions:
        self.train_function = theano.function(
            train_function_ins,
            [monitor, cost],
            updates=updates_train,
            on_unused_input='warn',
            mode=mode,
            name='train'
        )
        self.generate_function = theano.function(
            [self.timesteps, u0_gen],
            v_gen,
            updates=updates_generate,
            on_unused_input='warn',
            mode=mode,
            name='generate'
        )

        self.generate_hidden_states = theano.function(
            train_function_ins,
            u_dynamics,
            on_unused_input='warn',
            updates=updates_train,
            mode=mode,
            name='generate hidden states'
        )

    def build_grurbm(self, n_visible, n_hidden):
        """Construct a symbolic RNN-RBM and initialize parameters.

        **This is now a translation invariant and periodic model**
        **Equal number of hidden and hidden recurrent units!!**

        """
        #TODO: add bias terms?

        (mvec, bv, bh, wuh, wuv, uvec, uz, ur, w0_1, wz_1, wr_1,
         w0_rest, wr_rest, wz_rest) = self.get_inits(n_visible, n_hidden)
        params = [mvec, bv, bh, wuh, wuv, uvec, uz, ur,
                  w0_1, wz_1, wr_1, w0_rest, wr_rest, wz_rest]

        # Create the TIP vectors corresponding to the TIP matrices:
        # RBM mats:
        mvec_rep = rep_vec(mvec, n_visible, n_hidden)  # RBM weight mat
        wuh_rep = rep_vec(wuh, n_hidden, n_hidden)  # for hidden bias
        wuv_rep = rep_vec(wuv, n_visible, n_hidden)  # for visible bias
        # GRU mats:
        uvec_rep = rep_vec(uvec, n_hidden, n_hidden)
        uz_rep = rep_vec(uz, n_hidden, n_hidden)
        ur_rep = rep_vec(ur, n_hidden, n_hidden)
        w0_1_rep = rep_vec(w0_1, n_visible, n_hidden)
        wz_1_rep = rep_vec(wz_1, n_visible, n_hidden)
        wr_1_rep = rep_vec(wr_1, n_visible, n_hidden)
        w0_rest_rep = rep_vec(w0_rest, n_hidden, n_hidden)
        wz_rest_rep = rep_vec(wz_rest, n_hidden, n_hidden)
        wr_rest_rep = rep_vec(wr_rest, n_hidden, n_hidden)

        v = T.matrix()  # a training sequence
        v.tag.test_value = np.random.randn(5, n_visible).astype(np.float32)

        ### GRU evolution: ###
        def evolve(uvecs, w0s, uzs, wzs, urs, wrs, u_l_tm1, u_lm1_t):
            """Given u_l_tm1 and u_lm1_t, returns u_l_t.

            :return: u_l_t; shape=(n_hidden, )
            """
            # apply dropout:
            if self.dropout > 0.:
                if self.training_mode:
                    u_lm1_t *= rng.binomial(size=u_lm1_t.shape,
                                            n=1,
                                            p=1 - self.dropout,
                                            dtype=theano.config.floatX)
                else:
                    u_lm1_t *= 1 - self.dropout

            # dot products:
            #TODO: I could parallelize this, see e.g. here:
            # http://deeplearning.net/tutorial/lstm.html
            u_lm1_t_dot_W0 = convolution(w0s[None, :], u_lm1_t[None, :])[0]
            u_l_tm1_dot_Uz = convolution(uzs[None, :], u_l_tm1[None, :])[0]
            u_lm1_t_dot_Wz = convolution(wzs[None, :], u_lm1_t[None, :])[0]
            u_l_tm1_dot_Ur = convolution(urs[None, :], u_l_tm1[None, :])[0]
            u_lm1_t_dot_Wr = convolution(wrs[None, :], u_lm1_t[None, :])[0]

            # update and reset gate evolution:
            z_t = T.nnet.sigmoid(u_l_tm1_dot_Uz + u_lm1_t_dot_Wz)
            r_t = T.nnet.sigmoid(u_l_tm1_dot_Ur + u_lm1_t_dot_Wr)

            # hidden state dot prod:
            r_t_u_tm1_dot_U = convolution(uvecs[None, :], u_l_tm1[None, :] * r_t[None, :])[0]

            # hidden state evolution:
            u_l_t = (1 - z_t) * u_l_tm1 + z_t * T.tanh(r_t_u_tm1_dot_U + u_lm1_t_dot_W0)

            return u_l_t

        #####################
        ### TRAINING MODE ###
        #####################
        # - outer loop over layers, inner loop over time
        u0_trn = T.zeros((n_hidden, ))  # training initial state is always zero
        self.trn_initial_state = [u0_trn]

        def evolve_trn(u_lm1_t, u_l_tm1, uvecs, w0s, uzs, wzs, urs, wrs):
            return evolve(uvecs, w0s, uzs, wzs, urs, wrs, u_l_tm1, u_lm1_t)

        if self.input_noise > 0. and self.training_mode:  # overrides input dropout!
            noise = rng.binomial(size=v.shape,
                                 n=1,
                                 p=self.input_noise,
                                 dtype=theano.config.floatX)
            v1 = T.neq(v, noise).astype(theano.config.floatX)  # zero if equal, one otherwise

        elif self.input_dropout > 0.:
            if self.training_mode:
                v1 = v * rng.binomial(size=v.shape,
                                      n=1,
                                      p=1 - self.input_dropout,
                                      dtype=theano.config.floatX)
            else:
                v1 = (1 - self.input_dropout) * v
        else:
            v1 = v


        # uppropagate from input to first layer, because input is different shape:
        u_1, updates_trn = theano.scan(evolve_trn,
                        sequences=[v1],
                        outputs_info=[u0_trn],
                        non_sequences=[uvec_rep[0], w0_1_rep[0],
                                       uz_rep[0], wz_1_rep[0],
                                       ur_rep[0], wr_1_rep[0]],
                        name='first uppropagation scan')

        def uppropagate_trn(uvecs, w0s, uzs, wzs, urs, wrs, u_lm1):
            """Propagates hidden state of shape (timesteps, n_hidden)
            upwards to higher layer.
            - u_s's shape (timesteps, n_hidden)
            - scan this from l=2 to depth
            """

            u_l, updates_trn_inner = theano.scan(evolve_trn,
                        sequences=[u_lm1],
                        outputs_info=[u0_trn],
                        non_sequences=[uvecs, w0s, uzs, wzs, urs, wrs],
                        name='recurrence train scan')

            return u_l, updates_trn_inner  # shape (timesteps, n_hidden)

        # uppropagate rest:
        u_2_to_L, updates_trn_2 = theano.scan(uppropagate_trn,
                    sequences=[uvec_rep[1:], w0_rest_rep, uz_rep[1:], wz_rest_rep, ur_rep[1:], wr_rest_rep],
                    outputs_info=[u_1],
                    name='rest of uppropagation scan')

        updates_trn.update(updates_trn_2)

        # get last layer for RBM biases:
        u_L = u_2_to_L[-1]

        # all layers together:
        u_dynamics = T.concatenate([u_1[None, :, :], u_2_to_L], axis=0)  # (depth, timesteps, n_hidden)

        # compute dynamic biases (batch):
        bv_t = bv + convolution(wuv_rep, u_L[:, None, :])[:, 0, :]
        bh_t = bh + convolution(wuh_rep, u_L[:, None, :])[:, 0, :]

        # draw samples from RBM:
        # note sampled v is v_tp1, not v_t!! I.e. v_tp1 is t=1...T-1 when input is t=0...T-2
        # Note that v is fed into the function, not v1!!
        v_sample, cost, monitor, updates_trn3 = build_rbm_tip(v, mvec_rep, bv_t, bh_t, k=25)

        updates_trn.update(updates_trn3)



        #######################
        ### GENERATIVE MODE ###
        #######################
        # - outer loop over time, inner loop over layers
        # Generating mode init variables:
        u0_gen = T.matrix()
        u0_gen.tag.test_value = np.random.randn(self.depth, n_hidden).astype(np.float32)

        def evolve_gen(uvecs, w0s, uzs, wzs, urs, wrs, u_l_tm1, u_lm1_t):
            return evolve(uvecs, w0s, uzs, wzs, urs, wrs, u_l_tm1, u_lm1_t)

        def recurrence_gen(u_tm1):  # u_tm1 ALL layers, shape=(depth, n_hidden)

            # Note the external field:
            bv_t = convolution(wuv_rep, u_tm1[-1][None, :]) + self.ext
            bh_t = convolution(wuh_rep, u_tm1[-1][None, :])

            v_t, _, _, updates_gen = build_rbm_tip(T.zeros((1, n_visible)),
                                                   mvec_rep,
                                                   bv_t,
                                                   bh_t,
                                                   k=25)
            v_t = v_t[0]

            # uppropagate v_t -> u_L_t:
            # first layer:
            u_1_t = evolve_gen(uvec_rep[0], w0_1_rep[0], uz_rep[0],
                               wz_1_rep[0], ur_rep[0], wr_1_rep[0], u_tm1[0], v_t)
            # rest of layers:
            u_2_to_L_t, updates_gen2 = theano.scan(evolve_gen,
                                sequences=[uvec_rep[1:], w0_rest_rep, uz_rep[1:],
                                           wz_rest_rep, ur_rep[1:], wr_rest_rep, u_tm1[1:]],
                                outputs_info=[u_1_t],
                                name='uppropagate gen scan')

            updates_gen.update(updates_gen2)

            u_t = T.vertical_stack(u_1_t[None, :], u_2_to_L_t)

            return [u_t, v_t], updates_gen

        (u_gen, v_gen), updates_gen = theano.scan(recurrence_gen,
                    outputs_info=[u0_gen, None],
                    n_steps=self.timesteps,
                    name='recurrence gen scan')

        return (v, v_sample, cost, monitor, params, updates_trn, v_gen,
                updates_gen, u_L, u_dynamics, u0_gen)


    def get_inits(self, n_visible, n_hidden):
        connected = self.connected_weights
        rnn_spectral_radius = self.recurrent_spectral_radius
        rbm_spectral_radius = self.rbm_spectral_radius
        input_spectral_radius = self.input_spectral_radius
        depth = self.depth

        #RBM parameters:
        # RBM depth=1
        mvec = tip_init(1, n_visible, connected, rbm_spectral_radius, name='mvec')  # RBM global weight
        bv = shared_zeros()
        bh = shared_zeros()
        wuh = tip_init(1, n_hidden, connected, rbm_spectral_radius, name='wuh')
        wuv = tip_init(1, n_visible, connected, rbm_spectral_radius, name='wuv')

        rbm_parameters = (mvec, bv, bh, wuh, wuv)

        #GRU parameters; initialized for 'depth' number of layers:
        uvec = tip_init(depth, n_hidden, connected, rnn_spectral_radius, name='uvec')
        uz = tip_init(depth, n_hidden, connected, rnn_spectral_radius, name='uz')
        ur = tip_init(depth, n_hidden, connected, rnn_spectral_radius, name='ur')
        # input to hidden layer (depth=1 gives shape (1, N):
        w0_1 = tip_init(1, n_visible, connected, input_spectral_radius, name='w0')
        wz_1 = tip_init(1, n_visible, connected, input_spectral_radius, name='wz')
        wr_1 = tip_init(1, n_visible, connected, input_spectral_radius, name='wr')
        # higher layer inits:
        w0_rest = tip_init(depth - 1, n_hidden, connected, input_spectral_radius, name='w0')
        wz_rest = tip_init(depth - 1, n_hidden, connected, input_spectral_radius, name='wz')
        wr_rest = tip_init(depth - 1, n_hidden, connected, input_spectral_radius, name='wr')

        rnn_parameters = (uvec, uz, ur, w0_1, wz_1, wr_1, w0_rest, wr_rest, wz_rest)

        parameters = rbm_parameters + rnn_parameters

        if self.filename is not None:  # override above with loaded values
            loaded_params = np.load(self.filename + '.npz')
            num_params = len(loaded_params['param_list'])
            for n in range(num_params):
                try:
                    parameters[n].set_value(loaded_params['param_list'][n])
                except Exception as e:
                    print("{}: Parameter values not set!".format(e))
                    pass

        return parameters














class RNNModel_mod1(object):

    def __init__(self, nh, ni, out_activation=None):
        '''
        - Uses full simulation in cost function
        (not very good...)
        nh :: dimension of the hidden layer
        ni :: dimension of input data
        time  :: initialization/ burn-in period

        -Use this for e.g. music (input vector = notes and has multiple 1's) or
        text (input is a char vector and has one 1).
        - Hidden activation is tanh
        - Output layer activation is sigmoid or something else
        - Output dim = input dim
        - Input data is boolean valued vectors
        - Input data is trained by putting in x_t with t = 0, ..., T s.t.
        t = 0, ..., T - 1 is taken as example sequence and t = T value is
        the label

        '''

        if out_activation is 'tanh':
            self.out_activation = T.tanh
        else:
            self.out_activation = T.nnet.sigmoid

        # parameters of the model
        #TODO: replace this with ESN initialization
        self.init_scale = 1.
        self.ni = ni
        self.nh = nh
        # Initialize parameters:
        self.params, self.names, self.caches = self.initialize_params()

        # Input as matrix:
        x = T.matrix('Input')  # x.shape = (# timesteps, ni)
        # -- time is always the first dimension!
        # -- x_t.shape = (ni,), h_t.shape = (nh,)

        ####################################
        ### Recurrence in training mode: ###
        ####################################
        x_trn_init = T.reshape(x[0], (1, ni))
        timesteps = x.shape[0]

        [h_trn, x_trn], _ = theano.scan(fn=self.recurrence,
                                        outputs_info=[self.h0, T.unbroadcast(x_trn_init, 0)],
                                        n_steps=timesteps)

        # cost, gradients and learning rate
        lr = T.scalar('Learn rate')
        decay_rate = .99
        #burn_in = T.iscalar('Burn-in')  # removed for now
        burn_in = 0
        cost = ((x[burn_in + 1:] - x_trn[burn_in:-1, 0, :]) ** 2).mean()
        gradients = T.grad(cost, self.params)

        # Gradient updates:
        updates = OrderedDict((p, p - lr * g / T.sqrt(ch + 1e-8))
                              for p, g, ch in zip(self.params, gradients, self.caches))
        # Keep hidden state for sequential learning:
        updates.update({self.h0:h_trn[-1, :, :]})
        # update caches:
        cache_updates = OrderedDict((ch, decay_rate * ch + (1 - decay_rate) * g ** 2)
                                   for ch, g in zip(self.caches, gradients))
        updates.update(cache_updates)

        # theano functions
        self.cost_value = theano.function(inputs=[x],
                                          outputs=cost)
        self.train_on_batch = theano.function(inputs=[x, lr],
                                              outputs=cost,
                                              updates=updates)

        ######################################
        ### Recurrence in generative mode: ###
        ######################################
        timesteps = T.iscalar('Timesteps')
        # Initial data:
        x0 = T.matrix('Initial x')
        #x0.tag.test_value = np.random.rand(9, 1).astype(np.float32)
        # pass through initial data:
        [h_gen, _], _ = theano.scan(fn=self.recurrence_mod,
                                   sequences=x0,
                                   outputs_info=[self.h0, None])
        # assign initial h and x:
        h_init = h_gen[-2, :, :]  # shape (times, 1, nh) (?)
        x_init = T.reshape(x0[-1], (1, ni))  # problem here for ni > 1??
        #x_init = x0[-1]
        #x_init = self.b_out # this works, so the reshape is shit...

        [h_gen, x_gen], _ = theano.scan(fn=self.recurrence,
                                        outputs_info=[h_init, T.unbroadcast(x_init, 0)],
                                        n_steps=timesteps)
        # about the T.unbroadcast: https://github.com/Theano/Theano/issues/2985

        # theano functions
        self.simulate = theano.function(inputs=[x0, timesteps],
                                        outputs=[h_gen, x_gen],
                                        updates=[(self.h0, h_gen[-1, :, :])])

    def initialize_params(self):
        ni = self.ni
        nh = self.nh
        init_scale = self.init_scale
        self.W_hid  = theano.shared(init_scale * np.random.uniform(-1.0, 1.0,\
                   (nh, nh)).astype(theano.config.floatX))
        self.W_in   = theano.shared(init_scale * np.random.uniform(-1.0, 1.0,\
                   (nh, ni)).astype(theano.config.floatX))
        self.W_out   = theano.shared(init_scale * np.random.uniform(-1.0, 1.0,\
                   (ni, nh)).astype(theano.config.floatX))
        self.b_hid  = theano.shared(np.zeros((1, nh), dtype=theano.config.floatX))
        self.b_out   = theano.shared(np.zeros((1, ni), dtype=theano.config.floatX))
        self.h0  = theano.shared(np.zeros((1, nh), dtype=theano.config.floatX))  # init h

        # group parameters into list:
        params = [self.W_hid, self.W_in, self.W_out, self.b_hid, self.b_out]
        names = ['Hidden weights', 'Input weights',
                       'Output weights', 'Hidden bias', 'Output bias']

        # initial caches for RMSProp:
        self.W_hid_cache  = theano.shared(np.ones((nh, nh)).astype(theano.config.floatX))
        self.W_in_cache   = theano.shared(np.ones((nh, ni)).astype(theano.config.floatX))
        self.W_out_cache   = theano.shared(np.ones((ni, nh)).astype(theano.config.floatX))
        self.b_hid_cache  = theano.shared(np.ones((1, nh), dtype=theano.config.floatX))
        self.b_out_cache   = theano.shared(np.ones((1, ni), dtype=theano.config.floatX))
        caches = [self.W_hid_cache, self.W_in_cache, self.W_out_cache, self.b_hid_cache, self.b_out_cache]

        return params, names, caches

    def recurrence_mod(self, x_t, h_t):  # need inputs in diff order for x input sequence :/
            h_tp1 = T.tanh(T.dot(h_t, self.W_hid.T) + T.dot(x_t, self.W_in.T) + self.b_hid)
            xhat_tp1 = self.out_activation(T.dot(h_tp1, self.W_out.T) + self.b_out)
            return [h_tp1, xhat_tp1]

    def recurrence(self, h_t, x_t):
            h_tp1 = T.tanh(T.dot(h_t, self.W_hid.T) + T.dot(x_t, self.W_in.T) + self.b_hid)
            xhat_tp1 = self.out_activation(T.dot(h_tp1, self.W_out.T) + self.b_out)
            return [h_tp1, xhat_tp1]

    def save(self, folder):
        for param, name in zip(self.params, self.names):
            np.save(os.path.join(folder, name + '.npy'), param.get_value())

