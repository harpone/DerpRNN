from __future__ import division, print_function

__version__ = "v 0.9"

import sys
import time

import matplotlib.pyplot as plt
from collections import OrderedDict
from random import shuffle

import theano
from theano.tensor.fourier import fft
from theano.compile.nanguardmode import NanGuardMode
from theano import tensor as T
from theano.tensor.signal.conv import conv2d as conv_signal
#from theano.tensor.shared_randomstreams import RandomStreams
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from theano.compile.debugmode import DebugMode

try:
    from theano.misc.pkl_utils import dump
except ImportError:
    dump = theano.misc.pkl_utils.Pickler.dump
import cPickle

# Local imports:
from utils.utils import *
from layers import *

#Don't use a python long as this don't work on 32 bits computers.
np.random.seed(0xbeef)
rng = RandomStreams(seed=np.random.randint(1 << 30))
theano.config.warn.subtensor_merge_bug = False
theano.config.exception_verbosity = 'low'
theano.config.compute_test_value = 'off'

# Debug mode:
#mode = 'DebugMode'
DebugMode.check_py = False  # checking for nans
mode = None
#mode ='FAST_COMPILE'
theano.config.warn.signal_conv2d_interface = False

#convolution = conv_nnet  # won't work...
convolution = conv_signal

# global parameters:
min_steps_in_batch = 3  # require this many nonzero notes in batch

# Activation aliases:
#TODO: try tanh based on hard_sigmoid!!
#TODO: double relu as tanh replacement?
tanh = T.tanh
sigmoid = T.nnet.hard_sigmoid
#relu = T.nnet.relu

class DeepRNN(object):

    """
    - not just music/ piano data
    - can have either SRU or GRU units (only)
    - output layer can be sigmoid, RBM, softmax, ... so out layer is a separate
      'build_output_layer' method, which needs to work in both train and
      generate modes
    - no TI for the moment; TI version can be child class
    - implement 'pretrain' method
    - readout layer should always return v_sample, cost, monitor, updates (not
      just in the case of RBM layer)

    """

    def __init__(self,
                 depth=1,
                 width=1,
                 readout_width=1,
                 readin_layer='tanh',
                 rnn_layer='sru',
                 readout_layer='sigmoid',
                 n_visible=88,
                 out_layers='all',
                 optimizer='adadelta',
                 state_from_file=None,
                 sparsity=.15,
                 readin_input_scale=2.,
                 recurrence_spectral_radius=1.2,
                 recurrence_input_scale=2.,
                 readout_scale=2.,
                 dropout=0.,
                 input_dropout=0.,
                 input_noise=0.,
                 research_mode=False):
        """
        :param depth: uint; number of rnn layers
        :param width: uint; hidden and hidden recurrent sizes are
            size width * n_visible
        :param readout_width: uint; multiple of HIDDEN number of units
        :param optimizer: 'sgd' or 'adadelta'
        :param sparsity: float
            percent of initial nonzero values in the initialization of the
            weight matrices.
        :param rnn_spectral_radius: float
            spectral radius of the recurrent hidden weight matrix.
        :param state_from_file: str
            filename of parameters/ optimzer state, which were saved during
            training
        :param dropout: 0 < float < 1
            dropout=.2 means 20% chance of neuron dropped out of graph
        :param input_dropout: dropout just for input to first layer
        :param input_noise: 0 < float < 1
            Instead of just dropping ones, may also insert "error" ones
            Should not be used together with dropout!
            Fucking slow!!
        :param f: antisymmetric activation. Can be 'tanh' or 'drelu'
        :param g: one sided activation. Can be 'sigmoid' or 'relu'

        kwargs:
        :kwarg

        :return:
        """

        if mode is not None:
            print('Running in {} mode.'.format(mode))
            sys.stdout.flush()
        # Determine recurrent and readout layers:
        if readin_layer is 'tanh':
            self.readin_layer = tanh_readin
        if rnn_layer is 'sru':
            self.rnn_layer = sru_layer  # SRU = Standard Recurrence Unit
        elif rnn_layer is 'gru':
            self.rnn_layer = gru_layer  # GRU = Gated Recurrence Unit
        elif rnn_layer is 'kpz':
            self.rnn_layer = kpz_layer  # KPZ-like layer; !!!EXPERIMENTAL!!!
        if readout_layer is 'sigmoid':
            self.readout_layer = sigmoid_readout
        elif readout_layer is 'rbm':
            self.readout_layer = rbm_readout
        elif readout_layer is 'softmax':
            self.readout_layer = softmax_readout

        if out_layers is 'last':
            self.n_layers = 1
        elif out_layers is 'all':
            self.n_layers = depth

        self.epsilon = 1e-7  # epsilon parameter for Adadelta
        self.dataset = None
        self.costs = None
        self.monitors = None
        self.depth = depth
        self.width = width
        self.readout_width = readout_width
        self.filename = state_from_file
        self.training_steps_done = 0
        self.initial_state = None
        self.initial_visible = None
        self.sparsity = sparsity
        self.recurrence_spectral_radius = recurrence_spectral_radius
        self.recurrence_input_scale = recurrence_input_scale
        self.readin_input_scale = readin_input_scale
        self.readout_scale = readout_scale
        self.dropout = dropout
        self.input_dropout = input_dropout
        self.input_noise = input_noise
        self.param_vec = None

        # train or generate mode?
        self.training_mode = True

        lr = T.scalar('Learn rate', dtype=theano.config.floatX)
        lr.tag.test_value = 1.
        gamma = T.scalar('Adadelta gamma (slowness)')
        gamma.tag.test_value = .9
        self.timesteps = T.iscalar('Song length')
        self.timesteps.tag.test_value = 19
        self.n_visible = n_visible
        self.n_hidden = width * self.n_visible
        self.n_readout_hidden = readout_width * self.n_hidden

        # Initialize external field to zeros: #TODO
        self.ext = theano.shared(np.zeros((self.n_visible, ), dtype=theano.config.floatX))

        ######################
        ### Build network: ###
        ######################
        (v, v_sample, cost, monitor, params,
         updates_train, v_gen, updates_generate,
         h_1_to_L, u0_gen, h0_readin) = self.build_rnn_network

        self.parameters = params
        self.optimizer = optimizer

        # TODO: how does "consider_constant" affect the *other* radouts??
        gradient = T.grad(cost, params, consider_constant=[v_sample])

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
            #mode=NanGuardMode(nan_is_error=True, inf_is_error=False),
            mode=mode,
            name='train function'
        )
        self.generate_function = theano.function(
            [self.timesteps, u0_gen],
            v_gen,
            updates=updates_generate,
            on_unused_input='warn',
            #mode=NanGuardMode(nan_is_error=True, inf_is_error=False),
            mode=mode,
            name='generate function'
        )

        self.generate_hidden_states = theano.function(
            train_function_ins,
            [h0_readin, h_1_to_L],
            on_unused_input='warn',
            updates=updates_train,
            mode=mode,
            name='generate hidden states function'
        )

    @property
    def build_rnn_network(self):
        """

        :return:
        """

        # - initialize rnn layer params & define rnn_step function
        # - initialize readout layer params & define readout function
        # - combine params
        # - evolve will call rnn_step, gru_step or whatevz with the rnn params
        # - input to first hidden uppropagation needs to have W and U matrices
        #   of different sizes as non_sequences! Code ALL parameters separately
        #   for first and rest of layers! Nonsequences are vectors!
        # - readout layer is called with readout_params
        # - readout also takes the input vector as input (for BRM)

        n_visible = self.n_visible
        n_hidden = self.n_hidden
        n_hidden_readout = self.n_readout_hidden

        # Get model independent parameters:
        # TODO: explain this in docstring!
        readin_params, readin_operators = self.get_readin_params()
        rnn_params, rnn_operators = self.get_rnn_params()  # shape (depth, ..., ...)
        readout_params, readout_operators = self.get_readout_params()

        params = readin_params + rnn_params + readout_params

        # for debugging and inspection purposes:
        self.operators = readin_operators + rnn_operators + readout_operators

        # Input variable:
        v = T.matrix()  # shape (timesteps, n_visible)
        v.tag.test_value = np.random.randn(5, n_visible).astype(theano.config.floatX)

        # Apply noise to input:
        #NOTE: damn this is slow!!
        if self.input_noise > 0. and self.training_mode:  # overrides input dropout!
            noise = rng.binomial(size=v.shape,
                                 n=1,
                                 p=self.input_noise,
                                 dtype=theano.config.floatX)
            v1 = T.neq(v, noise).astype(theano.config.floatX)  # zero if equal, one otherwise
        # TODO: fix dropout!!
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

        ####################
        ### Readin layer ###
        ####################
        h0 = self.readin_layer(readin_operators, v1)  # shape (timesteps, n_hidden)

        ##################
        ### Recurrence ###
        ##################
        def evolve_gen(rnn_params_, h_l_tm1, h_lm1_t):
            """ Given h_l_tm1, h_lm1_t, returns h_l_t.
            Used in generative mode.
            :param rnn_params_:
            :param h_l_tm1: shape (n_hidden, )
            :param h_lm1_t:
            :return:
            """

            h_l_t = self.rnn_layer(rnn_params_, h_l_tm1, h_lm1_t)

            return h_l_t

        def evolve_trn(h_lm1_t, h_l_tm1, rnn_params_):
            """ Given h_l_tm1, h_lm1_t, returns h_l_t.
            Used in training mode.
            :param h_lm1_t:
            :param h_l_tm1:
            :param rnn_params_:
            :return:
            """

            h_l_t = self.rnn_layer(rnn_params_, h_l_tm1, h_lm1_t)

            return h_l_t

        #####################
        ### TRAINING MODE ###
        #####################
        # - outer loop over layers, inner loop over time
        h_init_vals = np.zeros((self.depth, n_hidden)).astype(theano.config.floatX)
        h_init_trn = theano.shared(h_init_vals, name='h_init_trn')
        self.h_init_trn = h_init_trn
        # try load:
        if self.filename is not None:  # override with saved state values
            try:
                set_loaded_parameters(self.filename, [h_init_trn])
            except:
                #print('Failed to load parameters... creating new file.')
                pass
        # add to parameters:
        #params += [h_init_trn]  # NOT INCLUDED since violates TI


        def layer_recurrence(rnn_params_, h_init,  h_lm1):
            """Propagates hidden state of shape (timesteps, n_hidden)
            upwards to higher layer.
            """

            h_l, updates_trn_inner = theano.scan(evolve_trn,
                                                sequences=[h_lm1],  # slice (n_hidden, )
                                                outputs_info=[h_init],
                                                non_sequences=rnn_params_,
                                                name='recurrence train scan')

            return h_l, updates_trn_inner  # shape (timesteps, n_hidden)

        # perform layer-wise recurrence:
        h_1_to_L, updates_trn = theano.scan(layer_recurrence,
                    sequences=rnn_operators + [h_init_trn],  # shapes (depth, ...)
                    outputs_info=[h0],  # shape (timesteps, n_hidden)
                    name='layer recurrence scan')
        # shape (depth, timesteps, n_hidden)

        # get last layer for readout:
        # h_L = h_1_to_L[-1]  # shape (timesteps, n_hidden)

        #############################
        ### Training mode readout ###
        #############################
        depth = self.depth
        n_layers = self.n_layers
        contributing_layers = h_1_to_L[depth - n_layers:]
        v_sample, cost, monitor, updates_trn2 = self.readout_layer(readout_operators,
                                                                   v,
                                                                   contributing_layers,
                                                                   self.ext)
        # - RNG updates are needed in the case of RBM
        updates_trn.update(updates_trn2)


        #######################
        ### GENERATIVE MODE ###
        #######################
        # - outer loop over time, inner loop over layers

        # Generating mode init variables:
        h_init_gen = T.matrix()
        h_init_gen.tag.test_value = np.random.randn(depth, n_hidden).astype(np.float32)
        #print(rnn_operators.eval())  # is ok

        def time_recurrence(h_1_to_L_tm1):  # h ALL layers, shape=(depth, n_hidden)

            h_in = h_1_to_L_tm1[depth - n_layers:, None, :]  # shape (n_layers, 1, n_hidden)
            v_t, _, _, updates_gen = self.readout_layer(readout_operators,
                                                        T.zeros((1, n_visible)),
                                                        h_in,
                                                        self.ext)  # shape (1, n_visible)
            # Readin layer:
            h0_t = self.readin_layer(readin_operators, v_t)[0]  # shape (n_hidden, )

            # Evolve through layers:
            h_1_to_L_t, updates_gen2 = theano.scan(evolve_gen,
                                sequences=rnn_operators + [h_1_to_L_tm1],
                                outputs_info=[h0_t],
                                name='uppropagate gen scan')

            if updates_gen is not None:
                updates_gen.update(updates_gen2)
            else:
                updates_gen = OrderedDict()

            return [h_1_to_L_t, v_t], updates_gen

        (_, v_gen), updates_gen = theano.scan(time_recurrence,
                    outputs_info=[h_init_gen, None],
                    n_steps=self.timesteps,
                    name='time recurrence scan')
        v_gen = v_gen[:, 0, :]

        return (v, v_sample, cost, monitor, params, updates_trn, v_gen,
                updates_gen, h_1_to_L, h_init_gen, h0)

    def adadelta(self, params, gradient, lr, gamma):
        """Adadelta optimizer function.

        - gamma = 1 is SGD

        :param params: params list
        :return:
        """
        eps_ = self.epsilon

        # initialize state:
        gtm1 = [shared_zeros(param.get_value().shape, param.name + '_g') for param in params]
        stm1 = [shared_zeros(param.get_value().shape, param.name + '_s') for param in params]

        if self.filename is not None:  # override with saved state values
            print('Loading parameters from file {}.zip'.format(self.filename))
            sys.stdout.flush()
            try:
                set_loaded_parameters(self.filename, gtm1 + stm1)
            except:
                print('Failed to load parameters... creating new file.')
                sys.stdout.flush()
                pass

        # for storing the full state of optimizer:
        self.gtm1 = gtm1
        self.stm1 = stm1
        self.gradient = gradient

        gt = [(1 - gamma) * T.clip(grad_i, -1, 1) ** 2 + gamma * gtm1_i
              for grad_i, gtm1_i in zip(gradient, gtm1)]

        dparams = [T.sqrt((stm1_i + eps_) / (gt_i + eps_)) *
                   T.clip(grad_i, -1, 1)
                   for stm1_i, gt_i, grad_i in
                   zip(stm1, gt, gradient)]

        st = [(1 - gamma) * dpar_i ** 2 + gamma * stm1_i
              for dpar_i, stm1_i in
              zip(dparams, stm1)]

        param_updates = [(p, p - lr * dp) for p, dp in zip(params, dparams)]
        gt_updates = zip(gtm1, gt)
        st_updates = zip(stm1, st)

        return param_updates + gt_updates + st_updates

    def sgd(self, params, gradient, lr):

        return [(p, p - lr * g) for p, g in zip(params, gradient)]

    def train(self, dataset, lr=1., gamma=.9, beta1=0.9, beta2=0.999,
              min_batch_size=100, max_batch_size=None, num_epochs=200,
              save_as=None, early_stopping=-1E+6):
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

        if type(dataset) is not list:
            dataset = [dataset]

        self.dataset = dataset
        nsamples = len(dataset)

        # flatten all parameters into an array for FLANN NN computation:
        # TODO: not sure if this is very useful, since the parameter space
        # is very high dimensional... basically parameters can bounce around
        # minimum and NN will not converge to zero in a long time... take e.g.
        # large dim. arrays with random 0, 1's and compute nn after generating
        # a new one each time! But hmm I think it should be steadily decreasing...
        # anyway think about it
        # import pyflann
        # flann = pyflann.FLANN()
        # pyflann.set_distance_type('euclidean')

        param_vec = self.param_vec
        if param_vec is None:
            param_vec = np.array([])
            for param in self.parameters:
                param_vec = np.concatenate((param_vec, param.get_value().flatten()))
            param_vec = param_vec[None, :]

        if max_batch_size is None:
            max_batch_size = min_batch_size
            mean_batch_size = min_batch_size
        else:
            mean_batch_size = int((min_batch_size + max_batch_size) / 2)

        best_monitor = 1E+6
        done = False
        try:
            for epoch in xrange(num_epochs):
                if done:
                    break
                start_time = time.time()

                costs = []
                monitors = []

                # shuffle dataset:
                shuffle(dataset)

                for sample_number, sample in enumerate(dataset):

                    # split to batches:
                    sample_size = len(sample)
                    idx = np.random.randint(min_batch_size,
                                            max_batch_size + 1,
                                            int(sample_size / min_batch_size)).cumsum()
                    idx = idx[idx < sample_size - min_batch_size]
                    batches = np.split(sample, idx)
                    shuffle(batches)
                    nbatches = len(batches)

                    for n, batch in enumerate(batches):

                        # don't train with almost empty batch:
                        if np.sum(batch) < min_steps_in_batch:
                            continue
                        if batch.shape[0] < 3:  # just in case...
                            continue

                        monitor, cost = self.train_function(batch, *hyperparams)

                        #TODO: revert to saved parameters in case of nans (?)
                        if np.isnan(cost):
                            raise ValueError('\nNaN encountered, breaking out!')

                        if np.abs(cost) > 1E+9:
                            raise ValueError('\nCost blew up, breaking out!')

                        costs.append(cost)
                        monitors.append(monitor)
                        pct_progress = int(100 * n / nbatches)
                        print('\rSample: {:6}/{} -- Progress: {:3}% -- '
                              'Cost={:6.3f} -- Monitor={:6.3f}'.format(sample_number + 1,
                                                                       nsamples,
                                                                       pct_progress,
                                                                       float(cost),
                                                                       float(monitor)), end='')
                        sys.stdout.flush()
                        if monitor < early_stopping:
                            print('\nEarly stop.')
                            done = True
                            break

                costs = np.asarray(costs)
                monitors = np.asarray(monitors)
                costs[costs > 1e+6] = 1e+6  # getting rid of infs
                monitors[monitors > 1e+6] = 1e+6  # getting rid of infs
                self.costs = costs
                self.monitors = monitors
                avg_cost = np.round(np.mean(costs), 4)
                std_cost = np.round(np.std(costs), 4)
                avg_monitor = np.round(np.mean(monitors), 4)
                std_monitor = np.round(np.std(monitors), 4)
                time_elapsed = time.time() - start_time

                # Nearest neighbors in parameter space:
                # param_vec_next = np.array([])
                # for param in self.parameters:
                #     param_vec_next = np.concatenate((param_vec_next, param.get_value().flatten()))

                #flann_params = flann.build_index(param_vec, target_precision=.9)
                #nn_dist = np.sqrt(flann.nn(param_vec, param_vec_next, 1)[1][0])

                # add to previous parameter vectors:
                # param_vec = np.vstack((param_vec, param_vec_next))

                print('\rEpoch {:4}/{} | Cost mean={:6.3f}, std={:6.3f} | '
                      'Monitor mean={:6.4f}, std={:6.3f} | '
                      'Time={} s\n'.format(epoch + 1,
                                                               num_epochs,
                                                               avg_cost,
                                                               std_cost,
                                                               avg_monitor,
                                                               std_monitor,
                                                               np.round(time_elapsed, 0)), end='')
                sys.stdout.flush()

                if save_as is not None and avg_monitor < best_monitor:
                    #print('Saving results...')
                    best_monitor = avg_monitor
                    # save full state, not just parameters:
                    # param_list = []
                    # gtm1_list = []
                    # stm1_list = []
                    # for n in range(len(self.parameters)):
                    #     try:
                    #         param_list.append(self.parameters[n])
                    #         gtm1_list.append(self.gtm1[n])
                    #         stm1_list.append(self.stm1[n])
                    #     except:
                    #         break

                    saved_state = self.parameters + self.gtm1 + self.stm1

                    with open(save_as + '.zip', 'w') as f:
                        dump(saved_state, f)

                    # savefile = file(save_as + '.save', mode='wb')
                    # cPickle.dump(saved_state, savefile)
                    # savefile.close()




        except KeyboardInterrupt:
            self.costs = costs
            self.monitors = monitors
            print('\nInterrupted by user.')

    def generate(self,
                 filename=None,
                 show=True,
                 timesteps=200,
                 initial_data=None,
                 temperature=1.,
                 ext_magnitude=0.,
                 ext_regularization=.5,
                 ext_overemphasize=1.):
        """

        :param filename:
        :param show:
        :param timesteps:
        :param initial_data:
        :param ext_magnitude:
        :param ext_regularization:
        :param ext_overemphasize: values > 1. will overemphasize center configurations
        :return:

        Generate a sample sequence, plot the resulting piano-roll and save
        it as a MIDI file.

        - Uses pythonmidi and Hexahedria's noteStateMatrixToMidi

        filename : string, None
            A MIDI file will be created at this location. If filename=None,
            will not create a MIDI file but just returns the piano roll.
        show : boolean
            If True, a piano-roll of the generated sequence will be shown."""

        # set to generate mode:
        self.training_mode = False

        # get initial hidden state from seed data:
        if initial_data is not None:
            initial_state = self.generate_hidden_states(initial_data, 0., .9)[1][:, -1, :]
        else:
            #initial_state = np.zeros((self.depth, self.n_hidden), dtype=theano.config.floatX)
            #initial_state = np.random.randn(self.depth, self.n_hidden).astype(theano.config.floatX)
            initial_state = self.h_init_trn.get_value()
        #print(initial_state.shape) # shape (depth, n_hidden)

        # construct external field:
        n_vis = self.n_visible
        x = np.arange(n_vis)
        ext = -ext_magnitude * ((n_vis / 2 + ext_regularization) ** 2 /
                               (x + ext_regularization) /
                               (n_vis - x + ext_regularization) -
                               ext_overemphasize)
        ext = ext.astype(theano.config.floatX)
        self.ext.set_value(ext)

        # adjust temperature:
        if temperature != 1.:
            print('Generating with temperature {}'.format(temperature))
            sys.stdout.flush()
            if self.readout_layer == rbm_readout:
                rbm_params = self.parameters[3:8]
            elif self.readout_layer == rbm_tip_readout:
                rbm_params = self.parameters[10:-1]

            # bake in temperature:
            for param in rbm_params:
                param_val = param.get_value()
                param_val /= temperature
                param.set_value(param_val)
            # generate:
            generated_data = self.generate_function(timesteps, initial_state)
            generated_data = np.round(generated_data)
            # restore temperature = 1 values:
            for param in rbm_params:
                param_val = param.get_value()
                param_val *= temperature
                param.set_value(param_val)



        else:
            generated_data = self.generate_function(timesteps, initial_state)
            generated_data = np.round(generated_data)

        if initial_data is not None:
            self.generated_data = np.concatenate((initial_data, generated_data), axis=0).astype(np.int64)
        else:
            self.generated_data = generated_data.astype(np.int64)

        if filename is not None:

            statematrix = statemat_from_pianoroll(self.generated_data)
            noteStateMatrixToMidi(statematrix, name=filename)

        else:
            return self.generated_data

        if show:
            #extent = (0, self.dt * len(self.generated_data)) + self.r
            plt.figure(figsize=(12, 8))
            #plt.imshow(self.generated_data.T, origin='lower', aspect='auto',
            #             interpolation='nearest', cmap=plt.cm.gray_r,
            #             extent=extent)
            plt.pcolor(self.generated_data.T, cmap='Greens')
            if initial_data is not None:
                plt.vlines(initial_data.shape[0], 0, self.n_visible, colors='red')
            plt.xlabel('timestep')
            plt.ylabel('MIDI note number')
            plt.title('Seed data and generated piano-roll')

        # revert to training mode:
        self.training_mode = True

        # reset external field:
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


    def get_readin_params(self):

        n_visible = self.n_visible
        n_hidden = self.n_hidden
        scale = self.readin_input_scale

        if self.readin_layer == tanh_readin:
            W_in_val = np.sqrt(scale / n_visible) * np.random.randn(n_visible, n_hidden)
            b_in_val = np.random.randn(n_hidden) * 0.05

            W_in = theano.shared(W_in_val.astype(theano.config.floatX), 'W_in')
            b_in = theano.shared(b_in_val.astype(theano.config.floatX), 'b_in')

            readin_params = [W_in, b_in]

        if self.filename is not None:  # override with saved state values
            try:
                set_loaded_parameters(self.filename, readin_params)
            except:
                #print('Failed to load parameters... creating new file.')
                pass

        return readin_params, readin_params


    def get_rnn_params(self):

        n_hidden = self.n_hidden
        sparsity = self.sparsity
        rec_spec_rad = self.recurrence_spectral_radius
        in_spec_rad = self.recurrence_input_scale
        depth = self.depth

        if self.rnn_layer == sru_layer:
            Ws_sru_val = get_random_sparse_square(n_hidden, sparsity, rec_spec_rad)
            #Ws_sru_val = np.eye(n_hidden, dtype=theano.config.floatX)
            Ws_in_val = get_random_sparse_square(n_hidden, sparsity, in_spec_rad)
            Ws_sru_val = np.concatenate((Ws_sru_val, Ws_in_val), axis=0)
            Ws_sru_val = Ws_sru_val[None, :, :]
            for n in xrange(depth - 1):
                Ws_layer = get_random_sparse_square(n_hidden, sparsity, rec_spec_rad)
                Ws_in_layer = get_random_sparse_square(n_hidden, sparsity, in_spec_rad)
                Ws_layer = np.concatenate((Ws_layer, Ws_in_layer), axis=0)
                Ws_layer = Ws_layer[None, :, :]
                Ws_sru_val = np.concatenate((Ws_sru_val, Ws_layer), axis=0)

            Ws_sru = theano.shared(Ws_sru_val.astype(theano.config.floatX), 'Ws_sru')

            rnn_params = [Ws_sru]  # both weight matrices combined

        elif self.rnn_layer == gru_layer:

            # Weights and reset + update gate biases:
            Ws_gru_val = np.zeros((depth, 2 * n_hidden, 3 * n_hidden + 1))
            for layer in xrange(depth):
                for n in range(3):  # h, z and r
                    recurrence_mat = get_random_sparse_square(n_hidden, sparsity, rec_spec_rad)
                    Ws_gru_val[layer, :n_hidden, n * n_hidden:(n + 1) * n_hidden] = recurrence_mat
                for n in range(3):  # input matrices
                    recurrence_mat = get_random_sparse_square(n_hidden, sparsity, in_spec_rad)
                    Ws_gru_val[layer, n_hidden:, n * n_hidden:(n + 1) * n_hidden] = recurrence_mat

                update_bias = np.linspace(-5, 5, num=n_hidden)
                reset_bias = 0.01 * np.random.randn(n_hidden)
                bias_vec = np.concatenate((update_bias, reset_bias))
                Ws_gru_val[layer, :, -1] = bias_vec

            Ws_gru = theano.shared(Ws_gru_val.astype(theano.config.floatX), 'Ws_gru')


            rnn_params = [Ws_gru]

        elif self.rnn_layer == kpz_layer:

            Ws_kpz_val = np.zeros((depth, n_hidden, 2 * n_hidden + 1))
            for layer in xrange(depth):
                W = np.random.randn(n_hidden, n_hidden)
                maxeig = np.max(np.abs(np.linalg.eigvals(W)))
                W *= rec_spec_rad / maxeig
                U = in_spec_rad * np.random.randn(n_hidden, n_hidden) / np.sqrt(n_hidden)
                w = 0.01 * np.random.randn(n_hidden)



        else:
            raise Exception('No recurrence layer set!')

        if self.filename is not None:  # override with saved state values
            try:
                set_loaded_parameters(self.filename, rnn_params)
            except:
                #print('Failed to load parameters... creating new file.')
                pass

        return rnn_params, rnn_params


    def get_readout_params(self):

        n_visible = self.n_visible
        n_hidden = self.n_hidden
        n_hidden_readout = self.n_readout_hidden
        scale = self.readout_scale

        if self.readout_layer == sigmoid_readout:
            W_out_val = np.sqrt(scale / n_hidden) * np.random.randn(n_hidden, n_visible)
            b_out_val = np.random.randn(n_visible) * 0.05

            W_out = theano.shared(W_out_val.astype(theano.config.floatX), 'W_out')
            b_out = theano.shared(b_out_val.astype(theano.config.floatX), 'b_out')

            readout_params = [W_out, b_out]

        elif self.readout_layer == rbm_readout:

            Mmat_val = np.sqrt(2 / n_visible) * np.random.randn(n_visible, n_hidden_readout)

            Whidvis_val = np.sqrt(2 / n_hidden) * np.random.randn(n_hidden, n_visible)

            Whidhid_val = np.sqrt(2 / n_hidden) * np.random.randn(n_hidden, n_hidden_readout)

            bvis_val = -scale * np.ones((n_visible, ), dtype=theano.config.floatX)
            bhid_val = np.zeros((n_hidden_readout, ), dtype=theano.config.floatX)

            Mmat = theano.shared(Mmat_val.astype(theano.config.floatX), 'Mmat')
            Whidvis = theano.shared(Whidvis_val.astype(theano.config.floatX), 'Whidvis')
            Whidhid = theano.shared(Whidhid_val.astype(theano.config.floatX), 'Whidhid')
            bvis = theano.shared(bvis_val, 'bvis')
            bhid = theano.shared(bhid_val, 'bhid')

            readout_params = [Mmat, Whidvis, Whidhid, bvis, bhid]

        elif self.readout_layer == softmax_readout:
            W_out_val = np.sqrt(scale / n_hidden) * np.random.randn(n_hidden, n_visible)
            b_out_val = np.zeros(n_visible, dtype=theano.config.floatX)

            W_out = theano.shared(W_out_val.astype(theano.config.floatX), 'W_out')
            b_out = theano.shared(b_out_val.astype(theano.config.floatX), 'b_out')

            readout_params = [W_out, b_out]

        if self.filename is not None:  # override with saved state values
            try:
                set_loaded_parameters(self.filename, readout_params)
            except:
                #print('Failed to load parameters... creating new file.')
                pass

        return readout_params, readout_params


class InvariantDeepRNN(DeepRNN):
    """Translation invariant and periodic (TIP) version of the DeepRNN class.

    - No point in using readout_width > 1!!!

    """

    def __init__(self, **kwargs):
        """
        kwargs:
        width -- int; multiplier of input-to-rnn dimension.
            Will determine the total number of independent parameters
            as (5 * k + 6) * n_visible (for GRU).
        """
        readin_layer = kwargs.pop('readin_layer')
        rnn_layer = kwargs.pop('rnn_layer')
        readout_layer = kwargs.pop('readout_layer')

        try:
            self.constrain_eigvals = kwargs.pop('constrain_eigvals')  # boolean
        except KeyError:
            self.constrain_eigvals = False
            print('Eigenvalues not constrained.')

        if readin_layer is 'tanh':
            self.readin_layer = tanh_tip_readin
            kwargs['readin_layer'] = None
        if rnn_layer is 'sru':
            self.rnn_layer = sru_tip_layer
            kwargs['rnn_layer'] = None
        elif rnn_layer is 'gru':
            self.rnn_layer = gru_tip_layer
            kwargs['rnn_layer'] = None
        if readout_layer is 'rbm':
            self.readout_layer = rbm_tip_readout
            kwargs['readout_layer'] = None

        # call other parent class inits:
        super(InvariantDeepRNN, self).__init__(**kwargs)
        #DeepRNN.__init__(self, **kwargs)


    def get_readin_params(self):
        """Get the model parameters and operators. Since dot products are
        represented as convolutions, the operators will also be vectors
        of shape (n_in + n_out - 1, )
        """

        n_visible = self.n_visible
        n_hidden = self.n_hidden
        scale = self.readin_input_scale
        #if self.readin_layer == tanh_tip_readin:
        w_in_val = scale * np.sqrt(2 / n_visible) * np.random.randn(n_visible)
        b_in_val = np.array(0.)

        # W_in_val = np.sqrt(3 / n_visible) * np.random.randn(n_visible, n_hidden)
        # b_in_val = np.zeros(n_hidden).astype(theano.config.floatX)

        w_in = theano.shared(w_in_val.astype(theano.config.floatX), 'w_in')
        b_in = theano.shared(b_in_val.astype(theano.config.floatX), 'b_in')

        readin_params = [w_in, b_in]

        # get operators:
        w_in_op = rep_vec(w_in, n_visible, n_hidden)[0]

        readin_operators = [w_in_op, b_in]

        if self.filename is not None:  # override with saved state values
            try:
                set_loaded_parameters(self.filename, readin_params)
            except:
                #print('Failed to load parameters... creating new file.')
                pass

        return readin_params, readin_operators


    def get_rnn_params(self):
        """

        uhs = operators[0]
        w0s = operators[1]
        uzs = operators[2]
        wzs = operators[3]
        urs = operators[4]
        wrs = operators[5]

        :return:
        """

        n_hidden = self.n_hidden
        sparsity = self.sparsity
        rec_spec_rad = self.recurrence_spectral_radius
        input_rad = self.recurrence_input_scale  # input mat is now also orthogonal
        depth = self.depth

        if self.rnn_layer == sru_tip_layer:
            rnn_params, rnn_operators = get_sru_tip_params(n_hidden,
                                                           sparsity,
                                                           rec_spec_rad,
                                                           input_rad,
                                                           depth,
                                                           self.constrain_eigvals)

        elif self.rnn_layer == gru_tip_layer:

            rnn_params, rnn_operators = get_gru_tip_params(n_hidden,
                                                           sparsity,
                                                           rec_spec_rad,
                                                           input_rad,
                                                           depth,
                                                           self.constrain_eigvals)

        if self.filename is not None:  # override with saved state values
            try:
                set_loaded_parameters(self.filename, rnn_params)
            except:
                #print('Failed to load parameters... creating new file.')
                pass

        return rnn_params, rnn_operators


    def get_readout_params(self):

        n_visible = self.n_visible
        n_hidden = self.n_hidden
        n_hidden_readout = self.n_readout_hidden
        scale = self.readout_scale  # this now will adjust RBM hidden *bias*!!
        depth = self.depth
        n_layers = self.n_layers

        # mvec_val = np.sqrt(3 / n_visible) * np.random.randn(n_visible)
        mvec_val = 0.01 * np.random.randn(n_visible)
        whidvis_vals = np.sqrt(2 / n_hidden) * np.random.randn(n_layers, n_visible)
        whidhid_vals = np.sqrt(2 / n_hidden) * np.random.randn(n_layers, n_hidden)
        bvis_val = np.array(-scale)
        bhid_val = np.array(0.)

        # Mmat_val = np.sqrt(3 / n_visible) * np.random.randn(n_visible, n_hidden_readout)
        # Whidvis_val = np.sqrt(3 / n_hidden) * np.random.randn(n_hidden, n_visible)
        # Whidhid_val = np.sqrt(3 / n_hidden) * np.random.randn(n_hidden, n_hidden_readout)
        # bvis_val = np.zeros((n_visible, ), dtype=theano.config.floatX)
        # bhid_val = np.zeros((n_hidden_readout, ), dtype=theano.config.floatX)

        mvec = theano.shared(mvec_val.astype(theano.config.floatX), 'mmat')
        whidvis = theano.shared(whidvis_vals.astype(theano.config.floatX), 'whidvis')
        whidhid = theano.shared(whidhid_vals.astype(theano.config.floatX), 'whidhid')
        bvis = theano.shared(bvis_val.astype(theano.config.floatX), 'bvis')
        bhid = theano.shared(bhid_val.astype(theano.config.floatX), 'bhid')

        readout_params = [mvec, whidvis, whidhid, bvis, bhid]

        # get operators:
        mvec_op = rep_vec(mvec, n_visible, n_hidden_readout)[0]
        whidvis_op = rep_vec(whidvis, n_visible, n_hidden)  # shape (n_layers, n_vis + n_hid - 1)
        whidhid_op = rep_vec(whidhid, n_hidden, n_hidden_readout)  # shape (n_layers, n_hid + n_hid_ro - 1)

        readout_operators = [mvec_op, whidvis_op, whidhid_op, bvis, bhid]

        if self.filename is not None:  # override with saved state values
            try:
                set_loaded_parameters(self.filename, readout_params)
            except:
                #print('Failed to load parameters... creating new file.')
                pass

        return readout_params, readout_operators





























