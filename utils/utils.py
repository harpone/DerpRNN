from __future__ import division, print_function

# Heikki Arponen (2015)
# heikki@quantmechanics.com


import numpy as np
from theano import tensor as T
import cPickle
import midi
import os
import theano
#import pyflann
from ctools.nonpytools import *
from theano.tensor.signal.conv import conv2d as conv_signal
convolution = conv_signal


def set_loaded_parameters_old(filename, params):
    file_ = file(filename + '.save', 'rb')
    loaded_params = cPickle.load(file_)
    file_.close()

    for param in params:
        for loaded_param in loaded_params:
            if param.name == loaded_param.name:
                param.set_value(loaded_param.get_value())
                break

def set_loaded_parameters(filename, params):
    loaded_params = np.load(filename + '.zip')

    for param in params:
        for loaded_param_name in loaded_params.keys()[:-1]:  # no 'pkl'
            if param.name == loaded_param_name:
                param.set_value(loaded_params[loaded_param_name])
                break


def drelu(x):
    """
    Compute the element-wise rectified double linear activation function.
    Parameters
    ----------
    x : symbolic tensor
        Tensor to compute the activation function for.
    Returns
    -------
    symbolic tensor
        Element-wise "double rectifier" applied to `x`.
    """

    return 0.5 * (x - 1 + abs(x - 1)) + 0.5 * (x + 1 - abs(x + 1))



def shared_normal(num_rows, num_cols, scale=1., name=None):
    """Initialize a matrix shared variable with normally distributed
    elements."""
    if num_rows == 0:
        arr_value = np.random.normal(
        scale=scale, size=(num_cols, )).astype(theano.config.floatX)
    else:
        arr_value = np.random.normal(
        scale=scale, size=(num_rows, num_cols)).astype(theano.config.floatX)

    return theano.shared(arr_value, name=name)


def shared_zeros(shape, name=None):
    """Initialize a vector shared variable with zero elements."""
    return theano.shared(np.zeros(shape, dtype=theano.config.floatX), name=name)


def get_random_sparse(size, connections):
    """
    Obtain a matrix with normal * scale distributed values
    with random 'connections' number of connections from
    previous to next time step hidden units.
    :param size:
    :param connections:
    :return:
    """
    rand_mat = np.random.randn(size, size)
    idx_mat = np.zeros((size, size))
    idx_mat[:, :connections] = 1
    map(np.random.shuffle, idx_mat)
    return (idx_mat * rand_mat).astype(theano.config.floatX)


def get_random_sparse_square(size, sparsity, spectral_radius):
    """

    :param size:
    :param sparsity:
    :param spectral_radius:
    :return:
    """
    rand_mat = np.random.randn(size **2)
    idx_mat = np.zeros_like(rand_mat)
    connections = int(size ** 2 * sparsity)
    idx_mat[:connections] = 1
    idx_mat = idx_mat[None, :]
    map(np.random.shuffle, idx_mat)

    rand_mat = rand_mat.reshape((size, size))
    idx_mat = idx_mat.reshape((size, size))

    out = spectral_radius * idx_mat * rand_mat / (size * sparsity) ** .5

    return out.astype(theano.config.floatX)

def get_random_sparse_matrix(shape0, shape1, sparsity, spectral_radius):
    """BULLSHIT!! Don't use...

    :param shape0, shape1:
    :param sparsity:
    :param spectral_radius:
    :return:
    """
    rand_mat = np.random.randn(shape0 * shape1)
    idx_mat = np.zeros_like(rand_mat)
    connections = int(shape0 * shape1 * sparsity)
    idx_mat[:connections] = 1
    idx_mat = idx_mat[None, :]
    map(np.random.shuffle, idx_mat)

    rand_mat = rand_mat.reshape((shape0, shape1))
    idx_mat = idx_mat.reshape((shape0, shape1))

    # use magic number to get approx. max svd = 1:
    out = spectral_radius ** 2 * idx_mat * rand_mat / ((shape0 + shape1) * sparsity) ** .5 / 1.283

    return out.astype(theano.config.floatX)


def get_random_sparse_ti(size, connections):
    """
    Translation invariant version of get_random_sparse.
    :param size:
    :param connections:
    :return:
    """
    rand_vec = np.random.randn(1, 2 * size - 1)
    idx_vec = np.zeros(2 * size - 1)
    idx_vec[:connections] = 1
    np.random.shuffle(idx_vec)

    sparse_vec = idx_vec * rand_vec

    # reshape to square and return also that:
    sparse_mat = np.tile(sparse_vec, (size, 1))
    inc = 0
    for n in xrange(sparse_mat.shape[0]):
        sparse_mat[n] = np.roll(sparse_mat[n], inc)
        inc += 1
    sparse_mat = sparse_mat[:, size - 1:].T
    return sparse_vec, sparse_mat


def get_spectral_radius(mat):
    """Damn slow for big matrices... could be replaced
    with a fast max eigenvalue algo.

    :param mat:
    :return:
    """
    return np.max(np.abs(np.linalg.eig(mat)[0]))


def compute_entropy(data_array):
    """data_array.shape = (num_samples, sample_size)
    """

    flann = pyflann.FLANN()
    pyflann.set_distance_type('euclidean')
    params = flann.build_index(data_array, target_precision=.9)

    _, dists = flann.nn_index(data_array, 2, checks=params["checks"])
    dists = np.sqrt(dists[:, 1])
    dists = dists[dists > 0.]

    num_samples = dists.shape[0]
    entropy = 0.
    for n in xrange(num_samples):
        entropy += np.log(dists[n]) / num_samples

    return entropy


def midiToNoteStateMatrix(midifile, lowerbound=21, upperbound=109):
    # code from: https://github.com/hexahedria/biaxial-rnn-music-composition/

    pattern = midi.read_midifile(midifile)

    timeleft = [track[0].tick for track in pattern]

    posns = [0 for track in pattern]

    statematrix = []
    span = upperbound - lowerbound
    time = 0

    state = [[0,0] for x in range(span)]
    statematrix.append(state)
    while True:
        if time % (pattern.resolution / 4) == (pattern.resolution / 8):
            # Crossed a note boundary. Create a new state, defaulting to holding notes
            oldstate = state
            state = [[oldstate[x][0],0] for x in range(span)]
            statematrix.append(state)

        for i in range(len(timeleft)):
            while timeleft[i] == 0:
                track = pattern[i]
                pos = posns[i]

                evt = track[pos]
                if isinstance(evt, midi.NoteEvent):
                    if (evt.pitch < lowerbound) or (evt.pitch >= upperbound):
                        pass
                        # print "Note {} at time {} out of bounds (ignoring)".format(evt.pitch, time)
                    else:
                        if isinstance(evt, midi.NoteOffEvent) or evt.velocity == 0:
                            state[evt.pitch - lowerbound] = [0, 0]
                        else:
                            state[evt.pitch - lowerbound] = [1, 1]
                elif isinstance(evt, midi.TimeSignatureEvent):
                    if evt.numerator not in (2, 4):
                        # We don't want to worry about non-4 time signatures. Bail early!
                        #print("Found non even time signature event. Bailing early!".format(evt))
                        return statematrix


                try:
                    timeleft[i] = track[pos + 1].tick
                    posns[i] += 1
                except IndexError:
                    timeleft[i] = None

            if timeleft[i] is not None:
                timeleft[i] -= 1

        if all(t is None for t in timeleft):
            break

        time += 1

    return statematrix


def noteStateMatrixToMidi(statematrix, name="example", lowerbound=21, upperbound=109):
    # code from: https://github.com/hexahedria/biaxial-rnn-music-composition/
    statematrix = np.asarray(statematrix)
    pattern = midi.Pattern()
    track = midi.Track()
    pattern.append(track)

    span = upperbound-lowerbound
    tickscale = 90

    lastcmdtime = 0
    prevstate = [[0,0] for x in range(span)]
    for time, state in enumerate(statematrix + [prevstate[:]]):
        offNotes = []
        onNotes = []
        for i in range(span):
            n = state[i]
            p = prevstate[i]
            if p[0] == 1:
                if n[0] == 0:
                    offNotes.append(i)
                elif n[1] == 1:
                    offNotes.append(i)
                    onNotes.append(i)
            elif n[0] == 1:
                onNotes.append(i)
        for note in offNotes:
            track.append(midi.NoteOffEvent(tick=(time-lastcmdtime)*tickscale, pitch=note+lowerbound))
            lastcmdtime = time
        for note in onNotes:
            track.append(midi.NoteOnEvent(tick=(time-lastcmdtime)*tickscale, velocity=40, pitch=note+lowerbound))
            lastcmdtime = time

        prevstate = state

    eot = midi.EndOfTrackEvent(tick=1)
    track.append(eot)

    midi.write_midifile("{}.mid".format(name), pattern)


def activitymat_to_statemat(mat):

    halves = (mat == 0.5) * mat * 2.  # both levels = 1
    ones = (mat == 1.) * mat # only held width = 1
    ones[halves == 1.] = 1.

    statemat_recon = np.concatenate((ones[:, :,  None], halves[:, :, None]), axis=2)

    return statemat_recon


def process_dataset(path, outname=None, verbose=False, min_length=100):
    """Loads bunch of MIDI files and converts them
    to piano roll.

    - Uses pythonmidi and Hexahedria's midiToNoteStateMatrix
    - Uses so far only 4-time signatures
    - Doubles resolution and inserts pauses before a same note
    is struck again to signal a new note instead of a held note

    :param path:
    :return: list
    """
    files = os.listdir(path)

    assert len(files) > 0, 'No data!'

    dataset = []

    for f in files:
        try:
            if verbose:
                print('File {}'.format(str(f)))
            statemat = np.array(midiToNoteStateMatrix(path + f))
            piano_roll = pianoroll_from_statemat(statemat).astype(np.float32)
            if piano_roll.shape > min_length:
                dataset.append(piano_roll)
            else:
                print('Song too short... skipping this one.')
                break

        except IOError:
            print('fuck')
            pass

    if outname is not None:
        np.savez(outname, dataset)
        print('File saved as {}.npz'.format(outname))

    return dataset


def roll_and_dot(wvec, xvec):
    """
    wvec.shape = (n_in, )
    xvec.shape = (timesteps, n_in)
    """

    dot = T.dot(xvec, wvec)
    wvec = T.roll(wvec, 1)

    return wvec, dot, xvec


def ti_dot(wvec, xvec, outdim):
    """
    wvec.shape = (n_in, )
    xvec.shape = (timesteps, n_in)
    """

    tip_vec, _ = theano.scan(roll_and_dot,
                             outputs_info=[wvec, None, None],
                             n_steps=outdim,
                             non_sequences=[xvec],
                             name='TI dot scan')

    return tip_vec[1].T


def rep_vec(w, n_smaller, n_larger):
    """Creates a periodic vector corresponding to the
    weight matrix.

    param w :: shape = (depth, n_in)

    returns w_rep :: shape = (depth, n_in + n_out - 1)

    - smaller dimension always first!
    """
    target = n_smaller + n_larger - 1
    nreps = int(np.ceil(target / n_smaller))
    w_rep = T.tile(w, (1, nreps), ndim=2)[:, :target]

    return w_rep


def tip_input_init(depth, num_columns, scale=2, name=None):
    """Glorot style initialization.
    These are actually more or less equivalent to tip_init, but nevermind...
    :param depth:
    :param num_columns:
    :param name:
    :return:
    """

    vec_list = []
    for n in xrange(depth):
        val_array = np.sqrt(scale / num_columns) * np.random.randn(num_columns)
        vec_list.append(val_array)
    asarr = np.asarray(vec_list, dtype=theano.config.floatX)
    arr = theano.shared(asarr, name=name)

    return arr



def tip_init(depth, num_columns, sparsity=.15, spectral_radius=1.1, name=None):
    """Initialize a TIP vector such that the corresponding square TIP
    sparse initialization matrix has a specified spectral radius.

    :param num_columns: number of independent weights
    :param sparsity: float between 0 and 1
        Percentage of nonzero weights
    :param spectral_radius: float
        Usually values between 1 and 1.5 are fine
    :param name: str
        Optional
    :return: Theano shared matrix of shape (1, num_columns)
    """

    vec_list = []
    for n in xrange(depth):
        vec_list.append(get_random_sparse_tip_vec(num_columns,
                                           sparsity,
                                           spectral_radius).astype(theano.config.floatX))
    asarr = np.asarray(vec_list, dtype=theano.config.floatX)
    arr = theano.shared(asarr, name=name)

    return arr


def tip_init_constrained(depth, n_hidden, rec_spec_rad, name=None):
    """

    :param depth:
    :param n_hidden:
    :param rec_spec_rad:
    :param name:
    :return:
    """


def get_random_sparse_tip_vec(size, sparsity, spectral_radius):

    # create random sparse vec:
    connections = int(round(sparsity * size))
    rand_vec = np.random.randn(size)
    idx_vec = np.zeros((size, ))
    idx_vec[:connections] = 1
    np.random.shuffle(idx_vec)
    vec = idx_vec * rand_vec

    # compute max abs eig:
    max_eigval = max_eig(vec)

    # rescale vector:
    vec *= spectral_radius / max_eigval

    return vec


def get_sru_tip_params(n_hidden, sparsity, rec_spec_rad, input_rad, depth, constrain_eigvals):
    """

    :param n_hidden:
    :param sparsity:
    :param rec_spec_rad:
    :param input_rad:
    :param depth:
    :param constrain_eigvals:
    :return:
    """
    # TODO: input matrices are back to nonorthogonal
    if constrain_eigvals:
        # these are actually now just the eigenvalue phases!!
        assert n_hidden % 2 == 0, 'Number of hidden units should be even...'
        rnd_phases = np.random.rand(depth, int(n_hidden / 2) - 1) * 2 * np.pi
        uhs = theano.shared(rnd_phases.astype(theano.config.floatX), name='uhs')  # rec weight
        #whs = theano.shared(rnd_phases.astype(theano.config.floatX), name='uhs')  # input weight
    else:
        uhs = tip_init(depth, n_hidden, sparsity, rec_spec_rad, name='uhs')

    whs = tip_init(depth, n_hidden, sparsity, input_rad, name='whs')

    # rec_spec_rad as parameter:
    rhos_val = rec_spec_rad * np.ones((depth, ))  # NOTE THE SHAPE!!
    rhos = theano.shared(rhos_val.astype(theano.config.floatX),  # NOT broadcastable now!!!
                         name='rhos')
                         #broadcastable=(False, True))
    # input_rad as parameter:
    #sigma = theano.shared(np.array(input_rad).astype(theano.config.floatX), name='sigma')

    rnn_params = [rhos, uhs, whs]

    if constrain_eigvals:
        uhs_op = get_uhs_operator(uhs, depth, n_hidden, rhos)
        #whs_op = get_uhs_operator(whs, depth, n_hidden, sigma)
    else:
        uhs_op = rep_vec(uhs, n_hidden, n_hidden)  # shape (depth, 2 * n_hidden - 1)

    whs_op = rep_vec(whs, n_hidden, n_hidden)

    rnn_operators = [T.concatenate((uhs_op,
                                    whs_op), axis=1)]

    return rnn_params, rnn_operators


def get_gru_tip_params(n_hidden, sparsity, rec_spec_rad, input_rad, depth, constrain_eigvals):
    """

    :param n_hidden:
    :param sparsity:
    :param rec_spec_rad:
    :param input_rad:
    :param depth:
    :param constrain_eigvals:
    :return:
    """

    if constrain_eigvals:
        # these are actually now just the eigenvalue phases!!
        assert n_hidden % 2 == 0, 'Number of hidden units should be even...'
        rnd_phases = np.random.rand(depth, int(n_hidden / 2) - 1) * 2 * np.pi
        uhs = theano.shared(rnd_phases.astype(theano.config.floatX), name='uhs')
    else:
        uhs = tip_init(depth, n_hidden, sparsity, rec_spec_rad, name='uhs')

    uzs = tip_init(depth, n_hidden, sparsity, rec_spec_rad, name='uzs')
    urs = tip_init(depth, n_hidden, sparsity, rec_spec_rad, name='urs')
    # TODO: how does this affect things?? (both prev layer outs and prev time
    # should have equal contribution in z_t, r_t)
    #uzs = tip_input_init(depth, n_hidden, input_scale, name='uzs')
    #urs = tip_input_init(depth, n_hidden, input_scale, name='urs')

    w0s = tip_input_init(depth, n_hidden, input_rad, name='w0s')
    wzs = tip_input_init(depth, n_hidden, input_rad, name='wzs')
    wrs = tip_input_init(depth, n_hidden, input_rad, name='wrs')
    bzs_val = np.linspace(3., -3., depth, dtype=theano.config.floatX)
    bzs = theano.shared(bzs_val[:, None], name='bzs')  # shape (depth, 1)
    brs_val = np.random.randn(depth).astype(theano.config.floatX)
    brs = theano.shared(brs_val[:, None], name='brs')  # shape (depth, 1)

    # rec_spec_rad as parameter:
    rho = theano.shared(np.array(rec_spec_rad).astype(theano.config.floatX), name='rho')

    rnn_params = [rho, uhs, uzs, urs, w0s, wzs, wrs, bzs, brs]

    # get operators:
    # TODO: won't work!! Theano can't take grad through complex nodes...
    # if self.constrain_eigvals:
    #     exp_phases = T.exp(1j * uhs)
    #     neg_exp_phases = T.exp(-1j * uhs[:, ::-1])
    #     ones_ = T.ones((depth, 1), dtype=theano.config.floatX)
    #     #TODO: note that there are now +-rho eigvals, sign quite arbitrarily selected...
    #     eigvals = rec_spec_rad * T.concatenate((ones_, exp_phases, -ones_, neg_exp_phases))
    #     a_k = T.real(fft(eigvals, n_hidden, 1))  # the uhs in old notation, i.e. Circ vector
    #     # TODO: fft returns complex128 dtype, T.real casts to float64...
    #     # Need to check that the casting is optimized!!
    #     a_k = a_k.astype(theano.config.floatX)
    #     uhs_op = rep_vec(a_k, n_hidden, n_hidden)  # shape (depth, 2 * n_hidden - 1)
    if constrain_eigvals:
        uhs_op = get_uhs_operator(uhs, depth, n_hidden, rho)
    else:
        uhs_op = rep_vec(uhs, n_hidden, n_hidden)  # shape (depth, 2 * n_hidden - 1)

    uzs_op = rep_vec(uzs, n_hidden, n_hidden)
    urs_op = rep_vec(urs, n_hidden, n_hidden)
    w0s_op = rep_vec(w0s, n_hidden, n_hidden)
    wzs_op = rep_vec(wzs, n_hidden, n_hidden)
    wrs_op = rep_vec(wrs, n_hidden, n_hidden)

    rnn_operators = [T.concatenate((uhs_op,
                                   uzs_op,
                                   urs_op,
                                   w0s_op,
                                   wzs_op,
                                   wrs_op,
                                   bzs,
                                   brs), axis=1)]

    return rnn_params, rnn_operators


def get_uhs_operator(uhs, depth, n_hidden, rhos):
    """

    :param uhs:
    :param depth:
    :param n_hidden:
    :param rhos: can be shared variable or constant of shape (depth, )!!
    :return:
    """
    # Will use a Fourier matrix (will be O(n^2)...)
    # Doesn't seem to slow things down much though!
    exp_phases = [T.cos(uhs), T.sin(uhs)]
    neg_exp_phases = [T.cos(uhs[:, ::-1]), -T.sin(uhs[:, ::-1])]
    ones_ = [T.ones((depth, 1), dtype=theano.config.floatX),
             T.zeros((depth, 1), dtype=theano.config.floatX)]

    rhos_reshaped = T.reshape(rhos, (depth, 1), ndim=2)
    rhos_reshaped = T.addbroadcast(rhos_reshaped, 1)

    eigvals_re = rhos_reshaped * T.concatenate((ones_[0],
                                               exp_phases[0],
                                               -ones_[0],
                                               neg_exp_phases[0]), axis=1)
    eigvals_im = rhos_reshaped * T.concatenate((ones_[1],
                                               exp_phases[1],
                                               -ones_[1],
                                               neg_exp_phases[1]), axis=1)
    phase_array = -2 * np.pi * np.outer(np.arange(n_hidden), np.arange(n_hidden)) / n_hidden
    f_array_re_val = np.cos(phase_array) / n_hidden
    f_array_im_val = np.sin(phase_array) / n_hidden
    f_array_re = theano.shared(f_array_re_val.astype(theano.config.floatX), name='f_arr_re')
    f_array_im = theano.shared(f_array_im_val.astype(theano.config.floatX), name='f_arr_im')

    a_k = T.dot(eigvals_re, f_array_re) + T.dot(eigvals_im, f_array_im)
    uhs_op = rep_vec(a_k, n_hidden, n_hidden)  # shape (depth, 2 * n_hidden - 1)

    return uhs_op




