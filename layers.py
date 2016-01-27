
from __future__ import division, print_function

__version__ = "v 0.1"

import theano
from theano import tensor as T
from theano.tensor.signal.conv import conv2d as conv_signal
#from theano.tensor.shared_randomstreams import RandomStreams
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from theano.sandbox.softsign import softsign
from theano.compile.debugmode import DebugMode

# Local imports:
from utils.utils import *

#Don't use a python long as this don't work on 32 bits computers.
np.random.seed(0xbeef)
rng = RandomStreams(seed=np.random.randint(1 << 30))
theano.config.warn.subtensor_merge_bug = False
theano.config.exception_verbosity = 'low'

# Debug mode:
#mode = 'DebugMode'
mode = None
#mode ='FAST_COMPILE'
theano.config.warn.signal_conv2d_interface = False

#convolution = conv_nnet  # won't work...
convolution = conv_signal
crossent = T.nnet.binary_crossentropy
eps = 1E-6

# global parameters:
min_steps_in_batch = 3  # require this many nonzero notes in batch

#TODO: note softsign!!!
#tanh = T.tanh
tanh = softsign
sigmoid = T.nnet.hard_sigmoid
softmax = T.nnet.softmax

def sru_layer(operators, h_l_tm1, h_lm1_t):
    """

    :param operators: One layer's concatenated weights of shape (2 * n_hidden, n_hidden).
        Weights W, U are vertically stacked.
    :param h_l_tm1: shape (n_hidden, )
    :param h_lm1_t:
    :return:
    """
    u_l_t = T.concatenate((h_l_tm1, h_lm1_t))
    argument = T.dot(u_l_t, operators)

    return tanh(argument)


def gru_layer(operators, h_l_tm1, h_lm1_t):
    """

    :param operators: One layer's concatenated weights and biases of shape
        (2 * n_hidden, 3 * n_hidden + 1). Last col is z, r biases
        Weights W, U are vertically stacked, weights for h, z, r horizontally
    :param h_l_tm1: shape (n_hidden, )
    :param h_lm1_t:
    :return:
    """
    n_hidden = T.shape(h_l_tm1)[0]
    u_l_t = T.concatenate((h_l_tm1, h_lm1_t))

    z_and_r = T.dot(u_l_t, operators[:, n_hidden: 3 * n_hidden])
    z_t = sigmoid(z_and_r[:n_hidden] + operators[:n_hidden, -1])
    r_t = sigmoid(z_and_r[n_hidden:] + operators[n_hidden:, -1])
    rho_t = T.concatenate((r_t, T.ones((n_hidden, ))), axis=0)
    utilda_l_t = rho_t * u_l_t
    h_arg = T.dot(utilda_l_t, operators[:, :n_hidden])

    h_l_t = (1 - z_t) * h_l_tm1 + z_t * tanh(h_arg)

    return h_l_t

def sru_tip_layer(operators, h_l_tm1, h_lm1_t):
    """

    :param operators:
    :param h_l_tm1:
    :param h_lm1_t:
    :return:
    """
    n_hidden = h_l_tm1.shape[0]
    eff_hidden = 2 * n_hidden - 1
    uh_op = operators[:eff_hidden]
    wh_op = operators[eff_hidden:2 * eff_hidden]

    h_lm1_t_dot_W = convolution(wh_op[None, :], h_lm1_t[None, :])[0]
    h_l_tm1_dot_U = convolution(uh_op[None, :], h_l_tm1[None, :])[0]

    h_l_t = tanh(h_l_tm1_dot_U + h_lm1_t_dot_W)

    return h_l_t



def gru_tip_layer(operators, h_l_tm1, h_lm1_t):
    """

    :param operators:
    :param h_l_tm1: shape (n_hidden, )
    :param h_lm1_t:
    :return:
    """
    #TODO: do in single convolution pass? Or ensure MergeOptimize works!
    n_hidden = h_l_tm1.shape[0]
    eff_hidden = 2 * n_hidden - 1
    uh_op = operators[:eff_hidden]
    uz_op = operators[eff_hidden:2 * eff_hidden]
    ur_op = operators[2 * eff_hidden:3 * eff_hidden]
    w0_op = operators[3 * eff_hidden:4 * eff_hidden]
    wz_op = operators[4 * eff_hidden:5 * eff_hidden]
    wr_op = operators[5 * eff_hidden:6 * eff_hidden]
    bz = operators[-2]
    br = operators[-1]

    #print(h_lm1_t.dtype)  # 64??
    #print(h_l_tm1.dtype)
    #TODO: MergeOptimize fails with these convolutions!
    u_lm1_t_dot_W0 = convolution(w0_op[None, :], h_lm1_t[None, :])[0]
    u_l_tm1_dot_Uz = convolution(uz_op[None, :], h_l_tm1[None, :])[0]
    u_lm1_t_dot_Wz = convolution(wz_op[None, :], h_lm1_t[None, :])[0]
    u_l_tm1_dot_Ur = convolution(ur_op[None, :], h_l_tm1[None, :])[0]
    u_lm1_t_dot_Wr = convolution(wr_op[None, :], h_lm1_t[None, :])[0]

    # update and reset gate evolution:
    z_t = T.nnet.sigmoid(u_l_tm1_dot_Uz + u_lm1_t_dot_Wz + bz)
    r_t = T.nnet.sigmoid(u_l_tm1_dot_Ur + u_lm1_t_dot_Wr + br)

    # hidden state dot prod:
    r_t_u_tm1_dot_U = convolution(uh_op[None, :], h_l_tm1[None, :] * r_t[None, :])[0]

    # hidden state evolution:
    h_l_t = (1 - z_t) * h_l_tm1 + z_t * T.tanh(r_t_u_tm1_dot_U + u_lm1_t_dot_W0)

    return h_l_t


def kpz_layer(operators, h_l_tm1, h_lm1_t):
    """

    :param operators: shape ()
        One layer's concatenated weights of shape
        (n_rec, 2 * n_rec + 1). Last col is w vector
        Weights W, U are vertically stacked, weights for h, z, r horizontally
    :param h_l_tm1: shape (n_hidden, )
    :param h_lm1_t:
    :return:
    """
    n_rec = T.shape(h_l_tm1)[0]
    W = operators[:, 0:n_rec]
    U = operators[:, n_rec:2 * n_rec]
    w = operators[:, -1]  # shape (n_rec, )

    growth_term = h_l_tm1 * h_l_tm1 * T.sum(w) - 2 * h_l_tm1 * T.dot(w, h_l_tm1) + T.dot(w, h_l_tm1 * h_l_tm1)

    h_l_t = T.dot(h_l_tm1, W) + T.dot(h_lm1_t, U) + growth_term

    return h_l_t


def tanh_readin(operators, in_):
    """Tanh activation layer.
    :param operators: list of [weight, bias] with shapes (n_visible, n_hidden)
        and (n_hidden, )
    :param in_: shape (timesteps, n_visible)
    :return: shape (timesteps, n_hidden)
    """
    weight = operators[0]
    bias = operators[1]
    out = tanh(T.dot(in_, weight) + bias)

    return out


def tanh_tip_readin(operators, in_):
    """

    :param operators: shapes (1, n_visible + n_hidden - 1) or scalar
    :param in_: shape (timesteps, n_visible)
    :return out: shape (timesteps, n_hidden)
    """

    weight_vec = operators[0]
    bias = operators[1]
    dotprod = convolution(weight_vec[None, :], in_[:, None, :])[:, 0, :]  # get shape (timesteps, n_hidden)
    out = tanh(dotprod + bias)

    return out


def rbm_readout(operators, v_in, h, external):
    #TODO: adjust Gibbs sampling size?

    h_L = h[-1]  # shape (timesteps, n_hidden)
    Mmat = operators[0]  # shape (n_visible, n_hidden_readout)
    Whidvis = operators[1]  # shape (n_hidden, n_visible)
    Whidhid = operators[2]  # shape (n_hidden, n_hidden_readout)
    bvis = operators[3]
    bhid = operators[4]

    bv_t = bvis + T.dot(h_L, Whidvis)
    bh_t = bhid + T.dot(h_L, Whidhid)

    def gibbs_step(v):
        mean_h = T.nnet.sigmoid(T.dot(v, Mmat) + bh_t)  # shape (timesteps, n_hidden_readout)
        h = rng.binomial(size=mean_h.shape, n=1, p=mean_h,
                         dtype=theano.config.floatX)
        mean_v = T.nnet.sigmoid(T.dot(h, Mmat.T) + bv_t)
        v = rng.binomial(size=mean_v.shape, n=1, p=mean_v,
                         dtype=theano.config.floatX)
        return mean_v, v

    chain, updates = theano.scan(lambda v: gibbs_step(v)[1],
                                 outputs_info=[T.unbroadcast(v_in, 0)],
                                 n_steps=25,
                                 name='gibbs sample scan')
    v_sample = chain[-1]

    v_in_c = T.clip(v_in, eps, 1 - eps)
    #v_sample_c = T.clip(v_sample, eps, 1 - eps)

    mean_v = gibbs_step(v_sample)[0]
    mean_v_c = T.clip(mean_v, eps, 1 - eps)
    #monitor = -T.xlogx.xlogy0(v, mean_v) - T.xlogx.xlogy0(1 - v, 1 - mean_v)
    monitor = crossent(mean_v_c[:-1], v_in_c[1:])
    monitor = monitor.mean()

    def free_energy(v):
        return -(v * bv_t[:-1]).sum() - T.log(1 + T.exp(T.dot(v, Mmat) + bh_t[:-1])).sum()
    #TODO: do the slices bv_t[:-1] etc *really* match both v_in and v_sample??

    cost = (free_energy(v_in[1:]) - free_energy(v_sample[:-1])) / v_in.shape[0]
    # Note that this is not the *actual* cost, but just the difference of
    # free energies used in computing the gradients!! The monitor defined
    # below would correspond to the (unnormalized) negative log probability.
    #monitor = free_energy(v_in[1:]) / v_in.shape[0]

    return v_sample, cost, monitor, updates


def rbm_tip_readout(operators, v_in, h, external):
    """

    :param operators:
    :param v_in:
    :param h: shape (depth, timesteps, n_hidden)
    :param external:
    :return:
    """

    mvec_op = operators[0]
    whidvis_op = operators[1]  # shape (n_layers, n_vis + n_hid - 1)
    whidhid_op = operators[2]  # shape (n_layers, n_hid + n_hid_ro - 1)
    bvis_op = operators[3] + external  # add external inhomogeneous field
    bhid_op = operators[4]

    # compute dynamic biases (old):
    #bv_t = bvis_op + convolution(whidvis_op[None, :], h_L[:, None, :])[:, 0, :]
    #bh_t = bhid_op + convolution(whidhid_op[None, :], h_L[:, None, :])[:, 0, :]

    # compute dynamic biases with all layers:
    bv_biases, _ = theano.scan(lambda w_l, h_l: convolution(w_l[None, :], h_l[:, None, :])[:, 0, :],
                sequences=[whidvis_op, h],
                name='all RBM bv convolutions')  # shape (depth, timesteps, n_vis)
    bh_biases, _ = theano.scan(lambda w_l, h_l: convolution(w_l[None, :], h_l[:, None, :])[:, 0, :],
                sequences=[whidhid_op, h],
                name='all RBM bh convolutions')  # shape (depth, timesteps, n_hid)

    bv_t = bvis_op + T.sum(bv_biases, axis=0)
    bh_t = bhid_op + T.sum(bh_biases, axis=0)

    mvec_cT = mvec_op[::-1]
    def gibbs_step(v_in, bv, bh):  # input shape (timesteps, n_visible); no scan slicing!
        v_dot_W0 = convolution(mvec_op[None, :], v_in[:, None, :])[:, 0, :]  # get shape (timesteps, n_hidden)
        mean_h = T.nnet.sigmoid(v_dot_W0 + bh)
        h = rng.binomial(size=mean_h.shape, n=1, p=mean_h,
                         dtype=theano.config.floatX)  # shape (timesteps, n_hidden)
        h_dot_W0T = convolution(mvec_cT[None, :], h[:, None, :])[:, 0, :]
        mean_v = T.nnet.sigmoid(h_dot_W0T + bv)
        v_in = rng.binomial(size=mean_v.shape, n=1, p=mean_v,
                            dtype=theano.config.floatX)  # shape (timesteps, n_visible)
        return mean_v, v_in  # shape (timesteps, n_visible)

    chain, updates_rbm = theano.scan(lambda v, bv, bh: gibbs_step(v, bv, bh)[1],
                                     outputs_info=[T.unbroadcast(v_in, 0)],
                                     non_sequences=[bv_t, bh_t],
                                     n_steps=25,
                                     name='RBM sampling scan')
    v_sample = chain[-1]  # shape (timesteps, n_visible)

    #mean_v = gibbs_step(v_sample, bv_t, bh_t)[0][:-1]  # shape (timesteps - 1, n_visible)
    #monitor = -T.xlogx.xlogy0(v_in[1:], mean_v) - T.xlogx.xlogy0(1 - v_in[1:], 1 - mean_v)  # note sign
    #monitor = monitor.sum() / v_in.shape[0]

    v_in_c = T.clip(v_in, eps, 1 - eps)
    #v_sample_c = T.clip(v_sample, eps, 1 - eps)

    mean_v = gibbs_step(v_sample, bv_t, bh_t)[0]
    mean_v_c = T.clip(mean_v, eps, 1 - eps)
    #monitor = -T.xlogx.xlogy0(v, mean_v) - T.xlogx.xlogy0(1 - v, 1 - mean_v)
    monitor = crossent(mean_v_c[:-1], v_in_c[1:])
    monitor = monitor.mean()

    def free_energy(v):  # input shape (timesteps, n_visible)
        v_dot_W0 = convolution(mvec_op[None, :], v[:, None, :])[:, 0, :]
        # TODO: adjusted eps to 1E-6 and added eps in the log below (got nans; not sure if this fixes it)
        return -(v * bv_t[:-1]).sum() - T.log(1 + eps + T.exp(v_dot_W0 + bh_t[:-1])).sum()
    # v_t predicts v_sample_tp1:
    cost = (free_energy(v_in[1:]) - free_energy(v_sample[:-1])) / v_in.shape[0]

    return v_sample, cost, monitor, updates_rbm


def sigmoid_readout(operators, v_in, h_L, external):
    """Sigmoid readout layer. Cost is the binary crossentropy and
    monitor is RMSE.
    :param operators: list of [weight, bias] with shapes (n_hidden, n_visible)
        and (n_visible, )
    :param h_L: shape (timesteps, n_hidden)
    :return: shape (timesteps, n_visible)
    """
    weight = operators[0]
    bias = operators[1]
    v_pred = sigmoid(T.dot(h_L, weight) + bias)  # broadcastable bias??
    v_pred_c = T.clip(v_pred, 1.0e-7, 1.0 - 1.0e-7)
    v_in_c = T.clip(v_in, 1.0e-7, 1.0 - 1.0e-7)

    # Sample is just rounded to nearest integer:
    v_sample = T.round(v_pred)
    v_sample_c = T.clip(v_sample, eps, 1.0 - eps)

    # Cost:
    #cost = 1000 * ((v_pred[:-1] - v_in[1:]) ** 2).mean()
    #cost = -T.xlogx.xlogy0(v_in_c[1:], v_pred_c[:-1]) - \
    #       T.xlogx.xlogy0(1 - v_in_c[1:], 1 - v_pred_c[:-1])
    cost = crossent(v_pred_c[:-1], v_in_c[1:]) #TODO: v_sample_c !!!
    cost = cost.mean()

    # Monitor:
    #monitor = -T.xlogx.xlogy0(v_in_c[1:], v_sample_c[:-1]) - \
    #          T.xlogx.xlogy0(1 - v_in_c[1:], 1 - v_sample_c[:-1])
    monitor = crossent(v_sample_c[:-1], v_in_c[1:])
    monitor = monitor.mean()

    return v_sample, cost, monitor, None


def softmax_readout(operators, v_in, h_L, external):
    """Softmax readout layer. Cost is the binary crossentropy and
    monitor is RMSE.
    :param operators: list of [weight, bias] with shapes (n_hidden, n_visible)
        and (n_visible, )
    :param h_L: shape (timesteps, n_hidden)
    :return: shape (timesteps, n_visible)
    """
    weight = operators[0]
    bias = operators[1]

    v_pred = softmax(T.dot(h_L, weight) + bias)  # broadcastable bias??
    v_pred_c = T.clip(v_pred, 1.0e-7, 1.0 - 1.0e-7)
    v_in_c = T.clip(v_in, 1.0e-7, 1.0 - 1.0e-7)

    # Sampled value is just the argmax of softmax:
    v_sample = rng.multinomial(pvals=v_pred, dtype=theano.config.floatX)
    v_sample_c = T.clip(v_sample, eps, 1.0 - eps)

    # Cost:
    #cost = 1000 * ((v_pred[:-1] - v_in[1:]) ** 2).mean()
    #cost = -T.xlogx.xlogy0(v_in_c[1:], v_pred_c[:-1]) - \
    #       T.xlogx.xlogy0(1 - v_in_c[1:], 1 - v_pred_c[:-1])
    cost = crossent(v_pred_c[:-1], v_in_c[1:])
    cost = cost.mean()

    # Monitor:
    #monitor = -T.xlogx.xlogy0(v_in_c[1:], v_sample_c[:-1]) - \
    #          T.xlogx.xlogy0(1 - v_in_c[1:], 1 - v_sample_c[:-1])
    #TODO: changed monitor to v_pred_c!!!
    monitor = crossent(v_pred_c[:-1], v_in_c[1:])
    monitor = monitor.mean()

    return v_sample, cost, monitor, None

def sigmoid_readout_old(operators, v_in, h_L, g):
    """Sigmoid readout layer. Cost is the binary crossentropy and
    monitor is RMSE.
    :param params: list of [weight, bias] with shapes (n_hidden, n_visible)
        and (n_visible, )
    :param h_L: shape (timesteps, n_visible)
    :return: shape (timesteps, n_hidden)
    """
    weight = operators[0]
    bias = operators[1]
    v_pred = g(T.dot(h_L, weight) + bias)  # broadcastable bias??
    v_pred_c = T.clip(v_pred, 1.0e-7, 1.0 - 1.0e-7)
    v_in_c = T.clip(v_in, 1.0e-7, 1.0 - 1.0e-7)

    # Cost:
    cost = -T.xlogx.xlogy0(v_in_c[1:], v_pred_c[:-1]) - \
           T.xlogx.xlogy0(1 - v_in_c[1:], 1 - v_pred_c[:-1])
    cost = cost.sum() / v_in.shape[0]

    # Sample is just rounded to nearest integer:
    v_sample = T.round(v_pred)
    v_sample_c = T.clip(v_sample, 1.0e-7, 1.0 - 1.0e-7)

    # Monitor (needs to return something... for now):
    monitor = -T.xlogx.xlogy0(v_in_c[1:], v_sample_c[:-1]) - \
              T.xlogx.xlogy0(1 - v_in_c[1:], 1 - v_sample_c[:-1])
    monitor = monitor.sum() / v_in.shape[0]

    return v_sample, cost, monitor, None

