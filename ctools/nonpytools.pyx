__author__ = 'Heikki Arponen'

import numpy as np
import pandas as pd
cimport cython
cimport numpy as np

from libc.stdint cimport uint32_t, int32_t, uint64_t, int64_t
from libc.math cimport pow, M_PI, fabs
#from libc.math cimport fabs
from libc.math cimport pow

#ctypedef float     real_t
ctypedef np.float64_t dtype_t
#ctypedef uint32_t  uint_t
ctypedef int32_t   int_t
ctypedef np.int32_t uint_t

cdef extern from "complex.h":
    double complex cexp(double complex)
    double cabs(double complex)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def max_eig(vec):
    """Largest abs eig of a translation invariant and periodic
    matrix constructed of a vector vec.

    :param vec: numpy vector of ndim=1
    """

    cdef:
        int N = len(vec)
        double complex val, val_pos, val_neg
        size_t k, l
        double abs_val
        double c_max = 0
        double complex[:] cvec = vec.astype(complex)

    for k in range(N):
        val = 0
        for l in range(0, N):
            val_pos = (1 - l / N) * cexp(2 * M_PI * 1j * k * l / N) * cvec[l]
            val += val_pos
        abs_val = cabs(val)
        if abs_val > c_max:
            c_max = abs_val

    return c_max

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def pianoroll_from_statemat(np.ndarray[int64_t, ndim=3] statemat):

    cdef:
        int64_t song_length = statemat[:, :, 0].shape[0]
        int64_t nnotes = statemat[:, :, 0].shape[1]

        np.ndarray[int64_t, ndim=2] strikes = statemat[:, :, 1].astype(np.int64)
        np.ndarray[int64_t, ndim=2] holds = statemat[:, :, 0].astype(np.int64)

        np.ndarray[int64_t, ndim=2] pianoroll = np.zeros((2 * song_length, nnotes), dtype=np.int64)

        uint_t n, t, t_orig, next_strike, current_hold

    for n in range(nnotes):
        for t in range(2 * song_length - 2):
            t_orig = t // 2
            next_strike = strikes[t_orig + 1, n]
            current_hold = holds[t_orig, n]
            if t % 2 == 0 and current_hold == 1:
                pianoroll[t, n] = 1
            if t % 2 == 1 and current_hold == 1 and next_strike == 0:
                pianoroll[t, n] = 1

    return pianoroll

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def statemat_from_pianoroll(np.ndarray[int64_t, ndim=2] pianoroll):

    assert pianoroll.shape[0] % 2 == 0, 'Error: not an even length piano roll!'

    cdef:
        int64_t song_length = pianoroll.shape[0]
        int64_t nnotes = pianoroll.shape[1]

        np.ndarray[int64_t, ndim=2] strikes = np.zeros((song_length / 2, nnotes), dtype=np.int64)
        np.ndarray[int64_t, ndim=2] holds = np.zeros((song_length / 2, nnotes), dtype=np.int64)

        uint_t n, t, change, value

    for n in range(nnotes):
        # first note:
        value = pianoroll[0, n]
        if value == 1:
            holds[0, n] = 1
            strikes[0, n] = 1
        for t in range(1, song_length / 2):
            change = pianoroll[2 * t, n] - pianoroll[2 * t - 1, n]
            value = pianoroll[2 * t, n]
            if change == 0 and value == 1:
                holds[t, n] = 1
            if change == 1 and value == 1:
                holds[t, n] = 1
                strikes[t, n] = 1

    return np.concatenate((holds[:, :, None], strikes[:, :, None]), axis=-1).astype(np.int64)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def corrfun(np.ndarray[dtype_t, ndim=1] leftarr, np.ndarray[dtype_t, ndim=1] rightarr, uint_t ran):

    """Helper correlation function. Returns the array defined as
    $$
    C(t) = \sum\limits_{t' = min(0, -t)}^{T - max(0, t)} L(t' + t) R(t'),
    $$
    for $ -ran <= t <= ran$ and where $L=$ leftarr, $R=$ rightarr and $T=$ len(L) = len(R).

    :param leftarr: numpy.float64
    :param rightarr: numpy.float64
    :param ran: int
    :return:
    """
    assert(len(leftarr) == len(rightarr))

    cdef:
        uint_t size = len(leftarr)
        np.ndarray[dtype_t, ndim=1] corr_array = np.zeros(2*ran + 1, dtype = np.float64)
        uint_t n, m
        dtype_t temp

    #right hand side and zero:
    for n in range(ran + 1):
        temp = 0
        for m in range(size - n - 1):
            temp = temp + leftarr[m + n] * rightarr[m]
        corr_array[ran + n] = temp
    #left hand side:
    for n in range(1, ran + 1):
        temp = 0
        for m in range(n, size - 1):
            temp = temp + leftarr[m - n] * rightarr[m]
        corr_array[ran - n] = temp

    return corr_array / size

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def crosscorrelation(np.ndarray[dtype_t, ndim=2] data, uint_t ran):

    """Helper correlation function. Returns the array defined as
    $$
    C^{ij}(t) = \sum\limits_{t' = min(0, -t)}^{T - max(0, t)} L^i(t' + t) R^j(t'),
    $$
    for $ -ran <= t <= ran$ and where $L=$ leftarr, $R=$ rightarr and $T=$ len(L) = len(R).

    :param data: numpy.float64
    :param ran: int
    :return:
    """

    cdef:
        uint_t T = data.shape[0]
        uint_t N = data.shape[1]
        np.ndarray[dtype_t, ndim=3] corr_array = np.zeros((2*ran + 1, N, N), dtype = np.float64)
        np.ndarray[dtype_t, ndim=1] means = np.zeros(N, dtype = np.float64)
        uint_t n, m, t, s
        dtype_t temp

    assert data.shape[0] > 1, "Error: array time wise length too low!"

    # compute means:
    for i in range(N):
        temp = 0.
        for t in range(T):
            temp += data[t, i]
        temp /= T
        means[i] = temp

    # compute correlation:
    for s in range(2 * ran + 1):  # C(s, ..)
        for i in range(N):
            for j in range(N):
                temp = 0.
                lower = ran - s if ran - s > 0 else 0
                upper = T - 1 + ran - s if ran - s < 0 else T - 1
                for t in range(lower, upper):
                    temp += (data[t + s - ran, i] - means[i]) * (data[t, j] - means[j])
                temp /= (T - 1)
                corr_array[s, i, j] = temp

    # normalization:
    norm = np.trace(corr_array[ran])

    return corr_array / norm


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(False)  # set to True for speedup and no checks
def random_split(np.ndarray[dtype_t, ndim=2] data,
                           uint_t window,
                           uint_t samples):

    """ Sample an array of shape (T, N) into an array
    of shape (samples, window, N) according to a random
    index.
    """

    cdef:
        uint_t datalen = data.shape[0]
        uint_t datadim = data.shape[1]
        np.ndarray[int_t, ndim=1] index = np.random.randint(0, datalen - window, samples).astype(np.int32)
        uint_t t, i, j
        np.ndarray[dtype_t, ndim=3] datatensor = np.zeros((samples, window, datadim), dtype=np.float64)


    for i in range(samples):
        for t in range(window):
            for j in range(datadim):
                datatensor[i, t, j] = data[index[i] + t, j]

    return datatensor
