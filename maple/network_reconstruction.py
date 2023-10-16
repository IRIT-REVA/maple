#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pathlib
import sys

from itertools import cycle
from matplotlib import pylab as pl
from sklearn import metrics

with pathlib.Path(__file__).parent.parent.parent as p1:
    sys.path.append(str(p1))
    sys.path.append(str(p1/"spyking_circus"))
from maple.sorted_reader.from_circus import Results as Sorted
from maple.colors import extended_colortable


def parce_args():
    parser = argparse.ArgumentParser(
        description="Just an example",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("-f", "--filename_raw", help="raw data file")
    args = parser.parse_args()
    config = vars(args)
    #    print(config)

    return config


def set_input(fname_raw):
    config = parce_args()

    if config['filename_raw'] is None:
        if fname_raw is not None:
            print(f"Warning: using hardcoded data file: {fname_raw}")
            return fname_raw
        else:
            raise Exception("No input file provided!")
    else:
        return config['filename_raw']


def reconstruct_neuron(spks, nid, n, m, delay):

    print(f"Reconstructing incoming connectivity for neuron {nid}")
    spk = spks[nid]
    spksd = [s + delay for s in spks]

    # Computing Interspike Intervals:
    isis = np.diff(spk)
    nisis = isis.size

    # Computing Cross-spike Intervals for selected neuron:
    events = [[[] for _ in range(n)] for _ in range(nisis)]
#    t = [0] * n
    for ti in range(nisis):
        for j in range(n):
            if j != nid:
                sj = spksd[j]
                events[ti][j] = \
                    (sj[(sj > spk[ti]) & (sj < spk[ti + 1])] - spk[ti]).tolist()

    # Determining maximum number of Cross-spike Intervals per event:
    k_events = np.asarray([max([len(events[ti][j]) for j in range(n)])
                           for ti in range(nisis)])
    maxke = np.max(k_events)
    import math
    from scipy.spatial.distance import euclidean as ssde
    # Constructing the events:
    ises = np.zeros((maxke*n, nisis))
    for ti in range(nisis):
        for j in range(n):
            a = events[ti][j]
            for i in range(len(a)):
                ises[i*n + j, ti] = a[i]
    d = np.vstack((ises, isis)).T
    del events

    # Determing the reference event:
    chsz = 1000  # chunksize
    center = np.concatenate(
        [np.mean(
            metrics.pairwise.euclidean_distances(
                d[i:(i+chsz if i+chsz <= nisis else nisis), :], d),
            axis=1
         )
         for i in range(0, (nisis//chsz + 1)*chsz, chsz) if i < nisis]
    )
    center_index = np.argmin(center)

    # Determining order in an increasing manner with respect to
    # the reference event:
    non_ranked = metrics.pairwise.euclidean_distances(
        d[center_index:center_index+1, :], d
    )[0]
    del d

    closest_index = np.argsort(non_ranked)
    k_vector = k_events[closest_index]
    if m > len(k_vector):
        m_ = np.where(k_vector > 0)[0]
        if len(m_):
            m = m_[-1]
            print(f"Warning: reducing m to {m} because m > len(k_vector)")
    elif k_vector[m] == 0:
        m_ = np.where(k_vector[m:] > 0)[0]
        if len(m_):
            if m_[0] > 0:
                m += m_[0]
                print(f"Warning: increasing m to {m}")
        else:
            m_ = np.where(k_vector[:m] > 0)[0]
            if len(m_):
                m = m_[-1]
                print(f"Warning: reducing m to {m}")
    k = k_vector[m]

    # Constructing the system of equations:
    c = ises[:, center_index].reshape(maxke*n, 1)
    w = ises - c * np.ones((1, nisis))
    y = isis.reshape((1, nisis)) - \
        isis[center_index] * np.ones((1, nisis))
    x = np.copy(w)
    del isis
    del ises

    # Ordering the system of equations according to distance with respect to
    # the reference event:
    y = y[0, closest_index]
    y = y.reshape((1, nisis))
    x = x[:, closest_index]

    # Selecting up to 'm' events to solve the system of equations:
    y = y[:, 1:m + 1]
    x = x[:k*n, 1:m + 1]

    # Solving the system of equations:
    print(f"Employing L2 norm optimization for neuron {nid}")
    g = np.dot(y, np.linalg.pinv(x))

    if g.size == 0:
        print('')
    # Selecting only the first firing profile as connectivity proxy:
    G = g.reshape((k, -1))

    return G[0, :]


def reconstruct(spks, nids=None, m=1000):
    """
    reconstruct(spikes, M) reconstructs the incoming synaptic
    links from recorded spike trains alone.
    
    Parameters:
    ------------------
    spikes: list of spike trqins, per neuron.
    m:   number of events employed for reconstruction.
            
    Output
    ----------
    conn_float: recovered connectivity matrix
                    
    Example:
    ------------------
    reconstruction(spikes, 500) reconstructs the incoming links of all
    neurons using 500 recorded events.
    
    """
    
    n = len(spks)
    delay = 0

    conn_float = np.zeros((n, n), dtype=float)
    if nids is None:
        nids = range(n)

    for nid in nids:
        conn_float[nid, :] = reconstruct_neuron(spks, nid, n, m, delay)

    return conn_float


# =============================================================================
def to_bool(conn_float, thresh):

    nn = conn_float.shape[1]
    conn = np.zeros((nn, nn))
    # number of inclming excitatory connections:
    num_exc_in = np.empty(nn, dtype=int)
    # number of inclming inhibitory connections:
    num_inh_in = np.empty(nn, dtype=int)
    for nind in range(nn): #len(spikes)):
        inds_exc_in = np.where(conn_float[nind, :] > thresh)[0]
        conn[nind, inds_exc_in] = 1
        num_exc_in[nind] = len(inds_exc_in)
        inds_inh_in = np.where(conn_float[nind, :] < -thresh)[0]
        conn[nind, inds_inh_in] = -1
        num_inh_in[nind] = len(inds_inh_in)

    # number of inclming excitatory connections:
    num_exc_out = np.empty(nn, dtype=int)
    # number of inclming inhibitory connections:
    num_inh_out = np.empty(nn, dtype=int)
    for nind in range(nn): #len(spikes)):
        inds_exc_out = np.where(conn[:, nind] == 1)[0]
        num_exc_out[nind] = len(inds_exc_out)
        inds_inh_out = np.where(conn[:, nind] == -1)[0]
        num_inh_out[nind] = len(inds_inh_out)

    return conn, num_exc_in, num_inh_in, num_exc_out, num_inh_out


# =============================================================================
def plot_matrix(conn, num_exc_in, num_inh_in, num_exc_out, num_inh_out, title=None):

    nn = conn.shape[0]
    fig, ax = pl.subplots(nrows=2, ncols=2) #, sharex=True, sharey=True)
    ax[0, 0].set_title("Connectivity")
    ax[0, 0].pcolormesh(np.arange(0, nn), np.arange(0, nn), conn)
    ax[0, 0].set_ylabel('Neuron ID')
    ax[0, 1].scatter(num_exc_in, np.arange(0, nn), s=4)
    ax[0, 1].scatter(-num_inh_in, np.arange(0, nn), s=4)
    ax[0, 1].set_xlim([-nn, nn])
#    ax[0, 1].line(np.arange(0, nn), -num_inh_in s=4)
    ax[0, 1].set_xlabel("Number of in-connections")
    ax[1, 0].scatter(np.arange(0, nn), num_exc_out, s=4)
    ax[1, 0].scatter(np.arange(0, nn), -num_inh_out, s=4)
    ax[1, 0].set_ylim([-nn, nn])
#    ax[1, 0].scatter(np.arange(0, nn), num_exc_out/(num_inh_out + num_exc_out), s=4)
#    ax[1, 0].set_ylabel("Fract. of out conn. excitatory.")
    ax[1, 0].set_xlabel('Neuron ID')
    ax[1, 1].scatter(num_exc_in, num_exc_out, s=4)
    ax[1, 1].scatter(-num_inh_in, -num_inh_out, s=4)
    ax[1, 1].set_xlabel('Num in-connections')
    ax[1, 1].set_ylabel('Num out-connections')
    ax[1, 1].set_xlim([-nn, nn])
    ax[1, 1].set_ylim([-nn, nn])
    ax[1, 1].set_aspect('equal')
    ax[1, 1].grid('on')

    if title is not None:
        plt.suptitle(title)
    pl.show()


# =============================================================================
def plot_matrix1(conn,
                 num_exc_in,
                 num_inh_in,
                 num_exc_out,
                 num_inh_out,
                 title=None):

    nn = conn.shape[0]
    fig1, ax = pl.subplots(nrows=1, ncols=2)
    ax[0].set_xlabel('Number of in-connections')
    ax[0].set_ylabel('Fraction of excitatory in-conn')
    ax[0].scatter(num_exc_in + num_inh_in,
                  num_exc_in/(num_exc_in + num_inh_in),
                  s=4)
    ax[0].set_xlim([0, nn])
    ax[0].set_ylim([0, 1])
    ax[1].set_xlabel('Number of out-connections')
    ax[1].set_ylabel('Fraction of excitatory out-conn')
    ax[1].scatter(num_exc_out + num_inh_out,
                  num_exc_out/(num_exc_out + num_inh_out),
                  s=4)
    ax[1].set_xlim([0, nn])
    ax[1].set_ylim([0, 1])
    pl.show()


# =============================================================================
def plot_matrix_dyn1(dn,numse_exc_in,
                     numse_inh_in,
                     numse_exc_out,
                     numse_inh_out):

    maxnn = np.ceil((np.max(np.max(numse_exc_in)) +
                     np.max(np.max(numse_inh_in)))/100) * 100
    for k in range(numse_exc_in.shape[0] - 1):
        cc = cycle(extended_colortable)
        colors = [next(cc) for _ in range(numse_exc_in.shape[1])]
        fig, ax = pl.subplots()
        fig.suptitle(f"{dn[k]} -> {dn[k+1]}")
        ax.set_xlabel('Number of in-connections')
        x0 = numse_exc_in[k,:] + numse_inh_in[k,:]
        x1 = numse_exc_in[k+1,:] + numse_inh_in[k+1,:]
        y0 = numse_exc_in[k,:]/(numse_exc_in[k,:] + numse_inh_in[k,:])
        y1 = numse_exc_in[k+1,:]/(numse_exc_in[k+1,:] + numse_inh_in[k+1,:])
        ax.quiver(x0, y0, x1 - x0, y1 - y0, angles='xy',
                  scale_units='xy', scale=1,
                  width=0.002, lw=0.1, color=colors)
        ax.set_xlim([0, maxnn])
        ax.set_ylabel('Fraction of excitatory in')
        ax.set_ylim([0, 1])
    pl.show()


# =============================================================================
def plot_matrix_dyn(numse_exc_in,
                    numse_inh_in,
                    numse_exc_out,
                    numse_inh_out):

    fig = plt.figure()
    ax0 = fig.add_subplot(1, 1, 1, projection='3d')
    ax0.set_xlabel('Number of in-connections')
    ax0.set_ylabel('Fraction of excitatory in-conn')
    for e in range(numse_exc_in.shape[1]):
        ax0.plot(numse_exc_in[:,e] + numse_inh_in[:,e],
                 numse_exc_in[:,e]/(numse_exc_in[:,e] + numse_inh_in[:,e]),
                 range(numse_exc_in.shape[0]),
                 marker='o', markersize=4)
    ax0.set_ylim([0, 1])

    pl.show()


# =============================================================================
def save(conn_float, conn, filename_raw, thresh, num_isis):
    path = "/".join(filename_raw.split(sep='/')[:-1])
    np.savetxt(f"{path}/conn_{thresh}_{num_isis}.txt", conn, fmt='%d')
    np.savetxt(f"{path}/conn_float_{num_isis}.txt", conn_float, fmt='%f')


# =============================================================================
def load(filename_raw, thresh, num_isis):

    path = "/".join(filename_raw.split(sep='/')[:-1])
    conn = np.loadtxt(f"{path}/conn_{thresh}_{num_isis}.txt", dtype=int)
    conn_float = np.loadtxt(f"{path}/conn_float_{num_isis}.txt", dtype=float)
    conn1, num_exc_in, num_inh_in, num_exc_out, num_inh_out = \
        to_bool(conn_float, thresh)
    assert np.array_equal(conn, conn1)

    return conn_float, conn, num_exc_in, num_inh_in, num_exc_out, num_inh_out


# =============================================================================
def create(fname_raw=None):

    filename_raw = set_input(fname_raw)

    spikes_dict = Sorted(filename_raw, ['spike_times']).spike_times
    spikes = [[] for _ in range(len(spikes_dict))]
    for key, val in spikes_dict.items():
        spikes[int(key)] = val
    nn = len(spikes)
    print(f"{nn} neurons are detected in {filename_raw}")
    num_isis = 1000
    #    res1 = reconstruct_neuron(spikes, 5, nn, num_ISIs, 0)

    nids = None  # range(168,169)
    conn_float = reconstruct(spikes, nids, num_isis)

    thresh_con = 0.01
    conn, \
    num_exc_in, \
    num_inh_in, \
    num_exc_out, \
    num_inh_out = \
        to_bool(conn_float, thresh_con)

    conn_coef = (np.sum(num_exc_in) + np.sum(num_inh_in)) / (nn * nn)
    print(f"Num exc: {np.sum(num_exc_in)}")
    print(f"Num inh: {np.sum(num_inh_in)}")
    print(f"Connectivity coef: {conn_coef}")

    plot_matrix(conn, num_exc_in, num_inh_in, num_exc_out, num_inh_out)

    save(conn_float, conn, filename_raw, thresh_con, num_isis)
    #    save(conn, f"{path}/conn.txt")

    print('')


# =============================================================================
def from_file(fname_raw=None):

    filename_raw = set_input(fname_raw)
    thresh = 0.01
    num_isis = 1000
    conn_float, conn, num_exc_in, num_inh_in, num_exc_out, num_inh_out = \
        load(filename_raw, thresh, num_isis)
    plot_matrix(conn, num_exc_in, num_inh_in, num_exc_out, num_inh_out)
    plot_matrix1(conn, num_exc_in, num_inh_in, num_exc_out, num_inh_out)
    print('')