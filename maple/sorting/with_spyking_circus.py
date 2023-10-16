import h5py
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from pathlib import Path

from circus.shared.files import load_data
from circus.shared.parser import CircusParser
from circus.shared.probes import get_nodes_and_edges

figure_width = 15


# =============================================================================
def check_time_args(t_start, t_stop):
    """Checks for 't_start', 't_stop' values.
    """

    wav = "Wrong argument value:"
    assert t_start >= 0., \
           f"{wav} t_start = {t_start}. Please set nonnegative starting time."
    assert t_stop > t_start, \
           f"{wav}  t_stop = {t_stop}. Please set it larger than t_start."


# =============================================================================
def check_electrode_arg(electrodes, N_e):
    """Checks for correctness of electrode indexes.
    """

    wav = "Wrong argument value:"
    if np.iterable(electrodes):
        assert len(electrodes), \
            f"{wav}  Please set the electrodes you are interested in."
        assert min(electrodes) >= 0, \
            "Electrodes should be indexed with nonnegative integers."
    else:
        assert electrodes >= 0, \
            "Electrodes should be indexed with nonnegative integers."
        electrodes = [electrodes]

    assert max(electrodes) < N_e, \
        f"{wav}  The data analized used only {N_e} electrodes."

    return electrodes


def plot_peaks(file_name, t_start, t_stop, electrodes):
    """Shows putative peaks along with the original data in time domain.
    """

    check_time_args(t_start, t_stop)

    params = CircusParser(file_name)
    data_file = params.get_data_file()
    data_file.open()
    N_e = params.getint('data', 'N_e')
    sampling_rate = params.rate

    do_temporal_whitening = params.getboolean('whitening', 'temporal')
    do_spatial_whitening = params.getboolean('whitening', 'spatial')
    nodes, edges = get_nodes_and_edges(params)
    chunk_size = np.int64((t_stop - t_start) * sampling_rate)
    padding = np.int64(t_start * sampling_rate), \
              np.int64(t_start * sampling_rate)

    spatial_whitening = load_data(params, 'spatial_whitening') \
        if do_spatial_whitening else None
    temporal_whitening = load_data(params, 'temporal_whitening') \
        if do_temporal_whitening else None

    thresholds = load_data(params, 'thresholds')
    data = data_file.get_data(0, chunk_size, padding=padding, nodes=nodes)
    data_file.close()

    electrodes = check_electrode_arg(electrodes, N_e)

    if do_spatial_whitening:
        data = np.dot(data[0], spatial_whitening)
    if do_temporal_whitening:
        data = sp.ndimage.filters.convolve1d(data, temporal_whitening,
                                             axis=0, mode='constant')

    peaks = {}
    for i in range(N_e):
        peaks[i] = sp.signal.find_peaks(-data[:, i], height=thresholds[i])[0]

    idx = electrodes
    n_elec = len(idx)

    print(f"Horizontal dotted lines: signal thresholds.")
    print(f"Spike instances are higlighted with red triangles.")

    plt.figure(figsize=[figure_width, n_elec*3])
    for count, i in enumerate(idx):
        plt.subplot(n_elec, 1, count + 1)
        if count != n_elec - 1:
            plt.setp(plt.gca(), xticks=[])
        else:
            plt.xlabel('Time [ms]')
        scale_ms = sampling_rate / 1000
        x = np.arange(padding[0], padding[1] + chunk_size) / scale_ms
        y = data[:, i]
        ylim0 = np.abs(np.min([np.min([y]), -2 * thresholds[i]]))
        ylim1 = np.abs(np.max([np.max([y]),  2 * thresholds[i]]))
        plt.ylim(-ylim0 * 1.2, ylim1 * 1.1)
        plt.plot(x, y, lw=0.1, c=[0.5, 0.5, 0.5])
        plt.scatter((padding[0] + peaks[i]) / scale_ms,
                    [-ylim0 * 1.05]*len(peaks[i]),
                    marker='^', s=20, c='r')
        xmin, xmax = plt.xlim()
        plt.xlim(xmin, xmax)
        plt.plot([xmin, xmax],
                 [-thresholds[i], -thresholds[i]], lw=0.15, ls='--', c='b')
        plt.plot([xmin, xmax],
                 [thresholds[i], thresholds[i]], lw=0.15, ls='--', c='b')
        plt.title(f'Electrode {i}: detected {len(peaks[i])} spikes.')

    plt.tight_layout()
    plt.show()


# =============================================================================
def plot_fits(file_name, t_start, t_stop, electrodes, templates=None):
    """Shows reconstructed signal and the original data in time domain.
    """

    check_time_args(t_start, t_stop)

    params = CircusParser(file_name)
    data_file = params.get_data_file()
    data_file.open()
    _ = data_file.analyze(int(params.rate))
    N_e = params.getint('data', 'N_e')

    N_t = params.getint('detection', 'N_t')
    sampling_rate = int(params.rate)
    do_temporal_whitening = params.getboolean('whitening', 'temporal')
    do_spatial_whitening = params.getboolean('whitening', 'spatial')
    template_shift = params.getint('detection', 'template_shift')
    nodes, edges = get_nodes_and_edges(params)
    chunk_size = np.int64((t_stop - t_start) * sampling_rate)
    padding = (np.int64(t_start * sampling_rate),
               np.int64(t_start * sampling_rate))

    spatial_whitening = load_data(params, 'spatial_whitening') \
        if do_spatial_whitening else None
    temporal_whitening = load_data(params, 'temporal_whitening') \
        if do_temporal_whitening else None

    thresholds = load_data(params, 'thresholds')
    data, _ = data_file.get_data(0, chunk_size, padding=padding, nodes=nodes)
    data_file.close()

    electrodes = check_electrode_arg(electrodes, N_e)

    if do_spatial_whitening:
        data = np.dot(data, spatial_whitening)
    if do_temporal_whitening:
        data = sp.ndimage.filters.convolve1d(data, temporal_whitening,
                                             axis=0, mode='constant')
    try:
        result = load_data(params, 'results')
    except Exception:
        result = {'spiketimes': {}, 'amplitudes': {}}

    curve = np.zeros((N_e, np.int64((t_stop - t_start) * sampling_rate)),
                     dtype=np.float32)
    if templates is None:
        try:
            templates = load_data(params, 'templates')
        except Exception:
            templates = np.zeros((0, 0, 0))
    for key in list(result['spiketimes'].keys()):
        elec = int(key.split('_')[1])
        lims = (np.int64(t_start*sampling_rate) + template_shift,
                np.int64(t_stop*sampling_rate) - template_shift-1)
        idx = np.where((result['spiketimes'][key] > lims[0]) &
                       (result['spiketimes'][key] < lims[1]))
        for spike, (amp1, amp2) in zip(result['spiketimes'][key][idx],
                                       result['amplitudes'][key][idx]):
            spike -= np.int64(t_start * sampling_rate)
            tmp1 = templates[:, elec].toarray().reshape(N_e, N_t)
            tmp2 = templates[:, elec+templates.shape[1] // 2]\
                .toarray()\
                .reshape(N_e, N_t)

            curve[:, spike - template_shift:
                     spike + template_shift + 1] += amp1 * tmp1 + amp2 * tmp2
    print("Reconstructed signal is shown in red.")

    idx = electrodes
    n_elec = len(idx)

    plt.figure(figsize=[figure_width, n_elec*3])
    for count, i in enumerate(idx):
        plt.subplot(n_elec, 1, count + 1)
        if count != n_elec - 1:
            plt.setp(plt.gca(), xticks=[])
        else:
            plt.xlabel('Time [ms]')

        scale_ms = sampling_rate / 1000
        x = np.arange(padding[0], padding[1] + chunk_size) / scale_ms
        y = data[:, i]
        plt.plot(x, y, lw=0.1, c=[0.5, 0.5, 0.5])
        plt.plot(x, curve[i], lw=0.2, c='r')

        ylim0 = np.abs(np.min([np.min([y]), np.min(curve[i]), -2 * thresholds[i]]))
        ylim1 = np.abs(np.max([np.max([y]), np.max(curve[i]),  2 * thresholds[i]]))
        plt.ylim(-ylim0 * 1.1, ylim1 * 1.1)
        xmin, xmax = plt.xlim()
        plt.xlim(xmin, xmax)
        plt.plot([xmin, xmax],
                 [-thresholds[i], -thresholds[i]], lw=0.15, ls='--', c='b')
        plt.plot([xmin, xmax],
                 [thresholds[i], thresholds[i]], lw=0.15, ls='--', c='b')
        plt.title(f'Electrode {i}')

    plt.tight_layout()
    plt.show()


# =============================================================================
def show_image(file_name, electrode):
    """Displays images generated at the clustering stage.
    """

    raw = Path(file_name)
    imfile = raw.parent / raw.stem / raw.stem / \
             'plots' / f'cluster_neg_{electrode}.png'
    img = mpimg.imread(imfile)
    plt.figure(figsize=[figure_width, 2*figure_width/3])
    plt.imshow(img)
    plt.axis('off')
    plt.title(f'Clusters detected on electrode {electrode}:')
    plt.show()


# =============================================================================
class ChannelInfo:

    def __init__(self, d):
        (self.channel_id,
         self.row_index,
         self.group_id,
         self.electrode_group,
         self.label,
         self.raw_data_type,
         self.unit,
         self.exponent,
         self.ad_zero,
         self.tick,
         self.conversion_factor,
         self.adc_bits,
         self.high_pass_filter_type,
         self.high_pass_filter_cutoff_freq,
         self.high_pass_filter_order,
         self.low_pass_filter_type,
         self.low_pass_filter_cutoff_freq,
         self.low_pass_filter_order
         ) = d


# =============================================================================
def read_from_hd5(filename):

    f = h5py.File(filename, 'r')
    fk0 = list(f.keys())
    g_data = f['Data']
#    g_data['ProgramName']
    fk1 = list(g_data.keys())
    g_rec_0 = f['Data/Recording_0']
    fk2 = list(g_rec_0.keys())
    g_analogstream = f['Data/Recording_0/AnalogStream']
    fk3 = list(g_analogstream.keys())
    g_stream0 = f['Data/Recording_0/AnalogStream/Stream_0']
    fk4 = list(g_stream0.keys())
    g_channel_data = f['Data/Recording_0/AnalogStream/Stream_0/ChannelData']
    fk_chd = g_channel_data.shape
    g_channel_data_time_stamps = \
        f['Data/Recording_0/AnalogStream/Stream_0/ChannelDataTimeStamps']
    fk_chdts = g_channel_data_time_stamps.shape
    channel_data_time_stamps = g_channel_data_time_stamps[:]
    g_infoch = f['Data/Recording_0/AnalogStream/Stream_0/InfoChannel']
    fk_infoch = g_infoch.shape
    channel_info = [ChannelInfo(d) for d in g_infoch]
    fk_infoch0 = g_infoch[0]
    fk_infoch1 = g_infoch[1]
    data = np.empty(g_channel_data.shape, dtype=g_channel_data.dtype)
    g_channel_data.read_direct(data)

    return channel_info, g_channel_data


# =============================================================================
def check_hdf5(filename):

    channel_info, g_channel_data = read_from_hd5(filename)
    print(f"sampling_rate: {1e6/channel_info[0].tick}")
    print(f"data_dtype: {g_channel_data.dtype}")
