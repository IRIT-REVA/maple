import matplotlib.pyplot as plt
import numpy as np

import maple.bursts.max_interval as maxiv
from maple.basic_analysis import inter_spike_intervals
from maple.bursts.container import Bursts
from maple.electrode_positions import MeaGeometry

def main_test(test_seq_t, test_isi):

    indxs = maxiv.Detector.find_bursts(test_seq_t,
                                       test_isi,
                                       maxiv.default_pars)
#    bursts = Detector(filename, pars, template_id=None)
    sizes = maxiv.Detector.set_sizes(indxs)

    fraction_of_singles = np.array([len(np.argwhere(sz==1)) / sum(sz) for sz in sizes])
    avg_fraction_of_singles = fraction_of_singles.mean()


def make_size_distr(filename, pars):

    bd = maxiv.Detector(filename, pars)
    indxs = bd.burst_indx
#    bursts = Detector(filename, pars, template_id=None)
    return bd.sizes


def process_detected(file_names):

#    bs = [maxiv.Detector(fn, maxiv.default_pars).apply() for fn in file_names]
#    [b.set_sizes() for b in bs]
#    [b.plot_spikes_by_id(t_interval=[2., 4], elids=[12, 27]) for b in bs]

    bd1 = [maxiv.Detector(fn, maxiv.default_pars) for fn in file_names]
    bs1 = [Bursts(bd).read_ids() for bd in bd1]
    [b.set_sizes_ex().save_sizes(False) for b in bs1]
#    [b.read_sizes(compact=True) for b in bs1]
#    [b.set_fractions() for b in bs1]
#    [b.set_freqs() for b in bs1]
#    [b.plot_spikes_by_id(t_interval=[2., 14.], elids=[12, 27]) for b in bs1]
#    [b.plot_spikes_by_id(t_interval=[2., 4.], template_ids=[39, 40, 95, 96]) for b in bs1]
#    [Bursts(bd).read().plot_spikes_by_id(t_interval=[2., 4.], elids=[12, 27]) for bd in bd1]


def detect_new(file_names):

    bs = [maxiv.Detector(fn, maxiv.default_pars).apply() for fn in file_names]
    [b.analyze().save() for b in bs]
#    [b.plot_spikes_by_id(t_interval=[2., 4], elids=[23, 25]) for b in bs]


#def main(filename):

#    sizes, fraction_of_singles, avg_fraction_of_singles = \
#        find_sizes(filename, maxiv.Pars)

#    show_burst_hist(sizes,
#                    None,
#                    bin_size=1)

#    clusters_data = load_cluster_data(filename)
#    template_el_ind = clusters_data['electrodes']
#    template_colors = np.empty((len(template_el_ind),3))
#    for i in template_colors:


#    plot_electrode_geometry(template_el_ind, template_colors)

#    print('')


def show_burst_hist(sizes,
                    template_id,
                    bin_size=1):
    """Displays the distribution of  burst lengths as a histogram.
       If template_id is None, all available templates are used.
    """
    size_total = np.concatenate(sizes, axis=0)

    fig = plt.figure(figsize=[12, 9])
    ax = plt.subplot(1, 1, 1)
    nb_bins = int(np.ceil(np.max(size_total) / bin_size))
    maximum_interval = float(nb_bins) * bin_size
    hist_kwargs = {
        'bins' : nb_bins,
        'range' : (0.0, maximum_interval),
        'density' : True,
        'histtype' : 'step',
        'stacked' : True,
        'fill' : False

    }
    [ax.hist(a, color=None, ls='--', lw=0.5, **hist_kwargs)
     for a in sizes]
    ax.hist(size_total, color='k', ls='-', lw=0.5, **hist_kwargs)
    ax.set_xlim(0.0, maximum_interval)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.ticklabel_format(axis='y', style='sci', scilimits=(-2, 0))
    ax.set_xlabel("rel value")
    ax.set_ylabel("probability")
    if template_id is not None:
        ax.set_title(f"burst szes (template {template_id})")
    else:
        ax.set_title(f"burst sizes (all templates, black solid line: average)")
    fig.tight_layout()
    plt.show()


def analyze_bursts_tmp(file_names, mea_name):

    mea = MeaGeometry.initialize(mea_name)
    electrodes = load_electrodes(file_names)
    templs = [[np.argwhere(es == ch) for ch in mea.channels] for es in electrodes]

    bsizes = [find_sizes(fn, maxiv.default_pars) for fn in file_names]  # sizes, fraction_of_singles, avg_fraction_of_singles
    fraction_of_singles_templ = [s[1] for s in bsizes]
    fraction_of_singles_el = [[np.average([fst[t[0]] for t in ts]) for ts in tts]
                              for fst, tts in zip(fraction_of_singles_templ, templs)]
    maxfracsingles_templ = max([max(frs) for frs in fraction_of_singles_templ])
    maxfracsingles_el = max([max(frs) for frs in fraction_of_singles_el])
#    [plot_bursts(fn, dn, mea_name, frse, frst, maxfracsingles_el, maxfracsingles_templ) for fn, dn, frse, frst in zip(file_names, data_names, fraction_of_singles_el, fraction_of_singles_templ)]


def process_1(file_names, data_names, mea_name):

#    bd = [maxiv.Detector(fn, maxiv.default_pars).apply().save() for fn in file_names]
    #    plot_spike_times_by_bursts(file_names[fi], t_interval=tint, bdetector=bd, elids=ei)# [82, 83, 84, 85])
    bd = [maxiv.Detector(fn, maxiv.default_pars).read() for fn in file_names]

    mea = MeaGeometry.initialize(mea_name)
    electrodes = load_electrodes(file_names)
    templs = [[np.argwhere(es == ch) for ch in mea.channels] for es in electrodes]
    spike_times = load_spike_times(file_names)
    isis = inter_spike_intervals(spike_times)
    spike_tmax = [{key: np.max(val) for key, val in st.items()} for st in spike_times]
    isis_bursts = []
    sizes = [bd[i].sizes for i, fn in enumerate(file_names)]  # sizes, fraction_of_singles, avg_fraction_of_singles
    fraction_of_single_spikes = [np.array([len(np.argwhere(sz==1)) / sum(sz) for sz in sizes[i]]) for i, fn in enumerate(file_names)]
    fraction_of_spikes_in_bursts = [1 - fs for fs in fraction_of_single_spikes]
    avg_fraction_of_singles = [fs.mean() for fs in fraction_of_single_spikes]
    avg_fraction_of_spikes_in_bursts = [fb.mean() for fb in fraction_of_spikes_in_bursts]
    brsts = [np.array([sz[np.squeeze(np.argwhere(sz > 1))] for sz in sizes[i]]) for i, fn in enumerate(file_names)]
    burst_sizes = [np.array([np.average(sz[np.argwhere(sz > 1)]) for sz in sizes[i]]) for i, fn in enumerate(file_names)]
    figure, ax = plt.subplots(111)

    fraction_of_singles_el = [[np.average([fst[t[0]] for t in ts]) for ts in tts]
                              for fst, tts in zip(fraction_of_single_spikes, templs)]
    maxfracsingles_templ = max([max(frs) for frs in fraction_of_single_spikes])
    maxfracsingles_el = max([max(frs) for frs in fraction_of_singles_el])
#    [plot_bursts(fn, dn, mea_name, frse, frst, maxfracsingles_el, maxfracsingles_templ)
#     for fn, dn, frse, frst in zip(file_names, data_names, fraction_of_singles_el, fraction_of_single_spikes)]
