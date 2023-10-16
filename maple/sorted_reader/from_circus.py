from enum import Enum
from functools import singledispatch
from typing import List, Optional, Sequence, Union

import numpy as np

from circus.shared.files import load_data
from circus.shared.parser import CircusParser
from maple.misc import as_iterable
from maple.misc import consecutive_dict


# Results =====================================================================

class Results:

    _enum_names = [
        'params',
        'templates',
        'amplitudes',
        'thresholds',
        'spike_times',
        'clusters',
        'electrodes'
    ]

    Fields = Enum(
        'Fields',
        _enum_names
    )

    def __init__(self,
                 data: Union[str, List[str]],
                 fields: Optional[Fields] = None,
                 template_ids: Optional[Union[int, Sequence[int]]] = None):

        assert type(fields) is None or self.Fields
        if fields is None:
            fields = self.Fields

        if template_ids is not None:
            template_ids = as_iterable(template_ids)
        self.template_ids_to_load = template_ids

        self.data = data

        self.params = load_parameters(data) \
            if self.Fields.params in fields else None
        self.templates_np, self.templates_dict = load_templates(data) \
            if self.Fields.templates in fields else (None, None)
        self.amplitudes = load_amplitudes(data, template_ids) \
            if self.Fields.amplitudes in fields else None
        self.thresholds = load_thresholds(data) \
            if self.Fields.thresholds in fields else None
        self.spike_times = load_spike_times(data, template_ids) \
            if self.Fields.spike_times in fields else None
        self.clusters = load_clusters(data, template_ids) \
            if self.Fields.clusters in fields else None
        self.electrodes = load_electrodes(data) \
            if self.Fields.electrodes in fields else None

        self.remove_empty()

        print("")

    def remove_empty(self):

        empty = detect_empty_trains(
                load_spike_times(self.data, self.template_ids_to_load)
                ) if self.spike_times is None else \
                detect_empty_trains(self.spike_times)

        if self.templates_np is not None:
            self.templates_np = self.remove_templates(self.templates_np, empty)
        if self.templates_dict is not None:
            self.templates_dict = self.remove_templates(self.templates_dict, empty)
        if self.amplitudes is not None:
            self.amplitudes = self.remove_templates(self.amplitudes, empty)
#        if self.thresholds is not None:
#            self.thresholds = remove_templates(self.thresholds, empty)
        if self.spike_times is not None:
            self.spike_times = self.remove_templates(self.spike_times, empty)
        if self.clusters is not None:
            self.clusters = self.remove_templates(self.clusters, empty)
        if self.electrodes is not None:
            self.electrodes = self.remove_templates(self.electrodes, empty)

    # Remove template =========================================================
    @staticmethod
    def _remove_templates_from_dict(a: dict,
                                    ts: list,
                                    name: str):
        if len(ts):
            for t in ts:
                a.pop(t)
                print(f"Warning: removing empty template {t} in {name}")
            return consecutive_dict(a)

        return a

    @staticmethod
    def _remove_templates_from_nparray(a: np.ndarray,
                                       ts: list,
                                       name: str):

        if len(ts):
            print(f"Warning: removing empty template(s) {ts} in {name}")
            inds = np.concatenate([np.array(ts),
                                   np.array(ts) + a.shape[2]//2])
            return np.delete(a, inds, axis=2)

        return a

    def remove_templates_from_list(self,
                                   a: list,
                                   ts: list):

        return [self._remove_templates_from_nparray(b, t, fn)
                if isinstance(b, np.ndarray) else
                self._remove_templates_from_dict(b, t, fn)
                for b, t, fn in zip(a, ts, self.data)]

    def remove_templates(self,
                         a: Union[dict, np.ndarray, List[dict]],
                         ts: Union[List[int], List[List[int]]]):

        if isinstance(a, List):
            return self.remove_templates_from_list(a, ts)
        elif isinstance(a, np.ndarray):
            return self._remove_templates_from_nparray(a, ts, self.data)
        else:
            return self._remove_templates_from_dict(a, ts, self.data)


# Parameters ==================================================================

@singledispatch
def load_parameters(filename):
    """ Load parameters."""

    return None


@load_parameters.register
def _(filename: str):

    params = CircusParser(filename)
    _ = params.get_data_file()    # i.e. update N_t

    return params


@load_parameters.register
def _(fnames: list):

    return [load_parameters(fn) for fn in fnames]


# Templates ===================================================================

@singledispatch
def load_templates(_):

    return None


@load_templates.register
def _(fname : str):

    params = load_parameters(fname)

    N_e = params.getint('data', 'N_e')
    N_t = params.getint('detection', 'N_t')
    templ = load_data(params, 'templates').toarray()
    templ = templ.reshape(N_e, N_t, -1)
    num_t = templ.shape[2]//2

    return templ, {i: templ[:,:,(i,i+num_t)] for i in range(num_t)}


@load_templates.register
def _(fnames: list):

    a = [load_templates(fn) for fn in fnames]

    return [b[0] for b in a], [b[1] for b in a]


# Thresholds =================================================================

@singledispatch
def load_thresholds(_):

    return None


@load_thresholds.register
def _(fname : str):

    params = load_parameters(fname)
    return load_data(params, 'thresholds')


@load_thresholds.register
def _(fnames: list):

    return [load_thresholds(fn) for fn in fnames]


# Spike times =================================================================

@singledispatch
def load_spike_times(_,
                     template_ids: Optional[list] = None):

    return None


@load_spike_times.register
def _(fname: str,
      template_ids: Optional[list] = None):

    params = load_parameters(fname)
    sampling_rate = params.rate

    results = load_data(params, 'results')
    spike_times = results['spiketimes']
    if template_ids is None:
        template_ids = [int(k.split('_')[1]) for k in spike_times.keys()]
    else:
        if not np.iterable(template_ids):
            assert isinstance(template_ids, int)
            template_ids = [template_ids]

    spike_t = {tid : spike_times[f'temp_{tid}'] / (sampling_rate / 1e+3)
               for tid in sorted(template_ids)}

    return spike_t


@load_spike_times.register
def _(fnames: list,
      template_ids: Optional[list] = None):

    return [load_spike_times(fn, template_ids) for fn in fnames]


# Clusters ====================================================================

@singledispatch
def load_clusters(_,
                  template_ids: Optional[list] = None):
    return None


@load_clusters.register
def _(fname: str,
      template_ids: Optional[list] = None):

    def tpl(k):
        return int(k.split('_')[1])

    def name(k):
        return k.split('_')[0]

    params = load_parameters(fname)
    res = load_data(params, 'clusters')
    electrodes = res['electrodes']
    clusters = {tpl(k): v for k, v in res.items() if name(k) == 'clusters'}
    peaks = {tpl(k): v for k, v in res.items() if name(k) == 'peaks'}
    times = {tpl(k): v for k, v in res.items() if name(k) == 'times'}
    data = {tpl(k): v for k, v in res.items() if name(k) == 'data'}

    return {'electrodes': electrodes,
            'clusters': dict(sorted(clusters.items())),
            'peaks': dict(sorted(peaks.items())),
            'times': dict(sorted(times.items())),
            'data': dict(sorted(data.items())),
            }


@load_clusters.register
def _(fnames: list,
      template_ids: Optional[list] = None):

    return [load_clusters(fn, template_ids) for fn in fnames]


# Electrodes ==================================================================

@singledispatch
def load_electrodes(_):

    return None


@load_electrodes.register
def _(fname: str):

    return {i: e for i, e in enumerate(load_clusters(fname)['electrodes'])}


@load_electrodes.register
def _(fnames: list):

    return [load_electrodes(fn) for fn in fnames]


# =============================================================================

@singledispatch
def load_amplitudes(_,
                    template_ids: Optional[Union[int, list]] = None):

    return None


@load_amplitudes.register
def _(fname: str,
      template_ids: Optional[Union[int, list]] = None):

    params = load_parameters(fname)
    amplitudes = load_data(params, 'results')['amplitudes']

    if template_ids is None:
        template_ids = [int(k.split('_')[1]) for k in amplitudes.keys()]
    else:
        template_ids = as_iterable(template_ids)

    return {tid : amplitudes[f'temp_{tid}'] for tid in sorted(template_ids)}


@load_amplitudes.register
def _(fnames: list,
      template_ids: Optional[Union[int, list]] = None):

    return [load_amplitudes(fn, template_ids) for fn in fnames]


# Raw data ====================================================================

def import_raw_data(fname: str):

    params = load_parameters(fname)
    datafile = params.data_file
#    N_e = params.getint('data', 'N_e')
#    N_total = params.nb_channels
#    N_t = params.getint('detection', 'N_t')
#    _, positions = get_nodes_and_positions(params)
#    nodes, edges = get_nodes_and_edges(params)
#    inv_nodes = np.zeros(N_total, dtype=np.int32)
#    inv_nodes[nodes] = np.arange(len(nodes))
    datafile.open()

    return datafile


# Detect empty spike trains ===================================================

@singledispatch
def detect_empty_trains(_):

    return None


@detect_empty_trains.register
def _(spike_times: dict):

    return [int(k) for k, v in spike_times.items() if v.size == 0]


@detect_empty_trains.register
def _(spike_times: list):

    return [detect_empty_trains(st) for st in spike_times]
