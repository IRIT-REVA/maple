"""Extracts and displays some basic spike-derived metrics.
"""

from enum import Enum
from functools import singledispatch
from typing import Dict, List, Optional, Sequence, Union

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np

from maple.colors import kelly as kelly_colors
from maple.electrode_positions import MeaGeometry
from maple.misc import average_dict_values
from maple.misc import variance_dict_values
from maple.sorted_reader.from_circus import Results


def set_template_ids(template_id, results):
    """Preprocesses template ids.
    """

    template_ids = template_id if template_id is not None \
        else np.sort([int(k.split('_')[1])
                      for k in results['spiketimes'].keys()])
    if not np.iterable(template_ids):
        template_ids = [template_ids]

    return template_ids


# ExtendedResults =============================================================

class ExtendedResults(Results):

    _enum_names = \
        Results._enum_names + \
        [
            'isis',
            'freqs'
        ]

    Fields = Enum(
        'Fields',
        _enum_names
    )

    def __init__(self,
                 data: Union[str, List[str]],
                 mea_name: Optional[str] = None,
                 fields: Optional[Fields] = None,
                 templ_ids: Optional[list] = None):

        assert type(fields) is None or self.Fields

        if fields is None:
            fields = self.Fields

        if self.Fields.spike_times not in fields:
            if self.Fields.freqs in fields or \
               self.Fields.isis in fields:
                fields = fields + [self.Fields.spike_times]

        super().__init__(data, fields, templ_ids)

        self.mea = MeaGeometry.initialize(mea_name) \
            if mea_name is not None else None

        self.isis = inter_spike_intervals(self.spike_times) \
            if self.Fields.isis in fields else None
        self.freqs = spike_frequencies(self.spike_times) \
            if self.Fields.freqs in fields else None

        self.template_ids = self.template_inds() \
            if self.mea is not None else None

        self.num_templ = self.set_num_templates()

    @classmethod
    def spike_stats(cls, fnames):

        srt = cls(fnames,
                  fields=[
                      cls.Fields.thresholds,
                      cls.Fields.amplitudes,
                      cls.Fields.spike_times,
                      cls.Fields.electrodes,
                      cls.Fields.isis,
                      cls.Fields.freqs
                  ])

        isis_avg = [average_dict_values(isi) for isi in srt.isis]
        freq_avg = [average_dict_values(fr) for fr in srt.freqs]
#        isis_avg_avg = [np.average(s) for s in isis_avg]
        isis_var = [variance_dict_values(isi) for isi in srt.isis]
#        isis_avg_var = [np.average(s) for s in isis_var]
        freq_var = [variance_dict_values(fr) for fr in srt.freqs]

        return isis_avg, isis_var, freq_avg, freq_var

    def template_inds(self):

        assert self.mea is not None

        return [[[k for k, v in els.items() if v == ch]
                 for ch in self.mea.channels]
                for els in self.electrodes]

    def set_num_templates(self):

        def num(u):
            return len(u) if isinstance(u, dict) else [len(h) for h in u]

        k = self.electrodes if self.electrodes is not None else \
            self.templates_dict if self.templates_dict is not None else \
            None

        return num(k) if k is not None else None


# Inter-Spike Intervals =======================================================

@singledispatch
def inter_spike_intervals(_):

    return None


@inter_spike_intervals.register
def _(spike_times: dict):

    return {ti: np.diff(st) for ti, st in spike_times.items()}


@inter_spike_intervals.register
def _(spike_times: list):

    return [inter_spike_intervals(st) for st in spike_times]


# Spike Frequencies ===========================================================

@singledispatch
def spike_frequencies(_):
    return None


@spike_frequencies.register
def _(spike_times: dict):

    return {ti: 1000./np.diff(st) for ti, st in spike_times.items()}


@spike_frequencies.register
def _(spike_times: list):

    return [spike_frequencies(st) for st in spike_times]


'''
def amplitudes(results, template_ids):

    return [results['amplitudes'][f'temp_{tid}'][:, 0]
            for tid in template_ids]


def extract_amplitude_data(filename,
                           template_id,
                           extension=None):
    """Extracts template amplitudes from the results file.
       Returns them classified by template index and accumulated.
    """

    params = load_parameters(filename)
    extension = "" if extension is None \
        else "-" + extension
    results = load_data(params, 'results', extension=extension)
    template_ids = set_template_ids(template_id, results)

    ampl = amplitudes(results, template_ids)
    ampl_total = np.concatenate(ampl, axis=0)

    return ampl, ampl_total
'''
