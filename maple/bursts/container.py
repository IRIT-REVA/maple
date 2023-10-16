from itertools import cycle

import matplotlib.pyplot as plt
import numpy as np

from maple.basic_analysis import ExtendedResults as Sorted
from maple.colors import kelly as kelly_colors


class Bursts:

    def __init__(self, detector):

        self.file_name = detector.file_name
        self.suffix = detector.signature()
        self.path = "/".join(self.file_name.split(sep='/')[:-1])

        srtd = Sorted(self.file_name,
                      fields=[Sorted.Fields.spike_times,
                              Sorted.Fields.electrodes,
                              Sorted.Fields.isis,
                              Sorted.Fields.freqs])
        self.spike_times = srtd.spike_times
        self.electrodes = srtd.electrodes
        self.isis = srtd.isis

        self.ids = None
        self.sizes = {}
        self.sizes_ex = {}
        self.borders = {}
        self.freqs = {'all': srtd.freqs}
        self.fractions = {}

    def analyze(self):

        assert len(self.ids)

        self.set_sizes()
        self.set_sizes_ex()
        self.set_borders()
        self.set_fractions()

        return self

    def set_sizes(self):

        assert len(self.ids)

        self.sizes = {ti: np.empty(bi[-1]+1, dtype=np.int)
                      if bi.shape[0]
                      else np.empty(0, dtype=np.int)
                      for ti, bi in self.ids.items()}

        for sz, ix in zip(self.sizes.values(), self.ids.values()):
            for n in range(sz.shape[0]):
                sz[n] = len(np.where(ix == n)[0])

        return self

    def set_sizes_ex(self):

        assert len(self.ids)

        self.sizes_ex = {ti: np.empty(bi.shape[0], dtype=int)
                         for ti, bi in self.ids.items()}

        for sz, ix in zip(self.sizes_ex.values(), self.ids.values()):
            for n in range(ix[-1]+1):
                inds = np.where(ix == n)[0]
                sz[inds] = len(inds)

        return self

    def set_borders(self):

        assert len(self.ids)

        for ti, iid in self.ids.items():
            self.borders[ti] = np.zeros_like(iid)
            for idx in range(max(iid)):
                inds = np.where(iid == idx)[0]
                if inds.size > 1:
                    self.borders[ti][inds[0]] = 1   # burst first spike
                    self.borders[ti][inds[-1]] = 2  # burst last spike
                else:
                    self.borders[ti][inds[0]] = -1  # non-burst spike

        return self

    def set_fractions(self):

        if len(self.borders) == 0:
            self.set_borders()
        if len(self.sizes_ex) == 0:
            self.set_sizes_ex()

        self.fractions["spikes_outside"] = \
            {ti: v[v == -1].size / v.size for ti, v in self.borders.items()}
        self.fractions["spikes_inside"] = \
            {ti: 1. - s for ti, s in self.fractions["spikes_outside"].items()}
        self.fractions["time_outside"] = \
            {ti: isi[s[1:] == 1].sum() / st[-1]
             for (ti, isi), s, st in zip(self.isis.items(),
                                         self.sizes_ex.values(),
                                         self.spike_times.values())}
        self.fractions["time_inside"] = \
            {ti: 1. - s for ti, s in self.fractions["time_outside"].items()}

    def set_freqs(self):

        if len(self.borders) == 0:
            self.set_borders()
        if len(self.sizes_ex) == 0:
            self.set_sizes_ex()

        self.freqs['all_avg'] = \
            {key: np.average(val) for key, val in self.freqs['all'].items()}

        self.freqs['inburst'] = \
            {ti: fr[s[1:] > 1]
             for (ti, fr), s in zip(self.freqs['all'].items(),
                                    self.sizes_ex.values())}
        self.freqs['inburst_avg'] = \
            {key: np.average(val)
             for key, val in self.freqs['inburst'].items()}
        self.freqs['inburst_std'] = \
            {key: np.std(val) for key, val in self.freqs['inburst'].items()}

        self.freqs['outburst'] = \
            {ti: fr[s[1:] == 1]
             for (ti, fr), s in zip(self.freqs['all'].items(),
                                    self.sizes_ex.values())}
        self.freqs['outburst_avg'] = \
            {key: np.average(val)
             for key, val in self.freqs['outburst'].items()}
        self.freqs['outburst_std'] = \
            {key: np.std(val) for key, val in self.freqs['outburst'].items()}

        self.freqs['interburst'] = \
            {ti: 1000./np.diff(st[b == 1])
             for (ti, st), b in zip(self.spike_times.items(),
                                    self.borders.values())}
        self.freqs['interburst_avg'] = \
            {key: np.average(val)
             for key, val in self.freqs['interburst'].items()}
        self.freqs['interburst_std'] = \
            {key: np.std(val) for key, val in self.freqs['interburst'].items()}

        return self

    def read(self):

        self.read_ids()
        self.read_borders()
        self.read_sizes(compact=True)
        self.read_sizes(compact=False)

        return self

    def read_ids(self):

        fname = self.ids_filename()
        self.ids = {}
        with open(fname, 'r') as f:
            while line := f.readline().rstrip():
                a = line.split(sep=':')
                self.ids[int(a[0])] = np.fromstring(a[1], dtype=int, sep=' ')

        return self

    def read_borders(self):

        fname = self.borders_filename()
        self.borders = {}
        with open(fname, 'r') as f:
            while line := f.readline().rstrip():
                a = line.split(sep=':')
                self.borders[int(a[0])] = \
                    np.fromstring(a[1], dtype=int, sep=' ')

        return self

    def read_sizes(self, compact=True):

        fname = self.sizes_filename(compact)
        self.sizes = {}
        with open(fname, 'r') as f:
            while line := f.readline().rstrip():
                a = line.split(sep=':')
                b = np.fromstring(a[1], dtype=int, sep=' ')
                if compact:
                    self.sizes[int(a[0])] = b
                else:
                    self.sizes[int(a[0])] = b

        return self

    def save(self):

        self.save_ids()
        self.save_borders()
        self.save_sizes(compact=True)
        self.save_sizes(compact=False)

        return self

    def save_ids(self):

        fname = self.ids_filename()
        with open(fname, 'w') as f:
            for tpl, vals in self.ids.items():
                f.write(f"{tpl} : ")
                for v in vals:
                    f.write(f"{v} ")
                f.write('\n')

        return self

    def save_borders(self):

        fname = self.borders_filename()
        with open(fname, 'w') as f:
            for tpl, vals in self.borders.items():
                f.write(f"{tpl} : ")
                for v in vals:
                    f.write(f"{v} ")
                f.write('\n')

        return self

    def save_sizes(self, compact=True):

        fname = self.sizes_filename(compact)
        with open(fname, 'w') as f:
            s = self.sizes if compact else self.sizes_ex
            for tpl, vals in s.items():
                f.write(f"{tpl} : ")
                for v in vals:
                    f.write(f"{v} ")
                f.write('\n')

        return self

    def ids_filename(self):

        return f"{self.path}/burst_indx_{self.suffix}.txt"

    def borders_filename(self):

        return f"{self.path}/burst_borders_{self.suffix}.txt"

    def sizes_filename(self, compact=True):

        ex = '' if compact else '_ex'
        return f"{self.path}/burst_sizes{ex}_{self.suffix}.txt"

    def plot_spikes_by_id(self,
                          t_interval,
                          elids=None,
                          template_ids=None):

        assert np.iterable(t_interval) and \
               len(t_interval) == 2 and \
               t_interval[0] < t_interval[1]
        t_min = t_interval[0] * 1.e+3
        t_max = t_interval[1] * 1.e+3

        if elids is None:
            if not np.iterable(template_ids):
                assert isinstance(template_ids, int)
                template_ids = [template_ids]
            elids = np.unique([self.electrodes[tid] for tid in template_ids])
            etids = [np.flip([ti for ti, v in self.electrodes.items()
                              if v == e and ti in template_ids])
                     for e in elids]
        else:
            if not np.iterable(elids):
                assert isinstance(elids, int)
                elids = [elids]
            template_ids = \
                np.concatenate([[ti for ti, v in self.electrodes.items()
                                 if v == e] for e in elids])
            etids = [np.flip([ti for ti, v in self.electrodes.items()
                              if v == e]) for e in elids]
        spike_t = {tid: self.spike_times[tid] for tid in template_ids}

        colors = cycle(kelly_colors.values())
        color_single = (0.5, 0.5, 0.5)
        fig, ax = plt.subplots(nrows=len(elids),
                               sharex=True,
                               figsize=(16, 1.5*len(template_ids)))
        if not np.iterable(ax):
            ax = [ax]
        fig.suptitle("Spikes according to burst id", fontsize=16)
        for ei, et in enumerate(etids):
            for ti, tid in reversed(list(enumerate(et))):
                spt = np.intersect1d(np.where(spike_t[tid] >= t_min)[0],
                                     np.where(spike_t[tid] <= t_max)[0])
                bits = self.ids[tid][spt]
                ubits = np.unique(self.ids[tid][spt])
                clrs = np.empty((bits.size, 3))
                for a in ubits:
                    q = np.where(bits == a)[0]
                    clr = np.array(next(colors))/255 if q.size > 1 else \
                        color_single
                    clrs[q, :] = clr
                    if q.size > 1:
                        ax[ei].plot([spike_t[tid][spt[q[0]]],
                                     spike_t[tid][spt[q[-1]]]],
                                    [ti + 0.3, ti + 0.3],
                                    ls='-', lw=3, color=clr)
                for i, t in enumerate(spike_t[tid][spt]):
                    ax[ei].plot([t, t],
                                [ti - 0.5, ti + 0.3],
                                ls='-', lw=0.5, color=clrs[i, :])
            ax[ei].set_title(f"Electrode {elids[ei]}", y=1, pad=-16)
            ax[ei].set_yticks(list(range(len(et) - 1, -1, -1)))
            ax[ei].set_yticklabels([str(b) for b in np.flip(et, axis=0)])
            ax[ei].xaxis.set_visible(False)
            ax[ei].spines['left'].set_visible(False)
            ax[ei].spines['right'].set_visible(False)
            ax[ei].spines['top'].set_visible(False)
            ax[ei].spines['bottom'].set_visible(False)
            ax[ei].set_ylabel('template id')
            ax[ei].set_ylim(-0.7, len(et) - 0.2)
        ax[-1].xaxis.set_visible(True)
        ax[-1].spines['bottom'].set_visible(True)
        ax[-1].set_xlabel('time (ms)')
        fig.subplots_adjust(hspace=0.05)
        fig.show()
