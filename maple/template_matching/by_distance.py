from enum import Enum
from itertools import cycle
from typing import Dict, Sequence, Optional

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import plotly.graph_objects as go

from maple.data_descriptor import InputData
import maple.bursts.max_interval as maxiv
from maple.bursts.container import Bursts

from maple.colors import extended_colortable
from maple.colors import kelly as kelly_colors
from maple.clustering import cluster_templates_by_dist
from maple.clustering import Params as ClusteringParams
from maple.basic_analysis import ExtendedResults as Sorted


# =============================================================================
class T:
    def __init__(self,
                 fi: int,
                 ch: int,
                 ti: int):

        self.fi = fi
        self.ch = ch
        self.ti = ti


# =============================================================================
class TPair:

    def __init__(self, t1: T, t2: T, gid: int):

        self.t1 = t1
        self.t2 = t2
        self.dist = None
        self.chs = f"{t1.ch}:{t2.ch}"
        self.gid = gid

    def set_dist(self, ns, ts):

        i1 = self.t1.ti
        i2 = self.t2.ti
        c1 = self.t1.ch
        c2 = self.t2.ch
        nt1 = ns[self.t1.fi]
        nt2 = ns[self.t2.fi]
        t1 = ts[self.t1.fi]
        t2 = ts[self.t2.fi]

        self.dist = np.sqrt(np.average(np.square(
            (t1[c1, :, i1] + t1[c1, :, i1 + nt1]) -
            (t2[c1, :, i2] + t2[c1, :, i2 + nt2])
        )))


# =============================================================================
class TemplatePairs:

    def __init__(self,
                 dataex: InputData,
                 srt: Sorted):

        self._dnames = dataex.data2
        self._datalabels = dataex.labels2
        self._num_files = dataex.num_files

        self._srt = srt
        self._template_ids = self._srt.template_ids
        self._templates = self._srt.templates_np
        self._num_templates = self._srt.num_templ
        self._labels2 = dataex.labels2

        self.num_channels = None
        self.lag = None
        self.pair_names = None
        self.ps = None
        self.fis = None
        self.dls = None
        self.chs = None
        self.cc = None
        self.dists = None
        self.gids_ch = None
        self.clusters = None
        self.matching = {}

    def create(self, fi1, fi2, chs1, chs2, starting_gid):

        assert len(chs1) == len(chs2)
        assert fi1 <= fi2

        self.num_channels = len(chs1)
        self.lag = fi2 - fi1

        self.fis = [fi1, fi2]
        self.dls = [self._datalabels[fi1], self._datalabels[fi2]]
        self.chs = [chs1, chs2]
        self.cc = \
            {f"{c0}:{c1}": [c0, c1] for c0, c1 in zip(self.chs[0], self.chs[1])}
        self.pair_names = f"{self._labels2[fi1]} -> {self._labels2[fi2]}"
        self.ps = {}
        self.gids_ch = {s: [] for s in self.cc}
        gid = starting_gid
        for s, c in self.cc.items():
            for a1 in self._template_ids[fi1][c[0]]:
                for a2 in self._template_ids[fi2][c[1]]:
                    self.ps[gid] = TPair(T(fi1, c[0], a1), T(fi2, c[1], a2), gid)
                    self.gids_ch[s].append(gid)
                    gid += 1
            self.gids_ch[s] = np.array(self.gids_ch[s])

        return self

    def __str__(self):

        return f"{self.dls[0]}-{self.dls[1]}"

    def set_distances(self):

        for p in self.ps.values():
            p.set_dist(self._num_templates, self._templates)

        self.dists = {s: {} for s in self.cc}
        for p in self.ps.values():
            self.dists[p.chs][p.gid] = p.dist

        return self

    def find_matching_pairs(self): #, ds):

        self.matching['gids'] = {k: None for k in self.cc}
        self.matching['tids'] = {k: None for k in self.cc}
        self.matching['dists'] = {k: None for k in self.cc}
        self.matching['_'] = {k: None for k in self.cc}
        for s, c in self.cc.items():
            inds = np.where(self.clusters['labels_ch'][s] ==
                            self.clusters['relevant_label'])[0]
            gs = self.gids_ch[s][inds]
            self.matching['gids'][s] = gs
            self.matching['tids'][s] = \
                [[self.ps[g].t1.ti, self.ps[g].t2.ti] for g in gs]
            self.matching['dists'][s] = [self.ps[g].dist for g in gs]
            self.matching['_'][s] = self.clusters['labels_ch'][s][inds]
        return self

    def set_clusters(self, clusters):

        self.clusters = clusters.copy()
        self.clusters['labels_ch'] = {}
        self.clusters['labels_ch_'] = {}
        self.clusters['colors_ch'] = {}
        self.clusters['n_items_ch'] = {}
        self.clusters['n_outsiders_ch'] = {}
        self.clusters['n_clustered_ch'] = {}
        for s, c in self.cc.items():
            self.clusters['labels_ch'][s] = clusters['labels'][self.gids_ch[s]]
            self.clusters['labels_ch_'][s] = clusters['labels_'][self.gids_ch[s]]
            self.clusters['colors_ch'][s] = [clusters['colors'][i] for i in self.gids_ch[s]]
            self.clusters['n_items_ch'][s] = len(self.clusters['labels_ch'][s])
            self.clusters['n_outsiders_ch'][s] = \
                len(np.where(self.clusters['labels_ch'][s] == -1)[0])
            self.clusters['n_clustered_ch'][s] = \
                self.clusters['n_items_ch'][s] - \
                self.clusters['n_outsiders_ch'][s]

    def plot(self):

        dist = np.array([pp.dist for pp in self.ps.values()])
        chs = np.array([pp.t1.ch for pp in self.ps.values()])
        colors = [self.clusters['colors'][pp.gid] for pp in self.ps.values()]

        figtitle = "Pairwise distances"
        fig, ax = plt.subplots(ncols=1, nrows=1)
        fig.suptitle(figtitle, fontsize=16, fontweight='bold')
        ax.scatter(chs, dist, s=12, color=colors)
        ax.set_title(str(self), y=1.0, pad=-16)
#        [a.set_ylim(ylim) for a in ax]
#        [a.set_xlim([-2, len(self.channels) + 2]) for a in ax]
        ax.legend(handles=self.clusters['legend_handles'],
                      title='Cluster',
                      bbox_to_anchor=(1.02, 1),
                      loc='upper left',
                      borderaxespad=0.)
        ax.set_xlabel('Channel id', fontsize=14)
        fig.text(0.06, 0.5,
                  'Distance (au)',
                  va='center',
                  rotation='vertical',
                  fontsize=14)
        fig.show()


# =============================================================================
class ByDistance:

    def __init__(self,
                 dataex: InputData):

        self.dataex = dataex
        self.fnames = dataex.fnames
        self.dnames = dataex.data2
        self.datalabels = dataex.labels2
        self.num_files = dataex.num_files

        self.srt = Sorted(self.fnames,
                          dataex.mea_name,
                          [Sorted.Fields.templates,
                           Sorted.Fields.electrodes])
        self.electrodes = self.srt.electrodes
        self.templates = self.srt.templates_np
        self.template_ids = self.srt.template_ids
        self.num_templates = self.srt.num_templ
        self.num_pairs = 0
        self.pairs = {}
        self.lags = None
        self.pair_names = None
        self.trids = None

    def create(self,
               fis1: Sequence[int],
               fis2: Sequence[int],
               chs1: Sequence[int],
               chs2: Sequence[int]):

        assert len(fis1) == len(fis2)

        for fi1, fi2 in zip(fis1, fis2):
            prs = TemplatePairs(self.dataex, self.srt)
            self.pairs[str(prs)] = \
                prs.create(fi1, fi2, chs1, chs2, self.num_pairs)
            self.num_pairs += len(prs.ps)

        return self

    def select_related(self,
                       pars: ClusteringParams):

        self.set_lags()

        for p in self.pairs.values():
            p.set_distances()
        self.clusterize(pars)
        for p in self.pairs.values():
            p.find_matching_pairs()

        return self

    def set_lags(self):

        lags = np.unique([p.lag for p in self.pairs.values()])
        self.lags = \
            {i: [k for k, v in self.pairs.items() if v.lag == i] for i in lags}
        self.pair_names = \
            {i: [v.pair_names for k, v in self.pairs.items() if v.lag == i]
             for i in lags}
        self.trids = \
            {il: [a for a in range(len(v))]
             for il, v in self.pair_names.items()}

    def clusterize(self,
                   pars: ClusteringParams):

        ds1d = np.array([pp.dist for v in self.pairs.values()
                         for pp in v.ps.values()])
        gids = np.array([pp.gid for v in self.pairs.values()
                         for pp in v.ps.values()])

        clusters = cluster_templates_by_dist(ds1d,
                                             pars,
                                             extended_colortable[2:])
        clusters['dists1d'] = ds1d
        clusters['gids'] = gids
        clusters['relevant_label'] = 1
        for pv in self.pairs.values():
            pv.set_clusters(clusters)

    def set_pair_gids(self):

        j = 0
        for k, v in self.pairs.items():
            for pp in v.ps:
                pp.gid = j
                j += 1
            v.set_gids()

    def distances(self, p: TPair):

        i1 = p.t1.ti
        i2 = p.t2.ti
        c1 = p.t1.ch
        c2 = p.t2.ch
        nt1 = self.num_templates[p.t1.fi]
        nt2 = self.num_templates[p.t2.fi]
        t1 = self.templates[p.t1.fi]
        t2 = self.templates[p.t2.fi]
        return np.sqrt(
            np.average(
                np.square((t1[c1, :, i1] + t1[c1, :, i1 + nt1]) -
                          (t2[c2, :, i2] + t2[c2, :, i2 + nt2]))))

    @staticmethod
    def pairwise_distances(s1, s2, t1, t2):

        nt1 = t1.shape[1] // 2
        nt2 = t2.shape[1] // 2

        return np.sqrt([[np.average(np.square((t1[:, i] + t1[:, i + nt1]) -
                                              (t2[:, j] + t2[:, j + nt2])))
                         for j in s2]
                        for i in s1])

    def plot(self):

        for p in self.pairs.values():
            p.plot()
        self.plot_single_fig()

    def plot_single_fig(self):

        figtitle = "Pairwise distances"
        fig, ax = plt.subplots(ncols=1,
                               nrows=len(self.pairs),
                               sharex=True)
        fig.suptitle(figtitle, fontsize=16, fontweight='bold')
        ylim = max([max([max(d.values())
                         for d in p.dists.values()])
                    for p in self.pairs.values()])
        xlim = max([p.num_channels for p in self.pairs.values()])
        for pi, p in enumerate(self.pairs.values()):
            for s, d in p.dists.items():
                c = s.split(sep=':')
                y = list(d.values())
                x = [int(c[0])] * len(y)
                color = p.clusters['colors_ch'][s]
                ax[pi].scatter(x, y, s=12, color=color)
            ax[pi].set_title(str(p), y=1.0, pad=-16)
            ax[pi].set_ylim([0, ylim])
            ax[pi].set_xlim([-2, xlim + 2])
        ax[0].legend(
            handles=next(iter(self.pairs.values())).clusters['legend_handles'],
            title='Cluster',
            bbox_to_anchor=(1.02, 1),
            loc='upper left',
            borderaxespad=0.
        )
        ax[-1].set_xlabel('Channel id', fontsize=14)
        fig.text(0.06, 0.5,
                 'Distance (au)',
                 va='center',
                 rotation='vertical',
                 fontsize=14)
        fig.subplots_adjust(hspace=0.05)
        fig.show()


def test(dataex: InputData,
         pars: ClusteringParams,
         fis: Optional[Sequence[int]] = None):

    if fis is None:
        fis = dataex.file_indexes

    match = ByDistance(dataex)
    for fi in fis:
        chs = match.srt.mea.channels
        randchs = np.random.randint(0, np.max(chs), len(chs))
        match.create([fi], [fi],
                     chs, randchs)
    match.select_related(pars).plot()
