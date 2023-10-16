from itertools import cycle
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np

from maple.template_matching.by_distance import ByDistance as Matcher
from maple.colors import extended_colortable


# =============================================================================
class PathFinder:

    def __init__(self,
                 match: Matcher):

        chs = match.srt.mea.channels
        self.cc = \
            {f"{c0}:{c1}": [c0, c1] for c0, c1 in zip(chs, chs)}
        self.channels = chs

        self.lags = match.lags
        self.num_files = match.num_files
        self.datalabels = match.datalabels
        self.pair_names = match.pair_names
#        self.cc = match.cc
        self.num_templ = match.srt.num_templ
        self.pairs = match.pairs
        self.trids = match.trids

        self.paths = None
        self.path_dists = None

    def apply(self):

        self.set_paths()

        self.paths_global = \
            {il: np.concatenate(list(v for v in p.values() if len(v)))
             for il, p in self.paths.items()}

        return self

    def set_paths(self):

        self.paths = {i : {s: [] for s in self.cc.keys()} for i in self.lags}
        self.path_dists = {i : {s: [] for s in self.cc.keys()} for i in self.lags}
        for il, kl in self.lags.items():
            for pair_name in kl:
                prs = self.pairs[pair_name]
                for s, c in self.cc.items():
                    for pr, d in zip(prs.matching['tids'][s],
                                     prs.matching['dists'][s]):
                        path_ind = None
                        tis = [p[prs.fis[0]] for p in self.paths[il][s]]
                        inds = np.argwhere(np.array(tis) == pr[0])
                        if inds.shape[0]:
                            assert inds.shape[0] == 1
                            path_ind = inds[0, 0]
                        if path_ind is None:
                            self.paths[il][s]\
                                .append([None for _ in range(self.num_files)])
                            self.path_dists[il][s] \
                                .append([None for _ in range(self.num_files - 1)])
                            self.paths[il][s][-1][prs.fis[0]] = pr[0]
                            self.paths[il][s][-1][prs.fis[1]] = pr[1]
                            self.path_dists[il][s][-1][prs.fis[0]] = d
                        else:
                            self.paths[il][s][path_ind][prs.fis[1]] = pr[1]
                            self.path_dists[il][s][path_ind][prs.fis[0]] = d

        return self

    def plot_paths(self):

        cc = cycle(extended_colortable)
        colors = [next(cc) for _ in self.paths]
        fig, ax = plt.subplots(ncols=1, nrows=1)

        for ch, ps in enumerate(self.paths):
            for pind, p in enumerate(ps):
                x = [i for i, q in enumerate(p) if q is not None]
                y = [ch + pind * 0.1] * len(x)
                ax.plot(x, y, marker='o', c=colors[ch])
        plt.show()
        print()
