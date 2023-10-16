from itertools import product
from typing import Dict, List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go

from maple.data_descriptor import InputData
from maple.clustering import Params as ClusteringParams
from maple.template_matching.path_finder import PathFinder
from maple.template_matching.by_distance import ByDistance as Matcher


# =============================================================================
class Sources:

    def __init__(self,
                 treatments: list,
                 gids_global: list,
                 datalabels: List[str],
                 labids: dict):

        self.gids_global = gids_global
        self.datalabels = datalabels
        self.dlis = labids

        self.descrs = self.init_descr()
        self.subsets = self.init_subsets(treatments)
        self.tids = self.init_tids()

        self.sankey, self.fluxes = self.set_sankey_fluxes()

    def init_descr(self):

        res = {}
        for gs in self.gids_global:
            k = ['0' for _ in range(len(self.datalabels))]
            for g in gs:
                k[self.dlis[g.datalabel]] = '1'
            k = ''.join(k)
            if k in res:
                res[k].append(gs)
            else:
                res[k] = [gs]

        return dict(sorted(res.items()))

    def init_subsets(self, trs):

        # Check that some main constraints on treatments are satisfied:
        # (a) only up to four treatments are implemented:
        assert len(trs) <= 5
        # (b) there should be a base case preceding the treatments, i.e.
        # the first datafile cannot be the one after treatment:
        assert self.dlis[trs[0]]
        # (c) treatments should label consecutive datafiles in accending order:
        for dl in trs:
            assert dl in self.datalabels
        tdlis = np.array([self.dlis[dl] for dl in trs])
        assert np.all(np.ediff1d(tdlis) == 1)

        res = {
            1: {dl: self._select1(dl)
                for dl in trs}
        }
        if len(trs) > 1:
            res[2] = {f"{dl1}_{dl2}": self._select2(dl1, dl2)
                      for dl1, dl2 in zip(trs[:-1],
                                          trs[1:])}
        if len(trs) > 2:
            res[3] = {f"{dl1}_{dl2}_{dl3}": self._select3(dl1, dl2, dl3)
                      for dl1, dl2, dl3 in zip(trs[:-2],
                                               trs[1:-1],
                                               trs[2:])}
        if len(trs) > 3:
            res[4] = {f"{dl1}_{dl2}_{dl3}_{dl4}":
                      self._select4(dl1, dl2, dl3, dl4)
                      for dl1, dl2, dl3, dl4 in zip(trs[:-3],
                                                    trs[1:-2],
                                                    trs[2:-1],
                                                    trs[3:])}
        if len(trs) > 4:
            res[5] = {f"{dl1}_{dl2}_{dl3}_{dl4}_{dl5}":
                          self._select5(dl1, dl2, dl3, dl4, dl5)
                      for dl1, dl2, dl3, dl4, dl5 in zip(trs[:-4],
                                                         trs[1:-3],
                                                         trs[2:-2],
                                                         trs[3:-1],
                                                         trs[4:])}
        return res

    def init_tids(self):

        res = {}
        for n, ss in self.subsets.items():
            res[n] = {}
            for dls, vv in ss.items():
                dd = dls.split(sep='_')
                dlis = vv['dli']
                res[n][dls] = {}
                for mask, w in vv.items():
                    if mask != 'dli':
                        res[n][dls][mask] = {}
                        for m, dl in zip(mask, dlis):
                            if m == '1':
                                res[n][dls][mask][dl] = {}
                                for w1 in w.values():
                                    for w2 in w1:
                                        for w3 in w2:
                                            if w3.datalabel == \
                                               self.datalabels[dl]:
                                                res[n][dls][mask][dl][w3.ti] = {}
                                res[n][dls][mask][dl] = \
                                    dict(sorted(res[n][dls][mask][dl].items()))
        return res

    def _select1(self, dl):

        dli = [self.dlis[dl] - 1,
               self.dlis[dl]]

        d = {'dli': dli,

             '01': {k: v for k, v in self.descrs.items()
                    if not int(k[dli[0]]) and int(k[dli[1]])},
             '10': {k: v for k, v in self.descrs.items()
                    if int(k[dli[0]]) and not int(k[dli[1]])},
             '11': {k: v for k, v in self.descrs.items()
                    if int(k[dli[0]]) and int(k[dli[1]])},
             '00': {k: v for k, v in self.descrs.items()
                    if not int(k[dli[0]]) and not int(k[dli[1]])}}

        return d

    def _select2(self, dl1, dl2):

        d1 = self._select1(dl1)
        d2 = self._select1(dl2)

        d = {'dli': d1['dli'] +
                    d2['dli'][1:]}

        for k in list(product('01', repeat=3)):
            d[''.join(k)] = \
                {k: self.descrs[k] for k in d1[''.join(k[:-1])].keys() &
                                            d2[''.join(k[1:])].keys()}

        return d

    def _select3(self, dl1, dl2, dl3):

        d1 = self._select1(dl1)
        d2 = self._select1(dl2)
        d3 = self._select1(dl3)
        d = {'dli': d1['dli'] +
                    d2['dli'][1:] +
                    d3['dli'][1:]}

        for k in list(product('01', repeat=4)):
            d[''.join(k)] = \
                {k: self.descrs[k] for k in d1[''.join(k[:-2])].keys() &
                                            d2[''.join(k[1:-1])].keys() &
                                            d3[''.join(k[2:])].keys()}

        return d

    def _select4(self, dl1, dl2, dl3, dl4):

        d1 = self._select1(dl1)
        d2 = self._select1(dl2)
        d3 = self._select1(dl3)
        d4 = self._select1(dl4)
        d = {'dli': d1['dli'] +
                    d2['dli'][1:] +
                    d3['dli'][1:] +
                    d4['dli'][1:]}

        for k in list(product('01', repeat=5)):
            d[''.join(k)] = \
                {k: self.descrs[k] for k in d1[''.join(k[:-3])].keys() &
                                            d2[''.join(k[1:-2])].keys() &
                                            d3[''.join(k[2:-1])].keys() &
                                            d4[''.join(k[3:])].keys()}

        return d

    def _select5(self, dl1, dl2, dl3, dl4, dl5):

        d1 = self._select1(dl1)
        d2 = self._select1(dl2)
        d3 = self._select1(dl3)
        d4 = self._select1(dl4)
        d5 = self._select1(dl5)
        d = {'dli': d1['dli'] +
                    d2['dli'][1:] +
                    d3['dli'][1:] +
                    d4['dli'][1:] +
                    d5['dli'][1:]}

        for k in list(product('01', repeat=6)):
            d[''.join(k)] = \
                {k: self.descrs[k] for k in
                 d1[''.join(k[:-4])].keys() &
                 d2[''.join(k[1:-3])].keys() &
                 d3[''.join(k[2:-2])].keys() &
                 d4[''.join(k[3:-1])].keys() &
                 d4[''.join(k[4:])].keys()}

        return d

    def set_sankey_fluxes(self):

        node_labels = [[f"{dl}_{k}" for k in ['off', 'pass', 'on']]
                       for dl in self.datalabels]
        node_labels[0] = [n for n in node_labels[0] if 'on' in n]
        node_labels[-1] = [n for n in node_labels[-1] if 'off' in n]
        node_labels = [j for sub in node_labels for j in sub]

        fluxes = {}
        for k, v in self.descrs.items():
            dlis = [j[0] for j in np.argwhere([int(i) for i in k])]
            if len(dlis) > 1:
                for i, (sdli, tdli) in enumerate(zip(dlis[:-1], dlis[1:])):
                    sdl = self.datalabels[sdli]
                    stag = 'on' if sdli == dlis[0] else 'pass'
                    tdl = self.datalabels[tdli]
                    ttag = 'off' if tdli == dlis[-1] else 'pass'
                    key = f"{sdl}_{stag} {tdl}_{ttag}"
                    for w in v:
                        vv = [w[i], w[i + 1]]
                        if key in fluxes:
                            fluxes[key].append(vv)
                        else:
                            fluxes[key] = [vv]

        # Put keys in order for convenience:
        fluxes = {f"{a} {b}": fluxes[f"{a} {b}"]
                  for i, a in enumerate(node_labels)
                  for b in node_labels[i+1:]
                  if f"{a} {b}" in fluxes}

        sankey = {
            'sources': [k.split(sep=' ')[0] for k in fluxes.keys()],
            'targets': [k.split(sep=' ')[1] for k in fluxes.keys()],
            'values': [len(v) for v in fluxes.values()]
        }
        sankey['node_labels'] = [nl for nl in node_labels
                                 if nl in sankey['sources'] or
                                    nl in sankey['targets']]

        return sankey, fluxes

    def plot_as_sankey(self):

        if self.sankey is None:
            self.sankey, self.fluxes = self.set_sankey_fluxes()

        clrs_nd = plt.cm.tab20.colors[:-1:2]
        clrs_lk = plt.cm.tab20.colors[1::2]
        nd_colors = [f"rgb({c[0]*255},{c[1]*255},{c[2]*255})" for c in clrs_nd]
        lk_colors = [f"rgb({c[0]*255},{c[1]*255},{c[2]*255})" for c in clrs_lk]
        suff = {'on': 0, 'pass': 1, 'off': 2}

        x = [self.dlis[nl.split(sep='_')[0]]/(len(self.datalabels))
             for nl in self.sankey['node_labels']]

        y = [suff[n.split(sep='_')[1]] * 0.25 + 0.05
             for n in self.sankey['node_labels']]

        node_colors = [nd_colors[self.dlis[n.split(sep='_')[0]]]
                       for n in self.sankey['node_labels']]
        node_dict = {y:x for x, y in enumerate(self.sankey['node_labels'])}
        source_inds = [node_dict[x] for x in self.sankey['sources']]
        target_inds = [node_dict[x] for x in self.sankey['targets']]
        link_colors = [lk_colors[self.dlis[n.split(sep='_')[0]]]
                       for n in self.sankey['targets']]

        nodes = dict(
            pad=150,
            thickness=20,
            label=self.sankey['node_labels'],
            x=x,
            y=y,
            color=node_colors,
            line=dict(color="black", width=0.5),
        )
        links = dict(
            source=source_inds,
            target=target_inds,
            value=self.sankey['values'],
            color=link_colors,
        )

        fig = go.Figure(data=[go.Sankey(
            valuesuffix = ' neurons',
            arrangement="snap",
            node=nodes,
            link=links
        )])

        fig.update_layout(title_text="Experimental series",
                          font_size=12,
                          autosize=False,
                          width=2200,
                          height=1200)
        fig.show()

    def plot_fractions_multi(self, dls: List[str]):

        if self.sankey is None:
            self.sankey, self.fluxes = self.set_sankey_fluxes()

        clrs = {'on':   plt.cm.Paired.colors[3],
                'pass': plt.cm.Paired.colors[5],
                'off':  plt.cm.Paired.colors[1]}

        fractions = {dl: {k: 0 for k in clrs} for dl in dls}
        for k, v in self.fluxes.items():
            kk = k.split(sep=' ')
            src, trg = kk[0], kk[1]
            ss = src.split(sep='_')
            dl_s, type_s = ss[0], ss[1]
            if dl_s in dls:
                fractions[dl_s][type_s] += len(v)
            gg = trg.split(sep='_')
            dl_t, type_t = gg[0], gg[1]
            if dl_t in dls and type_t == 'off':
                fractions[dl_t][type_t] += len(v)

        maxval = max([sum(v.values()) for v in fractions.values()])
        for dl, vv in  fractions.items():
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=[3, 6])
            fig.suptitle(dl)
            ax.bar(1, vv['off'], width=0.9, bottom=0,
                   color=clrs['off'], label='off')
            ax.bar(1, vv['pass'], width=0.9, bottom=vv['off'],
                   color=clrs['pass'], label='pass')
            ax.bar(1, vv['on'], width=0.9, bottom=vv['off']+vv['pass'],
                   color=clrs['on'], label='on')
            ax.set_ylabel('Flux types (matched templates)', fontsize=14)
            ax.legend(bbox_to_anchor=(1.05, 1.), loc='upper left',
                      borderaxespad=0.)
            ax.set_ylim([0., maxval])
            ax.spines['top'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['right'].set_visible(False)
            fig.tight_layout()
            fig.show()
