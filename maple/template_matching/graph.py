from itertools import cycle
from typing import Dict, List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np


# =============================================================================
class Node:

    def __init__(self,
                 ti: int,
                 datalabel: str,
                 ch: int):

        self.ti = ti
        self.datalabel = datalabel
        self.ch = ch
        self.edges = {'in': [], 'out': []}
        self.chid = None
        self.gid = None

    @staticmethod
    def by_ti(ti: int,
              ns: List):

        nn = [n for n in ns if n.ti == ti]
        assert len(nn) == 1

        return nn[0]

    def longest_in_edge_ind(self):

        return np.argmax([e.weight for e in self.edges['in']])

    def remove_edge(self,
                    direction: str,
                    i: int):

        self.edges[direction] = \
            [self.edges[direction][j]
             for j in range(len(self.edges[direction])) if j != i]

    @staticmethod
    def set_chids(ns: List):

        for i, n in enumerate(ns):
            n.chid = i


# =============================================================================
class Edge:

    def __init__(self,
                 source: Node,
                 target: Node,
                 weight: float,
                 lag: int):

        self.source: Node = source
        self.target: Node = target
        self.weight: float = weight
        self.lag: int = lag

    def add_to_nodes(self):

        self.source.edges['out'].append(self)
        self.target.edges['in'].append(self)

    def rm_from_nodes(self):

        self.source.edges['out'] = \
            [e for e in self.source.edges['out'] if e != self]
        self.target.edges['in'] = \
            [e for e in self.target.edges['in'] if e != self]

        print(f"Removing lag {self.lag} edge at ch {self.source.ch} between "
              f"{self.source.datalabel}:{self.source.ti} and "
              f"{self.target.datalabel}:{self.target.ti} "
              f" (weight {self.weight})")


# =============================================================================
class Graph:

    def __init__(self):

        self.nodes = None
        self.edges = None
        self.gids = None
        self.gids_global = None
        self.neurons_el = None
        self.neurons = None

    def create(self,
               paths: Dict[str, dict],
               dists: dict,
               electrodes: List[int],
               cc: dict,
               datalabels: List[str]):

        for c in cc.values():
            assert c[0] == c[1]

        self.nodes = {s: {dl: [Node(ti, dl, e) for ti, e in es.items()
                               if e == c[0]]
                          for es, dl in zip(electrodes, datalabels)}
                      for s, c in cc.items()}

        for nn in self.nodes.values():
            for n in nn.values():
                Node.set_chids(n)

        self.edges = {s: [] for s in cc}
        for (il, ppp), ddd in zip(paths.items(), dists.values()):
            for (s, pp), dd in zip(ppp.items(), ddd.values()):
                for p, d in zip(pp, dd):
                    for fi in range(len(datalabels[:-il])):
                        if p[fi] is not None and p[fi + il] is not None:
                            src = Node.by_ti(p[fi],
                                             self.nodes[s][datalabels[fi]])
                            trg = Node.by_ti(p[fi + il],
                                             self.nodes[s][datalabels[fi + il]])
                            eg = Edge(src, trg, d[fi], il)
                            eg.add_to_nodes()
                            self.edges[s].append(eg)
        # set gids
        self.gids = {s: self.channel_gids(s, datalabels) for s in cc}
        self.gids_global = []
        for v in self.gids.values():
            self.gids_global.extend(list(v.values()))

        return self

    def parce_from(self,
                   n: Node,
                   gid: int,
                   gs: list):
        n.gid = gid
        gs.append(n)
        for e in n.edges['out']:
            if e.target.gid is None:
                g = self.parce_from(e.target, gid, gs)
                if g != gid:
                    return g
            elif e.target.gid != gid:
                return e.target.gid

        return gid

    def channel_gids(self,
                     s: str,
                     datalabels: List[str]):

        chgs = {}
        gid = 0
        for dl in datalabels:
            for n in self.nodes[s][dl]:
                if n.gid is None:
                    chgs[gid] = []
                    g = self.parce_from(n, gid, chgs[gid])
                    if g == gid:
                        gid += 1
                    else:
                        self.resolve_collision(chgs, s, g, gid)
                        return self.channel_gids(s, datalabels)

        return chgs

    def resolve_collision(self,
                          chgs: dict,
                          s: str,
                          g1: int,
                          g2: int):
        edges = []
        for g in [g1, g2]:
            ns = chgs[g]
            for n in ns:
                for e in n.edges['out']:
                    edges.append(e)

        # remove weakest edge
        iw = np.argmax([e.weight for e in edges])

        edges[iw].rm_from_nodes()
        self.edges[s] = [e for e in self.edges[s] if e != edges[iw]]
        for nn in self.nodes[s].values():
            for n in nn:
                n.gid = None

    def plot(self, s: str, datalabels: List[str]):

        cc = cycle(plt.cm.tab20.colors)
        clrs_eg = clrs_nd = [next(cc) for _ in range(len(self.gids_global))]

        xx = {dl: i for i, dl in enumerate(datalabels)}

        fig, ax = plt.subplots(nrows=1, ncols=1)
        ax.set_title(f"electrode {s.split(sep=':')[0]}")
        for dl, nn in self.nodes[s].items():
            for i, n in enumerate(nn):
                for e in n.edges['in']:
                    ax.plot([xx[e.source.datalabel], xx[dl]],
                            [e.source.gid + (e.lag - 1)/20,
                             n.gid + (e.lag - 1)/20],
                            color = clrs_eg[e.lag])
                ax.scatter(xx[dl], n.gid,
                           marker='o', color=clrs_nd[n.gid], s=20)

        xticks = ax.get_xticks()
        xlabels = ['' for _ in xticks]
        for dl, x in xx.items():
            ind = np.where(xticks == x)[0][0]
            xlabels[ind] = dl
        ax.set_xticklabels(xlabels)
        ax.set_xlabel('Experimental data')
        fig.show()
