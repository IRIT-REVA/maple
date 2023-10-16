from enum import Enum
from typing import Dict, List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np

import maple.bursts.max_interval as maxiv
from maple.basic_analysis import ExtendedResults as Sorted
from maple.bursts.container import Bursts
from maple.clustering import Params as ClusteringParams
from maple.data_descriptor import InputData
from maple.template_matching.path_finder import PathFinder
from maple.template_matching.by_distance import ByDistance as Matcher
from maple.template_matching.graph import Graph
from maple.template_matching.source_detector import Sources


# =============================================================================
class FrequencyExaminer:

    def __init__(self,
                 dataex: InputData):

        self.fnames = dataex.fnames
        self.datalabels = dataex.labels2
        self.dlis = dataex.labids
        self.freqs = [Sorted(fn, fields=[Sorted.Fields.freqs]).freqs
                      for fn in self.fnames]

        self.tids = None

    def set(self,
            tids):

        for n, nn in tids.items():
            for vvv in nn.values():
                for vv in vvv.values():
                    for fi, v in vv.items():
                        for ti, w in v.items():
                            w['freq_avg_all'] = np.average(self.freqs[fi][ti])
        self.tids = tids

    def plot_sel1_violin_multi(self, dls: List[str]):

        mask = [('off',         '10', -1),
                ('pass_before', '11', -1),
                ('pass_after',  '11',  0),
                ('on',          '01',  0)]

        fr = {}
        for k, v in self.tids[1].items():
            if k in dls:
                fr[k] = {}
                for m in mask:
                    a = self.dlis[k] + m[2]
                    fr[k][m[0]] = \
                        [w["freq_avg_all"] for w in v[m[1]][a].values()]

        clrs_e = [plt.cm.Paired.colors[1],
                  plt.cm.Paired.colors[5],
                  plt.cm.Paired.colors[5],
                  plt.cm.Paired.colors[3]]
        clrs_f = [plt.cm.Paired.colors[0],
                  plt.cm.Paired.colors[4],
                  plt.cm.Paired.colors[4],
                  plt.cm.Paired.colors[2]]
        colors = {c[0]: {'edge': clrs_e[i],
                         'face': clrs_f[i]} for i, c in enumerate(mask)}
        lim_max = min(max([max([max(b) for b in a.values()])
                           for a in fr.values()]), 400)
        fig = [None for _ in range(len(self.tids[1]))]
        ax = [None for _ in range(len(self.tids[1]))]
        for i, (k, v) in enumerate(fr.items()):
            fig[i], ax[i] = plt.subplots(nrows=1, ncols=1)
            fig[i].canvas.draw()
            fig[i].suptitle("Spike frequencies", fontsize=16)
            x = np.array(range(1, len(v) + 1))
            vpl = ax[i].violinplot(list(v.values()),
                                   x,
                                   points=60,
                                   widths=0.4,
                                   showmeans=True,
                                   showmedians=False,
                                   showextrema=False,
                                   bw_method=0.5)
            for patch, c in zip(vpl['bodies'], v.keys()):
                patch.set_facecolor(colors[c]['face'])
                patch.set_edgecolor(colors[c]['edge'])
            for (kk, vv), xx in zip(v.items(), x):
                ax[i].scatter(xx + np.random.normal(0, 0.07, len(vv)),
                              vv,
                              marker='.',
                              color=colors[kk]['edge'],
                              s=6)
            ax[i].set_ylim([0, lim_max*1.1])
            ax[i].set_xticks(x)
            ax[i].set_xticklabels(list(v.keys()))
            ax[i].set_title(f"Effect of {k.split(sep='.')[1]} "
                            f"at DIV {k.split(sep='.')[0]}")
            ax[i].set_ylabel('Frequency [Hz]')
            ax[i].spines['top'].set_visible(False)
            ax[i].spines['bottom'].set_visible(False)
            ax[i].spines['right'].set_visible(False)
        plt.show()

    def plot_sel2_violin_multi(self):

        marker = {s: m for s, m in zip(self.tids[2].keys(),
                                       ['o', '*', '^', 'x', 'v'])}
        clrs = plt.cm.Paired.colors
        mask = [
            ('off_no',      '100', -1, clrs[1], clrs[0]),
            ('off_on',      '101',  1, clrs[1], clrs[0]),
            ('on_pass#1',   '011',  0, clrs[7], clrs[6]),
            ('on_pass#2',   '011',  1, clrs[7], clrs[6]),
            ('pass_pass#0', '111', -1, clrs[5], clrs[4]),
            ('pass_pass#1', '111',  0, clrs[5], clrs[4]),
            ('pass_pass#2', '111',  1, clrs[5], clrs[4]),
            ('pass_off#0',  '110', -1, clrs[7], clrs[6]),
            ('pass_off#1',  '110',  0, clrs[7], clrs[6]),
            ('on_off',      '010',  0, clrs[3], clrs[2]),
            ('no_on',       '001',  1, clrs[3], clrs[2])
        ]

        def name(m, n):
            return m[0].split(sep='#')[0] + '\nat ' + self.datalabels[n]

        fr = {}
        for kk, v in self.tids[2].items():
            k = kk.split(sep='_')
            fr[kk] = {}
            for m in mask:
                a = self.dlis[k[0]] + m[2]
                fr[kk][name(m, a)] = \
                    [w["freq_avg_all"] for w in v[m[1]][a].values()]

        lim_max = min(400, max([max([max(b) for b in a.values()])
                                for a in fr.values()]))
        fig = [None for _ in range(len(self.tids[1]))]
        ax = [None for _ in range(len(self.tids[1]))]
        for i, (k, v) in enumerate(fr.items()):
            kk = k.split(sep='_')
            dls = [self.datalabels[self.dlis[kk[0]] - 1], kk[0], kk[1]]
            colors = {vv: {'edge': m[3],
                           'face': m[4]} for m, vv in zip(mask, v.keys())}

            fig[i], ax[i] = plt.subplots(nrows=1, ncols=1, figsize=[14, 4])
            fig[i].canvas.draw()
            fig[i].suptitle("Spike frequencies", fontsize=14)
            x = np.array(range(1, len(v) + 1))
            vpl = ax[i].violinplot(list(v.values()),
                                   x,
                                   points=60,
                                   widths=0.4,
                                   showmeans=True,
                                   showmedians=False,
                                   showextrema=False,
                                   bw_method=0.5)
            for patch, kk in zip(vpl['bodies'], v):
                patch.set_facecolor(colors[kk]['face'])
                patch.set_edgecolor(colors[kk]['edge'])
            for (kk, vv), xx in zip(v.items(), x):
                ax[i].scatter(xx + np.random.normal(0, 0.05, len(vv)),
                              vv,
                              marker='.',
                              color=colors[kk]['edge'],
                              s=6)
            ax[i].set_ylim([0, lim_max*1.1])
            ax[i].set_xticks(x)
            ax[i].set_xticklabels(list(v.keys()))
            ax[i].set_title(f"Transitions {dls[0]} -> {dls[1]} -> {dls[2]}")
            ax[i].set_ylabel('Frequency [Hz]')
        plt.show()

    def plot_violin_diff1_multi(self, dls: List[str]):

        mask = [('before', '11', -1),
                ('after',  '11',  0)]
        fr = {}
        for k, v in self.tids[1].items():
            if k in dls:
                fr[k] = {}
                for m in mask:
                    a = self.dlis[k] + m[2]
                    fr[k][m[0]] = [w["freq_avg_all"] for w in v[m[1]][a].values()]

        colors = {'edge': plt.cm.Paired.colors[5],
                  'face': plt.cm.Paired.colors[4]}
        diff = {l: np.array(f["after"]) - np.array(f["before"])
                       for l, f in fr.items()}
        lim_min = max(-200, min([np.min(f) for f in diff.values()]))
        lim_max = min(200, max([np.max(f) for f in diff.values()]))

        for i, (k, v) in enumerate(diff.items()):
            fig, ax = plt.subplots(1, 1, figsize=(6, 4))
            fig.suptitle("Spike frequency changes: after-before.", fontsize=16)
            vpl = ax.violinplot(v,
                                [1],
                                points=60,
                                widths=0.4,
                                showmeans=True,
                                showmedians=False,
                                showextrema=False,
                                bw_method=0.5)
            for patch in vpl['bodies']:
                patch.set_facecolor(colors['face'])
                patch.set_edgecolor(colors['edge'])
            ax.scatter(np.random.normal(1, 0.07, v.size),
                       v,
                       marker='.',
                       color=colors['edge'],
                       s=6)
            ax.set_ylim([lim_min*1.1, lim_max*1.1])
            ax.set_xticks([0, 1, 2])
            ax.set_xticklabels(['', k, ''])
            ax.set_ylabel('frequency changes: after-before [Hz]')
            ax.spines['top'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['right'].set_visible(False)
            fig.show()

        print('')

    def plot_sel3_violin(self):

        mask = {s: m
                for s, m in zip(['off', 'pass_before', 'pass_after', 'on'],
                                ['10', '11', '11', '01'])}
        fr = {}
        for k, v in self.tids[1].items():
            lag = [self.dlis[k] + a for a in [-1, -1, 0, 0]]
            fr[k] = {
                l: [w["freq_avg_all"] for w in v[m][a].values()] \
                for (l, m), a in zip(mask.items(), lag)
            }

        clrs_e = [plt.cm.Paired.colors[1],
                  plt.cm.Paired.colors[5],
                  plt.cm.Paired.colors[5],
                  plt.cm.Paired.colors[3]]
        clrs_f = [plt.cm.Paired.colors[0],
                  plt.cm.Paired.colors[4],
                  plt.cm.Paired.colors[4],
                  plt.cm.Paired.colors[2]]
        colors = {c: {'edge': clrs_e[i],
                      'face': clrs_f[i]} for i, c in enumerate(mask)}
        lim_max = max([max([max(b) for b in a.values()]) for a in fr.values()])
        fig = [None for _ in range(len(self.tids[1]))]
        ax = [None for _ in range(len(self.tids[1]))]
        for i, (k, v) in enumerate(fr.items()):
            fig[i], ax[i] = plt.subplots(nrows=1, ncols=1)
            fig[i].canvas.draw()
            fig[i].suptitle("Spike frequencies", fontsize=16)
            x = np.array(range(1, len(v) + 1))
            vpl = ax[i].violinplot(list(v.values()),
                                   x,
                                   points=60,
                                   widths=0.4,
                                   showmeans=True,
                                   showmedians=False,
                                   showextrema=False,
                                   bw_method=0.5)
            for patch, c in zip(vpl['bodies'], v.keys()):
                patch.set_facecolor(colors[c]['face'])
                patch.set_edgecolor(colors[c]['edge'])
            for (kk, vv), xx in zip(v.items(), x):
                ax[i].scatter(xx + np.random.normal(0, 0.1, len(vv)),
                              vv,
                              marker='.',
                              color=colors[kk]['edge'],
                              s=6)
            ax[i].set_ylim([0, lim_max*1.1])
            xticks = ax[i].get_xticks()
            xlabels = ['' for _ in xticks]
            for xx, cn in zip(x, fr[k].keys()):
                ind = np.where(xticks == xx)[0][0]
                xlabels[ind] = f"{cn}"
            #            ax[i].set_xticks(xticks, xlabels)
            ax[i].set_xticklabels(xlabels)
            ax[i].set_title(f"Effect of {k.split(sep='.')[1]} "
                            f"at DIV {k.split(sep='.')[0]}")
            ax[i].set_ylabel('Frequency [Hz]')
        plt.show()


# =============================================================================
class BurstExaminer:
    _category_names = [
        'all',
        'inburst',
        'outburst',
        'interburst'
    ]

    Categories = Enum(
        'Categories',
        _category_names
    )

    def __init__(self,
                 dataex: InputData,
                 categories: Categories):

        self.fnames = dataex.fnames
        self.datalabels = dataex.labels2
        self.dlis = dataex.labids
        self.categories = categories

        bd1 = [maxiv.Detector(fn, maxiv.default_pars) for fn in self.fnames]
        self.bursts = [Bursts(bd).read() for bd in bd1]
        [b.set_fractions() for b in self.bursts]
        [b.set_freqs() for b in self.bursts]

        self.tids = None

    def set(self, tids):

        for n, nn in tids.items():
            for vvv in nn.values():
                for vv in vvv.values():
                    for fi, v in vv.items():
                        for ti, w in v.items():

                            fs = self.bursts[fi].freqs
                            w['burst_freq_avg'] = {
                                'all': fs['all_avg'][ti],
                                'inburst': fs['inburst_avg'][ti],
                                'outburst': fs['outburst_avg'][ti],
                                'interburst': fs['interburst_avg'][ti],
                            }

                            frs = self.bursts[fi].fractions
                            w['burst_fractions'] = {
                                'spikes_outside': frs['spikes_outside'][ti],
                                'spikes_inside': frs['spikes_inside'][ti],
                                'time_outside': frs['time_outside'][ti],
                                'time_inside': frs['time_inside'][ti],
                            }

                            inds_b = \
                                np.where(self.bursts[fi].borders[ti] == 1)[0]
                            inds_s = \
                                np.where(self.bursts[fi].borders[ti] == -1)[0]
                            szs = self.bursts[fi].sizes[ti][inds_b]
                            w['burst_size_avg'] = \
                                np.average(szs) if szs.size else 0
                            num_events = inds_b.size + inds_s.size
                            w['burst_num_fraction'] = \
                                inds_b.size / num_events if num_events else 0

        self.tids = tids

    def plot_sel1_violin_multi(self, categories: Categories = None):

        if categories is None:
            categories = self.categories
        cnames = [cat.name for cat in categories]

        mask = [('off',         '10', -1),
                ('pass_before', '11', -1),
                ('pass_after',  '11',  0),
                ('on',          '01',  0)]
        fr = {}
        for k, v in self.tids[1].items():
            fr[k] = {}
            for m in mask:
                a = self.dlis[k] + m[2]
                fr[k][m[0]] = \
                    {c: [w["burst_freq_avg"][c] for w in v[m[1]][a].values()]
                     for c in cnames}
        clrs = plt.cm.Paired.colors
        clrs_e = [clrs[1],
                  clrs[5],
                  clrs[5],
                  clrs[3]]
        clrs_f = [clrs[0],
                  clrs[4],
                  clrs[4],
                  clrs[2]]
        colors = {m[0]: {'edge': clrs_e[i],
                         'face': clrs_f[i]} for i, m in enumerate(mask)}
        lim_max = min(max([max([max([max(c) for c in b.values()])
                                for b in a.values()])
                           for a in fr.values()]), 400)
        for i, (k, v) in enumerate(fr.items()):
            for cat in cnames:
                fig, ax = plt.subplots(nrows=1, ncols=1)
                fig.canvas.draw()
                fig.suptitle(f"{k} Spike frequencies", fontsize=16)
                x = np.array(range(1, len(v) + 1))
                y = [vv[cat] for vv in v.values()]
                vpl = ax.violinplot(y,
                                    x,
                                    points=100,
                                    widths=0.4,
                                    showmeans=True,
                                    showmedians=False,
                                    showextrema=False,
                                    bw_method=0.5)
                for patch, c in zip(vpl['bodies'], v.keys()):
                    patch.set_facecolor(colors[c]['face'])
                    patch.set_edgecolor(colors[c]['edge'])
                for (kk, vv), xx in zip(v.items(), x):
                    ax.scatter(xx + np.random.normal(0., 0.06, len(vv[cat])),
                               vv[cat],
                               marker='.',
                               color=colors[kk]['edge'],
                               s=6)
                ax.set_ylim([0, lim_max*1.1])
                ax.set_xticks(x)
                ax.set_xticklabels(list(v.keys()))
                ax.set_title(cat)
                ax.set_ylabel('Frequency [Hz]')
                fig.show()
            print("")

    def plot_fractions_sel1(self):

        marker = {s: m for s, m in zip(self.tids[1].keys(),
                                       ['o', '*', '^', 'x', 'v'])}

        clrs = plt.cm.Paired.colors
        mask = [('off',        '10', -1, clrs[1]),
                ('pass_after', '11',  0, clrs[5]),
                ('on',         '01',  0, clrs[3])]

        fr = {}
        for k, v in self.tids[1].items():
            fr[k] = {}
            for m in mask:
                a = self.dlis[k] + m[2]
                fr[k][m[0]] = {'interburst_freq':
                               [w['burst_freq_avg']['interburst']
                                for w in v[m[1]][a].values()],
                               'burst_size_avg':
                               [w['burst_size_avg']
                                for w in v[m[1]][a].values()],
                               'color': m[3],
                               'marker': marker[k]}

        fig, ax = plt.subplots(1, 1)
        fig.suptitle("Bursts in single treatments.", fontsize=16)
        for k, v in fr.items():
            for a, b in v.items():
                ax.scatter(np.log10(b['burst_size_avg']),
                           b['interburst_freq'],
                           marker=b['marker'],
                           color=b['color'],
                           label=f"{k} {a}")
        ax.legend()
        ax.set_xlim([0.4, 2])
        ax.set_xlabel('Burst size (log10 # of spikes)')
        ax.set_ylabel('Burst frequency (Hz)')
        fig.show()

    def plot_fractions_sel2(self):

        marker = {s: m for s, m in zip(self.tids[2].keys(),
                                       ['o', '*', '^', 'x', 'v'])}

        clrs = plt.cm.Paired.colors
        mask = [('off_no',      '100', -1, clrs[0]),
                ('off_on',      '101',  1, clrs[1]),
                ('on_pass',     '011',  0, clrs[6]),
                ('pass_pass#0', '111', -1, clrs[4]),
                ('pass_pass#1', '111',  0, clrs[5]),
                ('pass_off',    '110',  0, clrs[7]),
                ('on_off',      '010',  0, clrs[2]),
                ('no_on',       '001',  1, clrs[3])]

        def name(m, n):
            return m[0].split(sep='#')[0] + '\nat ' + self.datalabels[n]

        fr = {}
        for kk, v in self.tids[2].items():
            k = kk.split(sep='_')
            fr[kk] = {}
            for m in mask:
                a = self.dlis[k[0]] + m[2]
                fr[kk][name(m, a)] = {
                    'interburst_freq': [w['burst_freq_avg']['interburst']
                                        for w in v[m[1]][a].values()],
                    'outburst_freq': [w['burst_freq_avg']['outburst']
                                      for w in v[m[1]][a].values()],
                    'burst_size_avg': [w['burst_size_avg']
                                       for w in v[m[1]][a].values()],
                    'burst_num_frac': [w['burst_num_fraction']
                                       for w in v[m[1]][a].values()],
                    'color': m[3],
                    'marker': marker[kk]
                }
        fig = plt.figure()
        ax = fig.add_subplot()
        fig.suptitle("Bursts in treatment pairs.", fontsize=16)
        for k, v in fr.items():
            for a, b in v.items():
                ax.scatter(np.log10(b['burst_size_avg']),
                           b['interburst_freq'],
                           marker=marker[k],
                           color=b['color'],
                           label=f"{k} {a}")
        ax.legend()
        ax.set_xlim([0.4, 2])
        ax.set_xlabel('Burst size (log10 # of spikes)')
        ax.set_ylabel('Burst frequency (Hz)')
        fig.show()


# =============================================================================
def examine_mathching_templates(
        dataext: InputData,
        pars: ClusteringParams
):
    # Find small-distance template pairs.
    match = Matcher(dataext)
    fis = list(range(match.num_files))
    chs = match.srt.mea.channels
    for i in fis[1:]:
        match.create(fis[:-i], fis[i:], chs, chs)
    match.select_related(pars)
#    match.plot()

    # Create chains of matching pairs.
    anr = PathFinder(match).apply()

    # Build a graph of the matching templates and their connections across
    # multipla data files:
    gr = Graph().create(anr.paths,
                        anr.path_dists,
                        match.electrodes,
                        anr.cc,
                        match.datalabels)
#    gr.plot(list(anr.cc.keys())[20], match.datalabels)

    # Identify sources visible across multiple data files:
    dos = Sources(dataext.treatments,
                  gr.gids_global,
                  match.datalabels,
                  dataext.labids)
#    dos.plot_as_sankey()
#    dos.plot_fractions_multi(dataext.treatments)

    # Examine single spike frequencies:
    frqs = FrequencyExaminer(dataext)
    frqs.set(dos.tids)
#    frqs.plot_sel1_violin_multi(dataext.treatments)
#    frqs.plot_violin_diff1_multi(dataext.treatments)
#    frqs.plot_sel2_violin_multi()

    # Examine burst frequencies and abundance:
    frcats = BurstExaminer.Categories
    freq_categories = [frcats.all,
                       frcats.inburst,
                       frcats.outburst,
                       frcats.interburst]
    bs = BurstExaminer(dataext,
                       freq_categories)
    bs.set(dos.tids)
#    bs.plot_sel1_violin_multi(freq_categories[:-1])
    bs.plot_fractions_sel1()
    bs.plot_fractions_sel2()
