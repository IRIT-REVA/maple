from collections import Counter
from itertools import cycle

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import sklearn.cluster as skcl
from sklearn import metrics

from maple.basic_analysis import ExtendedResults as Sorted
from maple.colors import extended_colortable
from maple.template_matching.by_peak_polarity \
    import ByPeakPolarity as ClassifierByPeakPolarity


# =============================================================================
class Params:

    def __init__(self,
                 method: str,
                 eps: float,
                 min_samples: int):

        self.method = method
        self.eps = eps
        self.min_samples = min_samples

    def __str__(self):

        s = f"method: {self.method} \n"
        s += f"eps: {self.eps}, min_samples: {self.min_samples}" \
            if self.method == 'DBSCAN' else \
            f"min_samples: {self.min_samples}"

        return s


# =============================================================================
class Clusterizer:

    def __init__(self, dname, data):

        self.data = data
        self.dname = dname
        self.res = {}

    def apply(self, pars : Params):

        if pars.method == 'DBSCAN':
            db = skcl.DBSCAN(eps=pars.eps, min_samples=pars.min_samples) \
                     .fit(self.data)
        elif pars.method == 'OPTICS':
            db = skcl.OPTICS(eps=pars.eps, min_samples=pars.min_samples) \
                     .fit(self.data)
        else:
            BaseException("Clustering method is not implemented")

        self.res = self._set_statistics(db.labels_)

    @staticmethod
    def _set_statistics(labels, res=None):

        if res is None:
            res = {}

        res['n_items'] = len(labels)
        res['n_clusters'] = len(set(labels)) - (1 if -1 in labels else 0)

        cnts = Counter(labels)
        mc = [s for s in cnts.most_common() if s[0] != -1]

        # Rename labels sorting by cluster size:
        mc0 = sorted([s[0] for s in mc])
        mc1 = [s[1] for s in mc]
        res['n_outsiders'] = cnts[-1]
        res['n_clustered'] = res['n_items'] - res['n_outsiders']
        res['cluster_counts'] = \
            {k: v for k, v in zip(mc0, mc1)}
        labels_ordered = np.empty_like(labels)
        labels_ordered[np.where(labels == -1)[0]] = -1
        for i, j in zip(mc0, mc):
            labels_ordered[np.where(labels == j[0])[0]] = i
        res['labels'] = labels_ordered
        for i, j in enumerate(mc):
            labels_ordered[np.where(labels == j[0])[0]] = i
        res['labels_'] = labels_ordered

        return res

    def print(self, val):

        print(f"Cluster counts for {self.dname}:")
        units = 'cluster' if val['n_clusters'] == 1 else 'clusters'
        print(f"{val['n_items']} items are assigned to "
              f"{val['n_clusters']} {units}:")
        for k, v in val['cluster_counts'].items():
            print(f"  cluster {k}: {v}")
        print(f"  outsiders: {val['n_outsiders']}")

    def set_colors(self, colortable, color_outsider, res=None):

        if res is None:
            res = self.res

        n_clusters = self.res['n_clusters']

        res['color_outsider'] = tuple(color_outsider)
        res['colortable'] = colortable
        cc = cycle(colortable)
        if n_clusters > len(colortable):
            print(f"Warning: number of clusters {n_clusters} "
                  f"exceeds the number of available colors.")

        res['unique_colors'] = \
            [next(cc) for c in range(n_clusters)]
        res['colors'] = \
            [res['unique_colors'][c] if c != -1 else color_outsider
             for c in res['labels']]

        res['legend_handles'] = \
            [mpatches.Patch(color=res['unique_colors'][c],
                            label=str(c))
             for c in range(n_clusters)]
        res['legend_handles'].append(mpatches.Patch(color=color_outsider,
                                                    label='outsider'))

        return res

    def extract(self, mask):

        r = self._set_statistics(self.res['labels'][mask])
        r = self.set_colors(self.res['colortable'],
                            self.res['color_outsider'],
                            r)
        return r


# =============================================================================
# =============================================================================
def cluster_electrodes_by_freq_diff(names: dict,
                                    pars: Params):

    data_names = list(names.keys())
    fnames = names.values()
    numfiles = len(names)

    srt = Sorted(fnames,
                 fields=[Sorted.Fields.params,
                         Sorted.Fields.amplitudes,
                         Sorted.Fields.electrodes,
                         Sorted.Fields.freqs,
                         Sorted.Fields.templates])
    N_e = [pars.getint('data', 'N_e') for pars in srt.params][0]  # !!!!!

    templ_classes, num_templ_classes = \
        ClassifierByPeakPolarity.apply(srt.templates_np, srt.electrodes)

    frf = [[[[]
             for _ in range(numfiles)]
            for _ in range(N_e)]
           for _ in range(num_templ_classes)]
    for si in range(numfiles):
        for tik in srt.freqs[si].keys():
            ti = int(tik)
            e = srt.electrodes[si][ti]
            cl = templ_classes[si][ti]
            frf[cl][e][si].append(srt.freqs[si][tik])

    frf = np.array([[[np.average(np.concatenate(f)) if len(f) else 0 \
                      for f in fs] \
                     for fs in fse] \
                    for fse in frf])

    frfc = frf.transpose(2, 1, 0)
    frfcda = {}
    frfcdr = {}
    for si in range(1, numfiles):
        for sii in range(si):
            k = f"{si}_{sii}"
            frfcda[k] = frfc[sii,:,:] - frfc[si,:,:]
            a = frfc[sii,:,:] + frfc[si,:,:]
            frfcdr[k] = np.zeros_like(frfc[si,:,:])
            i,j = np.where(a)
            frfcdr[k][i,j] = (frfc[sii,i,j] - frfc[si,i,j]) / a[i,j]

    amp = [[[[]
             for _ in range(numfiles)]
            for _ in range(N_e)]
           for _ in range(num_templ_classes)]
    for si in range(numfiles):
        for tik in srt.freqs[si].keys():
            ti = int(tik)
            e = srt.electrodes[si][ti]
            cl = templ_classes[si][ti]
            amp[cl][e][si].append(srt.amplitudes[si][tik])

    amp = np.array([[[np.concatenate(f).sum() if len(f) else 0.
                      for f in fs]
                     for fs in fse]
                    for fse in amp])
    amp /= np.max(np.max(np.max(amp)))

    res_frfcda = {}
    for k, v in frfcda.items():
        ii = k.split(sep='_')
        dns = f"{data_names[int(ii[0])]} - {data_names[int(ii[1])]}"
        data2cluster = v
        cz = Clusterizer(dns, data2cluster)
        cz.apply(pars)
        res_frfcda[dns] = cz.set_colors(extended_colortable,
                                        color_outsider=(0., 0., 0.))
        cz.print(res_frfcda[dns])

    res_frfcdr = {}
    for k, v in frfcdr.items():
        ii = k.split(sep='_')
        dns = f"{data_names[int(ii[0])]} - {data_names[int(ii[1])]}"
        data2cluster = v
        cz = Clusterizer(dns, data2cluster)
        cz.apply_dbscan(pars)
        res_frfcdr[dns] = cz.set_colors(extended_colortable,
                                        color_outsider=(0., 0., 0.))

        cz.print(res_frfcdr[dns])

    return res_frfcda, res_frfcdr


def cluster_electrodes_by_freq(names: dict,
                               pars: Params):

    numfiles = len(names)
    fnames = list(names.values())
    srt = Sorted(fnames,
                 fields=[Sorted.Fields.params,
                         Sorted.Fields.amplitudes,
                         Sorted.Fields.electrodes,
                         Sorted.Fields.freqs,
                         Sorted.Fields.templates])

    N_e = [pars.getint('data', 'N_e') for pars in srt.params][0]

    templ_classes, num_templ_classes = \
        ClassifierByPeakPolarity.apply(srt.templates_np, srt.electrodes)

    frf = [[[[] for _ in range(numfiles)]
            for _ in range(N_e)]
           for _ in range(num_templ_classes)]
    for si in range(numfiles):
        for tik in srt.freqs[si].keys():
            ti = int(tik)
            e = srt.electrodes[si][ti]
            cl = templ_classes[si][ti]
            frf[cl][e][si].append(srt.freqs[si][tik])

    frf = np.array([[[np.average(np.concatenate(f)) if len(f) else 0 \
                      for f in fs] \
                     for fs in fse] \
                    for fse in frf])

    frfc = frf.transpose(2,1,0)
    amp = [[[[] for _ in range(numfiles)]
            for _ in range(N_e)]
           for _ in range(num_templ_classes)]
    for si in range(numfiles):
        for tik in srt.freqs[si].keys():
            ti = int(tik)
            e = srt.electrodes[si][ti]
            cl = templ_classes[si][ti]
            amp[cl][e][si].append(srt.amplitudes[si][tik])

    amp = np.array([[[np.concatenate(f).sum() if len(f) else 0. \
                      for f in fs] \
                     for fs in fse] \
                    for fse in amp])
    amp /= np.max(np.max(np.max(amp)))

    res = {}
    for si, dname in enumerate(names.keys()):

        data2cluster = frfc[si,:,:]
        cz = Clusterizer(dname, data2cluster)
        cz.apply(pars)
        res[dname] = cz.set_colors(extended_colortable,
                                   color_outsider=(0., 0., 0.))

        cz.print(res[dname])

        print("Silhouette Coefficient: %0.3f"
              % (metrics.silhouette_score(data2cluster, res[dname]['labels'])
                 if Counter(res[dname]['labels'])[0] < N_e
                 else -1.))

    return res


def cluster_templates_by_freq(names: dict,
                              pars: Params):

    srt = Sorted(names.values(), fields=[Sorted.Fields.freqs])
    freq_avg = [{k: np.average(v) for k, v in st.items()} for st in srt.freqs]

    res = {}
    for si, dname in enumerate(names.keys()):

        data2cluster = np.array(list(freq_avg[si].values()))
        data2cluster = data2cluster.reshape(-1, 1)
        cz = Clusterizer(dname, data2cluster)
        cz.apply(pars)
        res[dname] = cz.set_colors(extended_colortable,
                                   color_outsider=(0., 0., 0.))
        cz.print(res[dname])

        print("Silhouette Coefficient: %0.3f"
              % (metrics.silhouette_score(data2cluster, res[dname]['labels'])
                 if Counter(res[dname]['labels'])[0] < len(res[dname]['labels'])
                 else -1.)
              )

    return res


def cluster_templates_by_freq_globally(names, pars: Params):

    srt = Sorted(names.values(),
                 fields=[Sorted.Fields.spike_times,
                         Sorted.Fields.isis,
                 Sorted.Fields.freqs])
    freq_avg = [{k: np.average(v) for k, v in st.items()} for st in srt.freqs]
    for ist, st in enumerate(srt.freqs):
        for k, v in st.items():
            if v.any() == np.nan:
                print(f" {ist}: {k}")
    freq_avg_glob = np.concatenate([list(fa.values()) for fa in freq_avg])
    mask = np.concatenate([np.full(len(freq_avg[i]), i)
                           for i in range(len(freq_avg))])

    res_g = {}
    data2cluster = freq_avg_glob.reshape(-1, 1)
    cz = Clusterizer('global', data2cluster)
    cz.apply(pars)
    res_g = cz.set_colors(extended_colortable,
                          color_outsider=(0., 0., 0.))
    cz.print(res_g)

    print("Silhouette Coefficient: %0.3f"
          % (metrics.silhouette_score(data2cluster, res_g['labels'])
             if Counter(res_g['labels'])[0] < len(res_g['labels'])
             else -1.)
          )
    separators = np.cumsum([len(st) for st in srt.spike_times])[:-1]
    plot(res_g, data2cluster, separators)

    res = {}
    for si, dname in enumerate(names.keys()):
        res[dname] = cz.extract(np.where(mask == si)[0])
        plot(res[dname], list(freq_avg[si].values()))

    return res


def plot(clusters, data, separators=None):

    unique_labels = set(clusters['labels'])
    clusterized = np.where(clusters['labels'] != -1)[0]
    outsiders = np.where(clusters['labels'] == -1)[0]
    data = np.squeeze(data)
    datandim = data.ndim

    fig = plt.figure()
    if datandim == 1:
        ax = fig.add_subplot()
        ax.scatter(
            clusterized,
            data[clusterized],
            marker="o",
            facecolor=np.array(clusters['colors'])[clusterized],
            edgecolor="k",
            s=34
        )
        ax.scatter(
            outsiders,
            data[outsiders],
            marker="o",
            facecolor=clusters['color_outsider'],
            edgecolor="k",
            s=6
        )
        ylim = ax.get_ylim()
        if separators is not None:
            ax.plot([separators, separators], ylim, c='k')

    elif datandim > 2:
        ax = fig.add_subplot(projection='3d')
        for k, col in zip(unique_labels, clusters['colors']):

            class_member_mask = clusters['labels'] == k

            xy = data[class_member_mask & ~outsiders]
            ax.scatter(
                xy[:, 0],
                xy[:, 1],
                xy[:, 2],
                "o",
                facecolor=col,
                edgecolor="k",
                s=34,
            )

            xy = data[class_member_mask & outsiders]
            ax.scatter(
                xy[:, 0],
                xy[:, 1],
                xy[:, 2],
                "o",
                facecolor=clusters['color_outsider'],
                edgecolor="k",
                s=6,
            )

    fig.suptitle(f"Estimated number of clusters: {clusters['n_clusters']}")
    fig.show()


def cluster_templates_by_dist(d1d: np.ndarray,
                              pars: Params,
                              colortable=extended_colortable):

    data2cluster = d1d.reshape(-1, 1)
    cz = Clusterizer('global', data2cluster)
    cz.apply(pars)
    res = cz.set_colors(colortable,
                        color_outsider=(0., 0., 0.))
    cz.print(res)

    print("Silhouette Coefficient: %0.3f"
          % (metrics.silhouette_score(data2cluster, res['labels'])
             if Counter(res['labels'])[0] < len(res['labels'])
             else -1.)
          )
#    plot(res, data2cluster)

    return res


def examine_electrodes_by_templ_freq_clusters_classes_globally(
        names,
        mea_name: str,
        pars: Params):

    from visualizing import plot_template_freq_clusters_classes_per_electrode

    clusters = \
        cluster_templates_by_freq_globally(names, pars)

    plot_template_freq_clusters_classes_per_electrode(
        names.values(),
        mea_name,
        clusters,
        str(pars)
    )


def examine_electrodes_by_templ_freq_clusters_classes(names,
                                                      mea_name: str,
                                                      pars: Params):

    from visualizing import plot_template_freq_clusters_classes_per_electrode

    clusters = cluster_templates_by_freq(names, pars)

    plot_template_freq_clusters_classes_per_electrode(
        names.values(),
        mea_name,
        clusters,
        str(pars)
    )


def examine_cluster_templates_by_freq(names,
                                      mea_name: str,
                                      pars: Params):

    from visualizing import plot_clusters_per_electrode_detailed

    clusters = cluster_templates_by_freq(names, pars)

    [plot_clusters_per_electrode_detailed(
        mea_name,
        c,
        fn,
        f"{dn}: electrodes clustered by frequency classes \n"
        f"{pars}")
     for dn, c, fn in zip(clusters.keys(), clusters.values(), names.values())]


def examine_cluster_electrodes_by_freq(names,
                                       mea_name: str,
                                       pars: Params):

    from visualizing import plot_clusters_per_electrode_simple

    clusters = cluster_electrodes_by_freq(names, pars)

    [plot_clusters_per_electrode_simple(
        mea_name,
        c,
        f"{name}: electrodes clustered by frequency classes \n"
        f"{pars}")
     for name, c in clusters.items()]


def examine_cluster_electrodes_by_freq_diff(names,
                                            mea_name: str,
                                            pars: Params):

    from visualizing import plot_clusters_per_electrode_simple

    clusters_a, clusters_r = cluster_electrodes_by_freq_diff(names, pars)

    [plot_clusters_per_electrode_simple(
        mea_name,
        c,
        f"{name}: electrodes clustered by frequency differences \n"
        f"{pars}")
     for name, c in clusters_a.items()]

    [plot_clusters_per_electrode_simple(
        mea_name, c,
        f"{name}: electrodes clustered by relative frequency differences \n"
        f"{pars}")
     for name, c in clusters_r.items()]
