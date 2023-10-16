from enum import Enum
from itertools import cycle
from typing import Sequence

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

# from maple.colors import extended_colortable
# from maple.colors import kelly as kelly_colors
# from maple.clustering import Params as ClusteringParams
# from maple.basic_analysis import ExtendedResults as Sorted


class ByPeakPolarity:

    def __init__(self, templates, electrodes):

        self.templates = templates
        self.electrodes = electrodes

        self.classes = None
        self.num_classes = None
        self.counts = None
        self.counts_total = None

    def assign_classes(self):

        self.classes = [np.full_like(el, fill_value=-1) for el in
                        self.electrodes]

        for el, tmpl, cl in zip(self.electrodes, self.templates, self.classes):
            for ti in range(tmpl.shape[2] // 2):
                tp = np.squeeze(tmpl[el[ti], :, ti])
                extrtp, extrinds = self.find_extrema(tp)
                cl[ti] = self.classify(extrtp, extrinds)

        return self.classes

    @staticmethod
    def find_extrema(tp):

        sntp = np.sign(np.diff(tp))
        indp = []
        for i in range(1, sntp.size):
            if (sntp[i - 1] == -1 and sntp[i] == 1) or \
                    (sntp[i - 1] == 1 and sntp[i] == -1):
                indp.append(i)
        ars = np.argsort(tp)
        indps = [a for a in ars if a in indp]
        pi = np.array(indps)[[0, 1, -2, -1]]
        tpi = tp[pi]
        extr = np.argsort(np.abs(tpi))[-2:]

        return tpi[extr], pi[extr]

    @staticmethod
    def classify(val, ind):

        if val[0] < 0 and val[1] < 0:
            return 0
        if val[0] >= 0 and val[1] >= 0:
            return 1
        if (val[0] < 0 <= val[1]) and (ind[0] < ind[1]) or \
                (val[0] >= 0 > val[1]) and (ind[0] > ind[1]):
            return 2
        if (val[0] < 0 <= val[1]) and (ind[0] > ind[1]) or \
                (val[0] >= 0 > val[1]) and (ind[0] < ind[1]):
            return 3

        raise Exception("Template not classifiable.")

    def plot_template(self, fi, ti):

        tp = np.squeeze(self.templates[fi][self.electrodes[fi][ti], :, ti])
        extrtp, extrinds = self.find_extrema(tp)

        ax = plt.subplot()
        ax.plot(range(tp.size), tp)
        ax.scatter(extrinds, extrtp)
        plt.show()

    def set_counts(self):

        from collections import Counter

        _counts = [Counter(tc) for tc in self.classes]
        self.set_num_classes()

        self.counts = [np.zeros(self.num_classes) for _ in self.classes]
        for fi in range(len(self.classes)):
            for k, v in _counts[fi].items():
                self.counts[fi][k] = v
        self.counts_total = np.zeros(self.num_classes)
        for s in _counts:
            for k, v in s.items():
                self.counts_total[k] += v

    def set_num_classes(self):

        self.num_classes = max([np.max(tc) for tc in self.classes]) + 1

    @classmethod
    def apply(cls, templates, electrodes):

        classifier = cls(templates, electrodes)
        classes = classifier.assign_classes()
        classifier.set_counts()

        return classes, classifier.num_classes

