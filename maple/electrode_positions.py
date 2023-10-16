import importlib
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

import matplotlib.patches as mpatches

from maple.colors import kelly as kelly_colors


class MeaGeometry:

    total_nb_channels : int
    indexes : list
    channels : list
    labels : list
    label_channel_map : dict
    radius : float
    positions_x : list
    positions_y : list
    ncols : int
    nrows : int
    minx : float
    maxx : float
    miny : float
    maxy : float
    width : float
    height : float
    step_x : float
    step_y : float

    @classmethod
    def initialize(cls, name: str):
        mea = importlib.import_module('mea_' + name)

        cls.total_nb_channels = mea.total_nb_channels
        cls.radius = mea.radius / 10
        geometry = mea.channel_groups[1]["geometry"]
        cls.indexes = [a for a in geometry.keys()]
        cls.positions_x = [k[0] for k in geometry.values()]
        cls.positions_y = [k[1] for k in geometry.values()]
        cls.channels = mea.channel_groups[1]["channels"]
        if name == '256':
            cls.labels = cls.channels
        elif name in ['tu60', 'tu60_r15']:
            cls.labels = mea.electrode_index_map
        assert len(cls.labels) == len(cls.channels)
        nx = Counter(cls.positions_x)
        cls.ncols = Counter(cls.positions_x).most_common(1)[0][1]
        cls.nrows = Counter(cls.positions_y).most_common(1)[0][1]
        cls.minx = min(cls.positions_x)
        cls.maxx = max(cls.positions_x)
        cls.miny = min(cls.positions_y)
        cls.maxy = max(cls.positions_y)
        cls.width = cls.maxx - cls.minx
        cls.height = cls.maxy - cls.miny
        cls.step_x = cls.width / (cls.ncols - 1)
        cls.step_y = cls.height / (cls.nrows - 1)
        cls.width = cls.step_x * (cls.ncols - 1)
        cls.height = cls.step_y * (cls.nrows - 1)

        return cls

    @classmethod
    def row_col(cls, ind):
        assert ind in cls.indexes
        s = str(cls.labels[ind])
        return int(s[1]) - 1, int(s[0]) - 1

    @classmethod
    def plot_simple(cls,
                    colors,
                    labels=None,
                    sizes=None,
                    title=None,
                    legend=None,
                    legend_handles=None):

        fig = plt.figure(figsize=(10., 10.))
        a = fig.add_subplot(111)
        a.set_aspect('equal')
        a.axis('off')
        if sizes is None:
            sizes = [cls.radius]*len(cls.channels)
        if labels is None:
            labels = cls.labels
        else:
            assert len(labels) == len(cls.channels)
        for ch, label in zip(cls.channels, labels):
            border_x, border_y = cls.step_x*0.5, cls.step_y*0.5

            a.set_xlim(cls.minx-border_x, cls.maxx+border_x)
            a.set_ylim(cls.miny-border_y, cls.maxy+border_y)
            center = (cls.positions_x[ch], cls.positions_y[ch])
            circle = plt.Circle(center,
                                radius=sizes[ch],
                                facecolor=colors[ch],
                                edgecolor='k')
            a.add_patch(circle)
            a.annotate(label,
                       xy=[cls.positions_x[ch], cls.positions_y[ch]],
                       fontsize=8,
                       ha="center",
                       va="center")
        if legend is not None:
            legend_handles = \
                [legend_handles.append(mpatches.Patch(color=v, label=k))
                 for k, v in legend.items()]
            a.legend(handles=legend_handles,
                     bbox_to_anchor=(1.05, 1),
                     loc='upper left',
                     borderaxespad=0.)
        elif legend_handles is not None:
            a.legend(handles=legend_handles,
                     bbox_to_anchor=(1.05, 1),
                     loc='upper left',
                     borderaxespad=0.)
        if title is not None:
            fig.suptitle(title, fontsize=16)

        plt.show()

    @classmethod
    def plot_time(cls, template_electrodes=None):

        if template_electrodes is not None:
            tcounts = Counter(template_electrodes)
            template_maxcount = tcounts.most_common(1)[0][1]
            color_templ = 1. / template_maxcount
        color_empty = [0.8, 0.8, 0.8]
        fig = plt.figure(figsize=(10., 10.))
        a = fig.add_subplot(111)
        a.set_aspect('equal')
        for ch, label in zip(cls.channels, cls.labels):
            border_x, border_y = cls.step_x * 0.5, cls.step_y * 0.5

            a.set_xlim(cls.minx - border_x, cls.maxx + border_x)
            a.set_ylim(cls.miny - border_y, cls.maxy + border_y)
            center = (cls.positions_x[ch], cls.positions_y[ch])
            circle = plt.Circle(center, radius=cls.radius)
            if template_electrodes is None or \
                    (template_electrodes is not None and tcounts[ch] == 0):
                circle.set_color(color_empty)
            else:
                circle.set_color([1., 0., 0., tcounts[ch] * color_templ])
            a.add_patch(circle)
            a.annotate(label,
                       xy=[cls.positions_x[ch], cls.positions_y[ch]],
                       fontsize=8,
                       ha="center",
                       va="center")
        plt.show()

    @classmethod
    def plot_electrode_geometry(cls,
                                template_electrodes=None,
                                template_colors=None):
        from itertools import cycle
        template_colors = cycle(kelly_colors.values())
        ax_width, ax_height = 1 / cls.ncols, 1 / cls.nrows
        fig = plt.figure(figsize=(10, 10))
        axs = []
        left = np.empty(len(cls.channels), dtype=float)
        bottom = np.empty(len(cls.channels), dtype=float)
        for elind in cls.channels:
            left[elind] = (cls.positions_x[elind] - cls.minx) / cls.width
            bottom[elind] = (cls.positions_y[elind] - cls.miny) / cls.height
            a = fig.add_axes([left[elind], bottom[elind], ax_width, ax_height])
            a.set_aspect('equal')
            a.axis('off')
            a.set_xlim(0, cls.step_x)
            a.set_ylim(0, cls.step_y)
            center = (cls.step_x/2, cls.step_y/2)
            radius = cls.radius

            if template_electrodes is None:
                circle = plt.Circle(center, radius=radius)
                circle.set_color([0.3, 0.3, 0.3])
                a.add_patch(circle)

            else:
                templs = np.argwhere(template_electrodes == elind)
                nc = len(templs)
                if nc > 1:
                    shift = 2 * radius
                    for ci in range(nc):
                        angle = ci * 2 * np.pi / nc
                        circle = plt.Circle(
                            (center[0] + shift * np.cos(angle),
                             center[1] + shift * np.sin(angle)),
                            radius=radius,
                            color=np.array(next(template_colors))/256
                        )
                        a.add_patch(circle)
                elif nc == 1:
                    circle = plt.Circle(
                        center,
                        radius=radius,
                        color=np.array(next(template_colors))/256
                    )
                    a.add_patch(circle)
                else:
                    circle = plt.Circle(center, radius=radius)
                    circle.set_ec([0.3, 0.3, 0.3])
                    circle.set_fc('w')
                    a.add_patch(circle)

            axs.append(a)
            annot = a.annotate(f"{elind}", xy=(0, 0), xytext=(4, 4),
                               textcoords="offset points", size=7)
            annot.set_visible(True)
        plt.show()

    @classmethod
    def plot_network_geomerty(cls,
                              node_face_colors,
                              arrow_colors,
                              face_color_empty=None,
                              labels=None,
                              title=None,
                              legend=None,
                              legend_handles=None,
                              cbarmap=None):
        edge_color = [0.3, 0.3, 0.3]
        linewidth = 0.5
        radius = 0.75*cls.radius
        fig = plt.figure(figsize=(10., 10.))
        a = fig.add_subplot(111)
        a.set_aspect('equal')
        assert len(node_face_colors) == len(cls.channels)
        if labels is not None:
            assert len(labels) == len(cls.channels)
        for ch, clr in zip(cls.channels, node_face_colors):
            if labels is not None:
                label = labels[ch]
                assert len(label) == len(clr)
            center_el = (cls.positions_x[ch], cls.positions_y[ch])
            if len(clr) == 0:
                if face_color_empty is not None:
                    circle = plt.Circle(center_el,
                                        radius=radius,
                                        facecolor=face_color_empty,
                                        edgecolor=edge_color,
                                        lw=linewidth
                                        )
                    a.add_patch(circle)
            else:
                inds = [i for i, c in enumerate(clr)
                        if c is not None and len(c)]
                shift = 2 * radius if len(clr) > 1 else 0
                for ci in inds:
                    angle = ci * 2 * np.pi / len(clr)
                    pos = (center_el[0] + shift * np.cos(angle),
                           center_el[1] + shift * np.sin(angle))
                    circle = plt.Circle(pos,
                                        radius=radius,
                                        facecolor=clr[ci],
                                        edgecolor=edge_color,
                                        lw=linewidth)
                    a.add_patch(circle)
                    if labels is not None:
                        a.annotate(label[ci],
                                   xy=pos,
                                   fontsize=8,
                                   ha="center", va="center")
        border_x, border_y = cls.step_x * 0.5, cls.step_y * 0.5
        a.set_xlim(cls.minx - border_x, cls.maxx + border_x)
        a.set_ylim(cls.miny - border_y, cls.maxy + border_y)
        a.axis('off')
        if legend is not None:
            legend_handles = \
                [mpatches.Patch(color=v, label=k) for k, v in legend.items()]
            a.legend(handles=legend_handles,
                     bbox_to_anchor=(1.05, 1),
                     loc='upper left',
                     borderaxespad=0.)
        elif legend_handles is not None:
            a.legend(handles=legend_handles,
                     bbox_to_anchor=(1.05, 1),
                     loc='upper left',
                     borderaxespad=0.)

        if title is not None:
            fig.suptitle(title)
        if cbarmap is not None:
            fig1, ax1 = plt.subplots(figsize=(8., 1.))
            fig1.subplots_adjust(bottom=0.5)
            cbar = fig1.colorbar(cbarmap,
                                 cax=ax1,
                                 orientation='horizontal',
                                 label='Fraction of excitatory out-connections')
        plt.show()

    @classmethod
    def plot_electrode_geometry_single_ax(cls,
                                          face_colors,
                                          face_color_empty=None,
                                          labels=None,
                                          title=None,
                                          legend=None,
                                          legend_handles=None):
        edge_color = [0.3, 0.3, 0.3]
        linewidth = 0.5
        radius = 0.75*cls.radius
        fig = plt.figure(figsize=(10., 10.))
        a = fig.add_subplot(111)
        a.set_aspect('equal')
        assert len(face_colors) == len(cls.channels)
        if labels is not None:
            assert len(labels) == len(cls.channels)
        for ch, clr in zip(cls.channels, face_colors):
            if labels is not None:
                label = labels[ch]
                assert len(label) == len(clr)
            center_el = (cls.positions_x[ch], cls.positions_y[ch])
            if len(clr) == 0:
                if face_color_empty is not None:
                    circle = plt.Circle(center_el,
                                        radius=radius,
                                        facecolor=face_color_empty,
                                        edgecolor=edge_color,
                                        lw=linewidth
                                        )
                    a.add_patch(circle)
            else:
                inds = [i for i, c in enumerate(clr) if c is not None and len(c)]
                shift = 2 * radius if len(clr) > 1 else 0
                for ci in inds:
                    angle = ci * 2 * np.pi / len(clr)
                    pos = (center_el[0] + shift * np.cos(angle),
                           center_el[1] + shift * np.sin(angle))
                    circle = plt.Circle(pos,
                                        radius=radius,
                                        facecolor=clr[ci],
                                        edgecolor=edge_color,
                                        lw=linewidth)
                    a.add_patch(circle)
                    if labels is not None:
                        a.annotate(label[ci],
                                   xy=pos,
                                   fontsize=8,
                                   ha="center", va="center")
        border_x, border_y = cls.step_x * 0.5, cls.step_y * 0.5
        a.set_xlim(cls.minx - border_x, cls.maxx + border_x)
        a.set_ylim(cls.miny - border_y, cls.maxy + border_y)
        a.axis('off')
        if legend is not None:
            legend_handles = \
                [mpatches.Patch(color=v, label=k) for k, v in legend.items()]
            a.legend(handles=legend_handles,
                     bbox_to_anchor=(1.05, 1),
                     loc='upper left',
                     borderaxespad=0.)
        elif legend_handles is not None:
            a.legend(handles=legend_handles,
                     bbox_to_anchor=(1.05, 1),
                     loc='upper left',
                     borderaxespad=0.)

        if title is not None:
            fig.suptitle(title)
        plt.show()

        print("")

    @classmethod
    def plot_electrode_geometry_single_ax_orig(cls,
                                               template_electrodes=None,
                                               title=None):
        from itertools import cycle
        template_colors = cycle(kelly_colors.values())

        color_empty = [1., 1., 1.]
        edge_color = [0.3, 0.3, 0.3]
        linewidth = 0.5
        radius = 0.75 * cls.radius
        fig = plt.figure(figsize=(10., 10.))
        a = fig.add_subplot(111)
        a.set_aspect('equal')
        for ch, label in zip(cls.channels, cls.labels):
            if template_electrodes is not None:
                templs = np.argwhere(template_electrodes == ch)
                nc = len(templs)
            center_el = (cls.positions_x[ch], cls.positions_y[ch])
            if template_electrodes is None or \
                    (template_electrodes is not None and nc == 0):
                circle = plt.Circle(center_el,
                                    radius=radius,
                                    facecolor=color_empty,
                                    edgecolor=edge_color,
                                    lw=linewidth
                                    )
                a.add_patch(circle)
            else:
                shift = 2 * radius if nc > 1 else 0
                for ci in range(nc):
                    angle = ci * 2 * np.pi / nc
                    circle = plt.Circle(
                        (center_el[0] + shift * np.cos(angle),
                         center_el[1] + shift * np.sin(angle)),
                        radius=radius,
                        facecolor=np.array(next(template_colors)) / 256,
                        edgecolor=edge_color,
                        lw=linewidth
                    )
                    a.add_patch(circle)
        border_x, border_y = cls.step_x * 0.5, cls.step_y * 0.5
        a.set_xlim(cls.minx - border_x, cls.maxx + border_x)
        a.set_ylim(cls.miny - border_y, cls.maxy + border_y)
        a.axis('off')
        if title is not None:
            fig.suptitle(title)
        plt.show()

    @classmethod
    def make_label_channel_map(cls):
        cls.label_channel_map = {}
        for i, e in zip(cls.channels, cls.labels):
            cls.label_channel_map[e] = i
        return cls.label_channel_map
