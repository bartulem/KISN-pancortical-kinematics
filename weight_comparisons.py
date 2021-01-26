# -*- coding: utf-8 -*-

"""

@author: bartulem

Compare tuning-curve rate differences in weight/no-weight sessions.

"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kde
from scipy.stats import ttest_rel
from make_ratemaps import Ratemap

def extract_json_data(json_file='', weight=False, features=None,
                      weight_stability_bound=True):
    """
    Description
    ----------
    This function extracts json data (weight/tuning peaks/stability)
    and returns it packed into a dictionary.
    ----------

    Parameters
    ----------
    **kwargs (dictionary)
    json_file (str)
        Absolute path to the .json file of interest.
    weight (bool)
        Yey or ney on the 'weight' data; defaults to False.
    features (list)
        List of features you're interested in (der not necessary!); defaults to ['Speeds'].
    weight_stability_bound (int / bool)
        How much is the OF1 peak rate allowed to deviate from OF2; defaults to True.
    ----------

    Returns
    ----------
    data_dict (dict)
        A dictionary which contains the desired data.
    ----------
    """

    if features is None:
        features = ['Speeds']

    with open(json_file) as j_file:
        json_data = json.load(j_file)

    if weight:
        weight_dict = {}
        for feature in features:
            weight_dict[feature] = {'light1-light2': [], 'light1-weight': []}
            weight_dict[f'{feature}_1st_der'] = {'light1-light2': [], 'light1-weight': []}

    for cl_num in json_data.keys():
        if weight:
            for key in json_data[cl_num].keys():
                if key in weight_dict.keys():
                    if weight_stability_bound is True or abs(json_data[cl_num][key]['light1-light2']) < weight_stability_bound:
                        weight_dict[key]['light1-light2'].append(json_data[cl_num][key]['light1-light2'])
                        weight_dict[key]['light1-weight'].append(json_data[cl_num][key]['light1-weight'])

    if weight:
        return weight_dict



class WeightComparer:

    def __init__(self, weight_json_file='', chosen_features=None,
                 weight_stability_bound=True, save_dir='',
                 save_fig=False, fig_format='png'):
        if chosen_features is None:
            chosen_features = ['Speeds']
        self.weight_json_file = weight_json_file
        self.chosen_features = chosen_features
        self.weight_stability_bound = weight_stability_bound
        self.save_dir=save_dir
        self.save_fig=save_fig
        self.fig_format=fig_format

    def plot_weight_features(self, **kwargs):
        """
        Description
        ----------
        This method plots the raw distribution of light1-light2 and
        light1-weight peak rate differences, also in form of a
        scatter plot and probability density distribution.
        ----------

        Parameters
        ----------
        **kwargs (dictionary)
        weight_stability_bound (int / bool)
            How much is the OF1 peak rate allowed to deviate from OF2; defaults to True.
        min_max_range (bool)
            If True, finds the min/max of differences in the data, otherwise sets it to -[-20,20]; defaults to False.
        ----------

        Returns
        ----------
        weight_feature_distributions (plot)
            A plot with weight peak rate differences distributions.
        ----------
        """

        min_max_range = kwargs['min_max_range'] if 'min_max_range' in kwargs.keys() and type(kwargs['min_max_range']) == bool else False

        weight_dict = extract_json_data(json_file=self.weight_json_file,
                                        weight=True,
                                        features=self.chosen_features,
                                        weight_stability_bound=self.weight_stability_bound)

        for chosen_feature in self.chosen_features:
            if min_max_range:
                min_joint = int(np.floor(np.min(weight_dict[chosen_feature]['light1-light2'] + weight_dict[chosen_feature]['light1-weight']) / 10.0)) * 10
                max_joint = int(np.ceil(np.max(weight_dict[chosen_feature]['light1-light2'] + weight_dict[chosen_feature]['light1-weight']) / 10.0)) * 10
            else:
                min_joint=-20
                max_joint=20

            hist_color = [val for key, val in Ratemap.feature_colors.items() if key in chosen_feature][0]
            hist_bins = np.arange(min_joint, max_joint + 1, 1)
            feature_der = f'{chosen_feature}_1st_der'

            fig = plt.figure()
            gs1 = fig.add_gridspec(nrows=3, ncols=2, left=.075, right=.505,
                                   wspace=0.1, hspace=0.5)
            ax1 = fig.add_subplot(gs1[:-1, :])
            ax1.hist(weight_dict[chosen_feature]['light1-light2'], bins=hist_bins, color=hist_color, edgecolor='#000000', alpha=.2)
            ax1.hist(weight_dict[chosen_feature]['light1-weight'], bins=hist_bins, color=hist_color, edgecolor='#000000', alpha=.5)
            ax1.set_title(chosen_feature)
            ax1.tick_params(axis='y', which='both', length=0, labelsize=8)
            ax1.set_xlabel('Firing rate difference (spikes/s)', labelpad=.1)
            ax1.set_ylabel('Number of units', labelpad=.1)
            ax2 = fig.add_subplot(gs1[-1, :-1])
            ax2.scatter(x=weight_dict[chosen_feature]['light1-light2'], y=weight_dict[chosen_feature]['light1-weight'], color=hist_color, alpha=.5, s=2)
            ax2.plot([0, 1], [0, 1], ls='-.', lw=.5, color='#000000', transform=ax2.transAxes)
            ax2.set_xlim(min_joint, max_joint)
            ax2.set_ylim(min_joint, max_joint)
            ax2.set_xticks([min_joint, 0, max_joint])
            ax2.set_yticks([min_joint, 0, max_joint])
            ax2.set_xlabel('light1-light2', labelpad=.5)
            ax2.set_ylabel('light1-weight', labelpad=.1)
            ax2.tick_params(axis='both', which='both', labelsize=5)
            ax3 = fig.add_subplot(gs1[-1, -1])
            k = kde.gaussian_kde([weight_dict[chosen_feature]['light1-light2'], weight_dict[chosen_feature]['light1-weight']])
            xi, yi = np.mgrid[min_joint:max_joint:10*1j, min_joint:max_joint:10*1j]
            zi = k(np.vstack([xi.flatten(), yi.flatten()]))
            ax3.pcolormesh(xi, yi, zi.reshape(xi.shape), shading='gouraud', cmap='cividis')
            ax3.plot([0, 1], [0, 1], ls='-.', lw=.5, color='#FFFFFF', transform=ax3.transAxes)
            ax3.set_xlim(min_joint, max_joint)
            ax3.set_ylim(min_joint, max_joint)
            ax3.set_xticks([min_joint, 0, max_joint])
            ax3.set_yticks([])
            ax3.set_xlabel('light1-light2', labelpad=.5)
            ax3.tick_params(axis='both', which='both', labelsize=5)

            gs2 = fig.add_gridspec(nrows=3, ncols=2, left=0.55, right=0.98,
                                   wspace=0.1, hspace=0.5)
            ax4 = fig.add_subplot(gs2[:-1, :])
            ax4.hist(weight_dict[feature_der]['light1-light2'], bins=hist_bins, color=hist_color, edgecolor='#000000', alpha=.2, label='light1-light2')
            ax4.hist(weight_dict[feature_der]['light1-weight'], bins=hist_bins, color=hist_color, edgecolor='#000000', alpha=.5, label='light1-weight')
            ax4.set_title(feature_der)
            ax4.tick_params(axis='y', which='both', length=0, labelsize=8)
            ax4.set_xlabel('Firing rate difference (spikes/s)', labelpad=.1)
            ax4.legend(loc='best', fontsize='small')
            ax5 = fig.add_subplot(gs2[-1, :-1])
            ax5.scatter(x=weight_dict[feature_der]['light1-light2'], y=weight_dict[feature_der]['light1-weight'], color=hist_color, alpha=.5, s=2)
            ax5.plot([0, 1], [0, 1], ls='-.', lw=.5, color='#000000', transform=ax5.transAxes)
            ax5.set_xlim(min_joint, max_joint)
            ax5.set_ylim(min_joint, max_joint)
            ax5.set_xticks([min_joint, 0, max_joint])
            ax5.set_yticks([min_joint, 0, max_joint])
            ax5.set_xlabel('light1-light2', labelpad=.5)
            ax5.tick_params(axis='both', which='both', labelsize=5)
            ax6 = fig.add_subplot(gs2[-1, -1])
            ax6.set_yticks([])
            k2 = kde.gaussian_kde([weight_dict[feature_der]['light1-light2'], weight_dict[feature_der]['light1-weight']])
            zi2 = k2(np.vstack([xi.flatten(), yi.flatten()]))
            ax6.pcolormesh(xi, yi, zi2.reshape(xi.shape), shading='gouraud', cmap='cividis')
            ax6.plot([0, 1], [0, 1], ls='-.', lw=.5, color='#FFFFFF', transform=ax6.transAxes)
            ax6.set_xlim(min_joint, max_joint)
            ax6.set_ylim(min_joint, max_joint)
            ax6.set_xticks([min_joint, 0, max_joint])
            ax6.set_yticks([])
            ax6.set_xlabel('light1-light2', labelpad=.5)
            ax6.tick_params(axis='both', which='both', labelsize=5)
            if self.save_fig:
                if os.path.exists(self.save_dir):
                    fig.savefig(f'{self.save_dir}{os.sep}_weight_feature_distributions_{chosen_feature}.{self.fig_format}', dpi=300)
                else:
                    print("Specified save directory doesn't exist. Try again.")
                    sys.exit()
            plt.show()

    def plot_weight_statistics(self, **kwargs):
        """
        Description
        ----------
        This method plots the raw statistics of light1-light2 and
        light1-weight peak rate differences: their means and dependent
        samples t-test for each feature (and first derivative).
        ----------

        Parameters
        ----------
        **kwargs (dictionary)
        min_max_rate (list)
            Min and max rate on the plot; defaults to [1.5, 4].
        plot_feature_legend (bool)
            Yey or ney on the feature plot; defaults to False.
        ----------

        Returns
        ----------
        weight_statistics (plot)
            A plot with weight peak rate differences statistics.
        ----------
        """

        min_max_rate = kwargs['min_max_rate'] if 'min_max_rate' in kwargs.keys() and type(kwargs['min_max_rate']) == list else [1.5, 4]
        plot_feature_legend = kwargs['plot_feature_legend'] if 'plot_feature_legend' in kwargs.keys() and type(kwargs['plot_feature_legend']) == bool else False

        weight_dict = extract_json_data(json_file=self.weight_json_file,
                                        weight=True,
                                        features=self.chosen_features,
                                        weight_stability_bound=self.weight_stability_bound)
        weight_stats_dict={}
        light1_light2=[]
        light1_weight=[]
        p_values=[]
        cl_n=[]
        feature_colors=[]
        for feature in weight_dict.keys():
            weight_stats_dict[feature] = {}
            weight_stats_dict[feature]['light1-light2'] = np.mean(np.absolute(weight_dict[feature]['light1-light2']))
            light1_light2.append(np.mean(np.absolute(weight_dict[feature]['light1-light2'])))
            weight_stats_dict[feature]['light1-weight'] = np.mean(np.absolute(weight_dict[feature]['light1-weight']))
            light1_weight.append(np.mean(np.absolute(weight_dict[feature]['light1-weight'])))
            temp_p = ttest_rel(a=np.absolute(weight_dict[feature]['light1-light2']),
                               b=np.absolute(weight_dict[feature]['light1-weight']))[1]
            weight_stats_dict[feature]['p-value'] = temp_p
            p_values.append(temp_p)
            weight_stats_dict[feature]['cl_n'] = len(weight_dict[feature]['light1-light2'])
            cl_n.append(len(weight_dict[feature]['light1-light2']))
            feature_colors.append([val for key, val in Ratemap.feature_colors.items() if key in feature][0])

        # print(weight_stats_dict)

        fig, ax = plt.subplots(nrows=1, ncols=1)
        ax = plt.subplot(111)
        cax = ax.scatter(light1_light2, light1_weight, label=None,
                         c=np.log10(p_values), cmap='seismic_r',
                         s=cl_n, alpha=.85,
                         edgecolors=feature_colors, linewidths=3)

        ax.set_xlim(min_max_rate[0], min_max_rate[1])
        ax.set_ylim(min_max_rate[0], min_max_rate[1])
        ax.set_xlabel('light1-light2 mean (spikes/s)')
        ax.set_ylabel('light1-weight mean (spikes/s)')
        cbar = fig.colorbar(cax)
        cbar.set_label('log$_{10}$(p-value)')
        cax.set_clim(vmin=-4, vmax=0)
        for area in [100, 300, 500]:
            ax.scatter([], [], c='#000000', alpha=.3, s=area,
                       label=str(area) + ' units')
            ax.legend(scatterpoints=1, frameon=False,
                      labelspacing=1.5, title='Number of clusters')
        ax.plot([min_max_rate[0], min_max_rate[1]], [min_max_rate[0], min_max_rate[1]], ls='-.', lw=.5, color='#000000')
        if plot_feature_legend:
            y_start = 2.65
            for idx, (key, val) in enumerate(Ratemap.feature_colors.items()):
                if idx < 6:
                    if idx % 2 == 0:
                        ax.axhline(xmin=.6, xmax=.7, y=y_start, ls='-', lw=2.5, color=val)
                        ax.text(x=3.3, y=y_start-.025, s=key.lower().replace('_', ' '))
                        y_start -= .1
                else:
                    ax.axhline(xmin=.6, xmax=.7, y=y_start, ls='-', lw=2.5, color=val)
                    ax.text(x=3.3, y=y_start-.025, s=key.lower().replace('_', ' '))
                    y_start -= .1
        if self.save_fig:
            if os.path.exists(self.save_dir):
                fig.savefig(f'{self.save_dir}{os.sep}_weight_statistics.{self.fig_format}', dpi=300)
            else:
                print("Specified save directory doesn't exist. Try again.")
                sys.exit()
        plt.show()


