# -*- coding: utf-8 -*-

"""

@author: bartulem

Compare tuning-curve rate differences in weight/no-weight sessions.

"""

import os
import sys
import json
import scipy.stats
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from random import gauss
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from make_ratemaps import Ratemap

def extract_json_data(json_file='', weight=False, features=None,
                      peak_min=True, der='1st', rate_stability_bound=True):
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
    peak_min (int / bool)
        The minimum rate for light1 to even be considered; defaults to True.
    rate_stability_bound (int / bool)
        How much is the OF2 peak rate allowed to deviate from OF1 (in terms of percent OF1 rate); defaults to True.
    der (str)
        Derivative of choice; defaults to '1st'.
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
            weight_dict[feature] = {'light1': [], 'weight': [], 'light2': []}
            if der == '2nd' and feature == 'Speeds':
                continue
            else:
                weight_dict[f'{feature}_{der}_der'] = {'light1': [], 'weight': [], 'light2': []}

    for cl_num in json_data.keys():
        if weight:
            for key in json_data[cl_num]['features'].keys():
                if key in weight_dict.keys():
                    if (peak_min is True or json_data[cl_num]['features'][key]['light1'] > peak_min) \
                            and (peak_min is True or json_data[cl_num]['features'][key]['weight'] > peak_min) \
                            and (peak_min is True or json_data[cl_num]['features'][key]['light2'] > peak_min) \
                            and abs(json_data[cl_num]['features'][key]['light1']
                                    - json_data[cl_num]['features'][key]['weight']) \
                            > abs(json_data[cl_num]['features'][key]['light1']
                                  - json_data[cl_num]['features'][key]['light2']) \
                            and rate_stability_bound is True or abs(json_data[cl_num]['features'][key]['light1'] - json_data[cl_num]['features'][key]['light2']) \
                            < (json_data[cl_num]['features'][key]['light1'] * (rate_stability_bound / 100)):
                        weight_dict[key]['light1'].append(json_data[cl_num]['features'][key]['light1'])
                        weight_dict[key]['weight'].append(json_data[cl_num]['features'][key]['weight'])
                        weight_dict[key]['light2'].append(json_data[cl_num]['features'][key]['light2'])

    if weight:
        return weight_dict


def make_shuffled_distributions(weight_dict, n_shuffles=1000):
    shuffled_dict = {}
    for feature in tqdm(weight_dict.keys()):
        shuffled_dict[feature] = {'null_differences': np.zeros(n_shuffles), 'true_difference': 0, 'z-value': 1, 'p-value': 1}
        true_difference = np.median(np.diff(np.array([weight_dict[feature]['weight'], weight_dict[feature]['light1']]), axis=0))
        shuffled_dict[feature]['true_difference'] = true_difference
        for sh in range(n_shuffles):
            joint_arr = np.array([weight_dict[feature]['weight'], weight_dict[feature]['light1']])
            for col in range(joint_arr.shape[1]):
                np.random.shuffle(joint_arr[:, col])
                shuffled_dict[feature]['null_differences'][sh] = np.median(np.diff(joint_arr, axis=0))
        shuffled_dict[feature]['z-value'] = (true_difference - shuffled_dict[feature]['null_differences'].mean()) \
                                            / shuffled_dict[feature]['null_differences'].std()
        shuffled_dict[feature]['p-value'] = 1 - scipy.stats.norm.cdf(shuffled_dict[feature]['z-value'])
    return shuffled_dict


class WeightComparer:

    def __init__(self, weight_json_file='', chosen_features=None,
                 rate_stability_bound=True, peak_min=True,
                 save_dir='', save_fig=False, fig_format='png',
                 light_session=1, der='1st'):
        if chosen_features is None:
            chosen_features = ['Speeds']
        self.weight_json_file = weight_json_file
        self.chosen_features = chosen_features
        self.peak_min = peak_min
        self.rate_stability_bound = rate_stability_bound
        self.save_dir=save_dir
        self.save_fig=save_fig
        self.fig_format=fig_format
        self.light_session = light_session
        self.der = der

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
        peak_min (int / bool)
            The minimum rate for light1 to even be considered; defaults to True.
        rate_stability_bound (int / float / bool)
            How much is the OF2 peak rate allowed to deviate from OF1 (in terms of percent OF1 rate); defaults to True.
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
                                        peak_min=self.peak_min,
                                        rate_stability_bound=self.rate_stability_bound,
                                        der=self.der)

        shuffled_dict = make_shuffled_distributions(weight_dict=weight_dict)
        for feature in shuffled_dict.keys():
            print(feature, len(weight_dict[feature]['light1']), np.median(weight_dict[feature]['light1']), np.median(weight_dict[feature]['weight']),
                  shuffled_dict[feature]['null_differences'].mean(), shuffled_dict[feature]['true_difference'],
                  shuffled_dict[feature]['z-value'], shuffled_dict[feature]['p-value'])

        for chosen_feature in self.chosen_features:
            if min_max_range:
                min_joint = np.min(weight_dict[chosen_feature]['light1'] + weight_dict[chosen_feature]['weight'])
                max_joint = int(np.ceil(np.max(weight_dict[chosen_feature]['light1'] + weight_dict[chosen_feature]['weight']) / 10.0)) * 10
            else:
                min_joint=.1
                max_joint=100

            feature_color = [val for key, val in Ratemap.feature_colors.items() if key in chosen_feature][0]
            feature_der = f'{chosen_feature}_{self.der}_der'

            fig = plt.figure()
            gs1 = fig.add_gridspec(nrows=3, ncols=2, left=.075, right=.505,
                                   wspace=0.1, hspace=0.5)
            ax1 = fig.add_subplot(gs1[:-1, :])
            light_larger = np.array(weight_dict[chosen_feature]['light1']) > np.array(weight_dict[chosen_feature]['weight'])
            weight_larger = ~light_larger
            ax1.scatter(x=np.array(weight_dict[chosen_feature]['light1'])[light_larger], y=np.array(weight_dict[chosen_feature]['weight'])[light_larger],
                        color=feature_color, alpha=.25, s=10)
            ax1.scatter(x=np.array(weight_dict[chosen_feature]['light1'])[weight_larger], y=np.array(weight_dict[chosen_feature]['weight'])[weight_larger],
                        color=feature_color, alpha=.75, s=10)
            ax1.plot(np.median(weight_dict[chosen_feature]['light1']), np.median(weight_dict[chosen_feature]['weight']),
                     marker='o', markersize=20, color=feature_color, alpha=.5)
            ax1.plot([min_joint, max_joint], [min_joint, max_joint], ls='-.', lw=.5, color='#000000')
            axins1 = inset_axes(ax1, width='40%', height='30%', loc=2)
            k = scipy.stats.kde.gaussian_kde([np.log10(weight_dict[chosen_feature]['light1']), np.log10(weight_dict[chosen_feature]['weight'])])
            xi, yi = np.mgrid[np.log10(min_joint):np.log10(max_joint):10*1j, np.log10(min_joint):np.log10(max_joint):10*1j]
            zi = k(np.vstack([xi.flatten(), yi.flatten()]))
            axins1.pcolormesh(xi, yi, zi.reshape(xi.shape), shading='gouraud', cmap='cividis')
            axins1.plot([0, 1], [0, 1], ls='-.', lw=.5, color='#FFFFFF', transform=axins1.transAxes)
            axins1.set_xticks([])
            axins1.set_yticks([])
            axins1.set_xlim(np.log10(min_joint), np.log10(max_joint))
            axins1.set_ylim(np.log10(min_joint), np.log10(max_joint))
            ax1.set_title(chosen_feature)
            ax1.set_xlim(min_joint, max_joint)
            ax1.set_xscale('log')
            ax1.set_ylim(min_joint, max_joint)
            ax1.set_yscale('log')
            ax1.tick_params(axis='both', which='major', length=0, labelsize=8, pad=.5)
            ax1.set_xlabel(f'L{self.light_session} peak rate (spikes/s)', labelpad=.1)
            ax1.set_ylabel(f'W rate at L{self.light_session} peak (spikes/s)', labelpad=.1)

            ax2 = fig.add_subplot(gs1[-1, :-1])
            ax2.hist(shuffled_dict[chosen_feature]['null_differences'], bins=10, histtype='stepfilled',
                     color='#808080', edgecolor='#000000', alpha=.5)
            ax2.axvline(x=np.nanpercentile(shuffled_dict[chosen_feature]['null_differences'], 99), color='#FF6347', linestyle='dashdot', linewidth=.5)
            y_min_2, y_max_2 = ax2.get_ylim()
            ax2.plot(shuffled_dict[chosen_feature]['true_difference'], 0+.05*y_max_2, marker='o', color=feature_color, markersize=5)
            ax2.set_xlabel(f'L{self.light_session} - W difference (spikes/s)', labelpad=.5, fontsize=6)
            ax2.set_ylabel('Number of cells', labelpad=.1, fontsize=6)
            ax2.tick_params(axis='both', which='both', labelsize=5)

            ax3 = fig.add_subplot(gs1[-1, -1])
            ax3.scatter(x=[gauss(.2, .015) for x in range(len(weight_dict[chosen_feature]['light1']))], y=weight_dict[chosen_feature]['light1'],
                        color=feature_color, alpha=.15, s=10)
            ax3.boxplot(x=weight_dict[chosen_feature]['light1'], positions=[.35], notch=False, sym='', widths=.1)
            ax3.scatter(x=[gauss(.6, .015) for x in range(len(weight_dict[chosen_feature]['weight']))], y=weight_dict[chosen_feature]['weight'],
                        color=feature_color, alpha=.75, s=10)
            ax3.boxplot(x=weight_dict[chosen_feature]['weight'], positions=[.75], notch=False, sym='', widths=.1)
            ax3.set_xlim(0, 1)
            ax3.set_xticks([.275, .675])
            ax3.set_xticklabels(['light1', 'weight'], fontsize=6)
            ax3.tick_params(axis='x', which='both', length=0)
            ax3.yaxis.set_ticks_position('right')
            ax3.yaxis.set_label_position('right')
            ax3.set_yscale('log')
            ax3.set_yticks([.1, 1, 10, 100])
            # ax3.set_yticklabels(['-1', '0', '1', '2'], fontsize=6)
            ax3.set_ylim(.1, 150)
            ax3.set_yticklabels([])
            ax3.tick_params(axis='y', which='both', length=0)
            # ax3.set_ylabel('log$_{10}$ peak rate (spikes/s)', labelpad=.3, fontsize=6)
            # ax3.get_yaxis().get_major_formatter().labelOnlyBase = False

            gs2 = fig.add_gridspec(nrows=3, ncols=2, left=0.55, right=0.98,
                                   wspace=0.1, hspace=0.5)
            ax4 = fig.add_subplot(gs2[:-1, :])
            light_larger_der = np.array(weight_dict[feature_der]['light1']) > np.array(weight_dict[feature_der]['weight'])
            weight_larger_der = ~light_larger_der
            ax4.scatter(x=np.array(weight_dict[feature_der]['light1'])[light_larger_der], y=np.array(weight_dict[feature_der]['weight'])[light_larger_der],
                        color=feature_color, alpha=.25, s=10)
            ax4.scatter(x=np.array(weight_dict[feature_der]['light1'])[weight_larger_der], y=np.array(weight_dict[feature_der]['weight'])[weight_larger_der],
                        color=feature_color, alpha=.75, s=10)
            ax4.plot(np.median(weight_dict[feature_der]['light1']), np.median(weight_dict[feature_der]['weight']),
                     marker='o', markersize=20, color=feature_color, alpha=.5)
            ax4.plot([min_joint, max_joint], [min_joint, max_joint], ls='-.', lw=.5, color='#000000')
            axins4 = inset_axes(ax4, width='40%', height='30%', loc=2)
            k4 = scipy.stats.kde.gaussian_kde([np.log10(weight_dict[feature_der]['light1']), np.log10(weight_dict[feature_der]['weight'])])
            zi4 = k4(np.vstack([xi.flatten(), yi.flatten()]))
            axins4.pcolormesh(xi, yi, zi4.reshape(xi.shape), shading='gouraud', cmap='cividis')
            axins4.plot([0, 1], [0, 1], ls='-.', lw=.5, color='#FFFFFF', transform=axins4.transAxes)
            axins4.set_xticks([])
            axins4.set_yticks([])
            axins4.set_xlim(np.log10(min_joint), np.log10(max_joint))
            axins4.set_ylim(np.log10(min_joint), np.log10(max_joint))
            ax4.set_title(feature_der)
            ax4.set_xlim(min_joint, max_joint)
            ax4.set_xscale('log')
            ax4.set_ylim(min_joint, max_joint)
            ax4.set_yscale('log')
            ax4.tick_params(axis='both', which='major', length=0, labelsize=8, pad=.5)
            ax4.set_xlabel(f'L{self.light_session} peak rate (spikes/s)', labelpad=.1)

            ax5 = fig.add_subplot(gs2[-1, :-1])
            ax5.hist(shuffled_dict[feature_der]['null_differences'], bins=10, histtype='stepfilled',
                     color='#808080', edgecolor='#000000', alpha=.5)
            ax5.axvline(x=np.nanpercentile(shuffled_dict[feature_der]['null_differences'], 99), color='#FF6347', linestyle='dashdot', linewidth=.5)
            y_min_5, y_max_5 = ax5.get_ylim()
            ax5.plot(shuffled_dict[feature_der]['true_difference'], 0+.05*y_max_5, marker='o', color=feature_color, markersize=5)
            ax5.set_xlabel(f'L{self.light_session} - W difference (spikes/s)', labelpad=.5, fontsize=6)
            ax5.tick_params(axis='both', which='both', labelsize=5, pad=.5)

            ax6 = fig.add_subplot(gs2[-1, -1])
            ax6.scatter(x=[gauss(.2, .015) for x in range(len(weight_dict[feature_der]['light1']))], y=weight_dict[feature_der]['light1'],
                        color=feature_color, alpha=.15, s=10)
            ax6.boxplot(x=weight_dict[feature_der]['light1'], positions=[.35], notch=False, sym='', widths=.1)
            ax6.scatter(x=[gauss(.6, .015) for x in range(len(weight_dict[feature_der]['weight']))], y=weight_dict[feature_der]['weight'],
                        color=feature_color, alpha=.75, s=10)
            ax6.boxplot(x=weight_dict[feature_der]['weight'], positions=[.75], notch=False, sym='', widths=.1)
            ax6.set_xlim(0, 1)
            ax6.set_xticks([.275, .675])
            ax6.set_xticklabels(['light1', 'weight'], fontsize=6)
            ax6.tick_params(axis='x', which='both', length=0)
            ax6.yaxis.set_ticks_position('right')
            ax6.yaxis.set_label_position('right')
            ax6.set_yscale('log')
            ax6.set_yticks([.1, 1, 10, 100])
            ax6.set_yticklabels([])
            ax6.tick_params(axis='y', which='both', length=0)
            ax6.set_ylim(.1, 150)
            ax6.get_yaxis().get_major_formatter().labelOnlyBase = False
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
        min_max_rate (list / bool)
            Min and max rate on the plot; defaults to True.
        plot_feature_legend (bool)
            Yey or ney on the feature legend plot; defaults to False.
        ----------

        Returns
        ----------
        weight_statistics (plot)
            A plot with weight peak rate differences statistics.
        ----------
        """

        min_max_rate = kwargs['min_max_rate'] if 'min_max_rate' in kwargs.keys() \
                                                 and type(kwargs['min_max_rate']) == list and len(kwargs['min_max_rate']) == 2 else True
        plot_feature_legend = kwargs['plot_feature_legend'] if 'plot_feature_legend' in kwargs.keys() and type(kwargs['plot_feature_legend']) == bool else False

        weight_dict = extract_json_data(json_file=self.weight_json_file,
                                        weight=True,
                                        features=self.chosen_features,
                                        peak_min=self.peak_min,
                                        rate_stability_bound=self.rate_stability_bound,
                                        der=self.der)

        shuffled_dict = make_shuffled_distributions(weight_dict=weight_dict)

        weight_stats_dict={'features': {'light1': [], 'weight': [], 'p-values': [], 'cl_n': [], 'names': [], 'feature_colors': []},
                           'features_der': {'light1': [], 'weight': [], 'p-values': [], 'cl_n': [], 'names': [], 'feature_colors': []}}

        for feature in weight_dict.keys():
            if 'der' not in feature:
                weight_stats_dict['features']['light1'].append(np.median(weight_dict[feature]['light1']))
                weight_stats_dict['features']['weight'].append(np.median(weight_dict[feature]['weight']))
                weight_stats_dict['features']['p-values'].append(shuffled_dict[feature]['p-value'])
                weight_stats_dict['features']['cl_n'].append(len(weight_dict[feature]['light1']))
                weight_stats_dict['features']['names'].append(feature)
                weight_stats_dict['features']['feature_colors'].append([val for key, val in Ratemap.feature_colors.items() if key in feature][0])
            else:
                weight_stats_dict['features_der']['light1'].append(np.median(weight_dict[feature]['light1']))
                weight_stats_dict['features_der']['weight'].append(np.median(weight_dict[feature]['weight']))
                weight_stats_dict['features_der']['p-values'].append(shuffled_dict[feature]['p-value'])
                weight_stats_dict['features_der']['cl_n'].append(len(weight_dict[feature]['light1']))
                weight_stats_dict['features_der']['names'].append(feature)
                weight_stats_dict['features_der']['feature_colors'].append([val for key, val in Ratemap.feature_colors.items() if key in feature][0])

        print(weight_stats_dict)

        if type(min_max_rate) == bool:
            min_rate = int(np.floor(np.min(weight_stats_dict['features']['light1'] + weight_stats_dict['features']['weight'])))
            max_rate = int(np.ceil(np.max(weight_stats_dict['features']['light1'] + weight_stats_dict['features']['weight'])))
            min_rate_der = int(np.floor(np.min(weight_stats_dict['features_der']['light1'] + weight_stats_dict['features_der']['weight'])))
            max_rate_der = int(np.ceil(np.max(weight_stats_dict['features_der']['light1'] + weight_stats_dict['features_der']['weight'])))
        else:
            min_rate = min_max_rate[0]
            max_rate = min_max_rate[1]
            min_rate_der = min_max_rate[0]
            max_rate_der = min_max_rate[1]


        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))
        ax1 = plt.subplot(121)
        sca1 = ax1.scatter(weight_stats_dict['features']['light1'], weight_stats_dict['features']['weight'],
                           label=None, c=np.log10(weight_stats_dict['features']['p-values']), cmap='seismic_r',
                           s=np.array(weight_stats_dict['features']['cl_n'])*2, alpha=.85,
                           edgecolors=weight_stats_dict['features']['feature_colors'], lw=2)

        ax1.set_title('Features')
        ax1.set_xlim(min_rate, max_rate)
        ax1.set_ylim(min_rate, max_rate)
        ax1.set_xlabel(f'L{self.light_session} median peak (spikes/s)')
        ax1.set_ylabel(f'W median at L{self.light_session} peak (spikes/s)')
        for area in np.array([50, 100, 200]):
            ax1.scatter([], [], c='#000000', alpha=.3, s=area*2,
                        label=str(area) + ' units')
            ax1.legend(scatterpoints=1, frameon=False,
                       labelspacing=1.5, title='Number of clusters')
        ax1.plot([min_rate, max_rate], [min_rate, max_rate], ls='-.', lw=.5, color='#000000')
        ax2 = plt.subplot(122)
        sca2 = ax2.scatter(weight_stats_dict['features_der']['light1'], weight_stats_dict['features_der']['weight'],
                           label=None, c=np.log10(weight_stats_dict['features_der']['p-values']), cmap='seismic_r',
                           s=np.array(weight_stats_dict['features_der']['cl_n'])*2, alpha=.85,
                           edgecolors=weight_stats_dict['features_der']['feature_colors'], lw=3)

        ax2.set_title('Derivatives')
        ax2.set_xlim(min_rate_der, max_rate_der)
        ax2.set_ylim(min_rate_der, max_rate_der)
        ax2.set_xlabel(f'L{self.light_session} median peak (spikes/s)')
        ax2.plot([min_rate_der, max_rate_der], [min_rate_der, max_rate_der], ls='-.', lw=.5, color='#000000')
        cb_ax = fig.add_axes([.91,.124,.01,.754])
        cbar = fig.colorbar(sca2, orientation='vertical', cax=cb_ax)
        cbar.set_label('log$_{10}$(p-value)')
        sca1.set_clim(vmin=-4, vmax=0)
        sca2.set_clim(vmin=-4, vmax=0)
        if plot_feature_legend:
            y_start = max_rate_der - .5
            for idx, (key, val) in enumerate(Ratemap.feature_colors.items()):
                if (idx < 6 and idx % 2 == 0) or idx >= 6:
                    ax2.axhline(xmin=.05, xmax=.15, y=y_start, ls='-', lw=2.5, color=val)
                    ax2.text(x=min_rate_der + (max_rate_der-min_rate_der)*.175, y=y_start-.05,
                             s=key.lower().replace('_', ' '), fontsize=7)
                    y_start -= .5
        if self.save_fig:
            if os.path.exists(self.save_dir):
                fig.savefig(f'{self.save_dir}{os.sep}_weight_statistics.{self.fig_format}', dpi=300)
            else:
                print("Specified save directory doesn't exist. Try again.")
                sys.exit()
        plt.show()


