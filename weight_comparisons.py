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
import matplotlib.patches as patches
import matplotlib.colorbar as cbar
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter1d, uniform_filter1d
from tqdm import tqdm
from random import gauss
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from make_ratemaps import Ratemap
from neural_activity import Spikes

def extract_json_data(json_file='', weight=False, features=None,
                      peak_min=True, der='1st', rate_stability_bound=True,
                      ref_dict=None, middle_session='weight'):
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
    ref_dict (dict)
        The reference and second light session: defaults to {'ref_session': 'light1', 'other_session': 'light2'}
    middle_session (str)
        The middle session of interest; defaults to 'weight'.
    ----------

    Returns
    ----------
    data_dict (dict)
        A dictionary which contains the desired data.
    ----------
    """

    if ref_dict is None:
        ref_dict = {'ref_session': 'light1', 'other_session': 'light2'}
    if features is None:
        features = ['Speeds']

    with open(json_file) as j_file:
        json_data = json.load(j_file)

    if weight:
        weight_dict = {}
        for feature in features:
            weight_dict[feature] = {'peaks': {'light1': [], middle_session: [], 'light2': []},
                                    'information_rates': {'light1': [], middle_session: [], 'light2': []},
                                    'correlations': {f'light1-{middle_session}': [], f'light2-{middle_session}': [], 'lights': []}}
            if der == '2nd' and feature == 'Speeds':
                continue
            else:
                weight_dict[f'{feature}_{der}_der'] = {'peaks': {'light1': [], middle_session: [], 'light2': []},
                                                       'information_rates': {'light1': [], middle_session: [], 'light2': []},
                                                       'correlations': {f'light1-{middle_session}': [], f'light2-{middle_session}': [], 'lights': []}}

    for cl_num in json_data.keys():
        if weight:
            for key in json_data[cl_num]['features'].keys():
                if key in weight_dict.keys() and len(json_data[cl_num]['features'][key][ref_dict['ref_session']]) > 5:
                    ref_session = json_data[cl_num]['features'][key][ref_dict['ref_session']]
                    weight_session = json_data[cl_num]['features'][key][middle_session]
                    other_session = json_data[cl_num]['features'][key][ref_dict['other_session']]
                    ref_session_peak = np.max(ref_session)
                    weight_at_peak = weight_session[np.argmax(ref_session_peak)]
                    other_session_at_peak = other_session[np.argmax(ref_session_peak)]
                    if (peak_min is True or ref_session_peak > peak_min) \
                            and (peak_min is True or np.max(weight_session) > peak_min) \
                            and (peak_min is True or np.max(other_session) > peak_min) \
                            and rate_stability_bound is True or abs(json_data[cl_num]['baseline_firing_rates'][ref_dict['ref_session']]
                                                                    - json_data[cl_num]['baseline_firing_rates'][middle_session]) \
                            < (json_data[cl_num]['baseline_firing_rates'][ref_dict['ref_session']] * (rate_stability_bound / 100)):
                        weight_dict[key]['peaks'][ref_dict['ref_session']].append(ref_session_peak)
                        weight_dict[key]['peaks'][middle_session].append(weight_at_peak)
                        weight_dict[key]['peaks'][ref_dict['other_session']].append(other_session_at_peak)

                        weight_dict[key]['information_rates'][ref_dict['ref_session']].append(json_data[cl_num]['features'][key]['ICr-{}'.format(ref_dict['ref_session'])])
                        weight_dict[key]['information_rates'][middle_session].append(json_data[cl_num]['features'][key]['ICr-{}'.format(middle_session)])
                        weight_dict[key]['information_rates'][ref_dict['other_session']].append(json_data[cl_num]['features'][key]['ICr-{}'.format(ref_dict['other_session'])])

                        weight_dict[key]['correlations']['{}-{}'.format(ref_dict['ref_session'], middle_session)].append(scipy.stats.spearmanr(ref_session, weight_session)[0])
                        weight_dict[key]['correlations']['{}-{}'.format(ref_dict['other_session'], middle_session)].append(scipy.stats.spearmanr(other_session, weight_session)[0])
                        weight_dict[key]['correlations']['lights'].append(scipy.stats.spearmanr(ref_session, other_session)[0])

    if weight:
        return weight_dict


def make_shuffled_distributions(weight_dict, ref_dict, middle_session, n_shuffles=1000):
    shuffled_dict = {}
    for feature in tqdm(weight_dict.keys()):
        shuffled_dict[feature] = {'peaks': {'null_differences': np.zeros(n_shuffles), 'true_difference': 0, 'z-value': 1, 'p-value': 1},
                                  'correlations': {'null_differences': np.zeros(n_shuffles), 'true_difference': 0, 'z-value': 1, 'p-value': 1},
                                  'information_rates': {'null_differences': np.zeros(n_shuffles), 'true_difference': 0, 'z-value': 1, 'p-value': 1}}
        for attribute in ['peaks', 'information_rates']:
            true_difference_attribute = np.mean(np.diff(np.array([weight_dict[feature][attribute][middle_session],
                                                              weight_dict[feature][attribute][ref_dict['ref_session']]]), axis=0))
            shuffled_dict[feature][attribute]['true_difference'] = true_difference_attribute
        corr_arr = np.array([weight_dict[feature]['correlations']['{}-{}'.format(ref_dict['ref_session'], middle_session)],
                             weight_dict[feature]['correlations']['lights']])
        corr_arr[corr_arr > .99] = .99
        true_difference_corr = np.mean(np.diff(np.arctanh(corr_arr), axis=0))
        for sh in range(n_shuffles):
            joint_arr = np.array([weight_dict[feature]['peaks'][middle_session],
                                  weight_dict[feature]['peaks'][ref_dict['ref_session']]])
            joint_arr_ir = np.array([weight_dict[feature]['information_rates'][middle_session],
                                     weight_dict[feature]['information_rates'][ref_dict['ref_session']]])
            joint_arr_corr = np.arctanh(corr_arr.copy())
            for col in range(joint_arr.shape[1]):
                np.random.shuffle(joint_arr[:, col])
                shuffled_dict[feature]['peaks']['null_differences'][sh] = np.mean(np.diff(joint_arr, axis=0))
                np.random.shuffle(joint_arr_ir[:, col])
                shuffled_dict[feature]['information_rates']['null_differences'][sh] = np.mean(np.diff(joint_arr_ir, axis=0))
                np.random.shuffle(joint_arr_corr[:, col])
                shuffled_dict[feature]['correlations']['null_differences'][sh] = np.mean(np.diff(joint_arr_corr, axis=0))
        for attribute in ['peaks', 'information_rates']:
            shuffled_dict[feature][attribute]['z-value'] = (shuffled_dict[feature][attribute]['true_difference'] - shuffled_dict[feature][attribute]['null_differences'].mean()) \
                                                         / shuffled_dict[feature][attribute]['null_differences'].std()
            p_val_attribute = 1 - scipy.stats.norm.cdf(shuffled_dict[feature][attribute]['z-value'])
            if p_val_attribute < .5:
                shuffled_dict[feature][attribute]['p-value'] = p_val_attribute
            else:
                shuffled_dict[feature][attribute]['p-value'] = 1-p_val_attribute
        shuffled_dict[feature]['correlations']['z-value'] = (true_difference_corr - shuffled_dict[feature]['correlations']['null_differences'].mean()) \
                                                            / shuffled_dict[feature]['correlations']['null_differences'].std()
        p_val_correlations = 1 - scipy.stats.norm.cdf(shuffled_dict[feature]['correlations']['z-value'])
        if p_val_correlations < .5:
            shuffled_dict[feature]['correlations']['p-value'] = p_val_correlations
        else:
            shuffled_dict[feature]['correlations']['p-value'] = 1-p_val_correlations
        shuffled_dict[feature]['correlations']['true_difference'] = np.tanh(true_difference_corr)
        shuffled_dict[feature]['correlations']['null_differences'] = np.tanh(shuffled_dict[feature]['correlations']['null_differences'])
    return shuffled_dict


class WeightComparer:

    def __init__(self, weight_json_file='', chosen_features=None,
                 rate_stability_bound=True, peak_min=True,
                 save_dir='', save_fig=False, fig_format='png',
                 light_session=1, der='1st', baseline_dict={},
                 ref_dict=None, middle_session_type='weight'):
        if chosen_features is None:
            chosen_features = ['Speeds']
        if ref_dict is None:
            ref_dict = {'ref_session': 'light1', 'other_session': 'light2'}
        self.weight_json_file = weight_json_file
        self.chosen_features = chosen_features
        self.peak_min = peak_min
        self.rate_stability_bound = rate_stability_bound
        self.save_dir=save_dir
        self.save_fig=save_fig
        self.fig_format=fig_format
        self.light_session = light_session
        self.der = der
        self.baseline_dict = baseline_dict
        self.ref_dict = ref_dict
        self.middle_session_type = middle_session_type

    def baseline_rate_change_over_time(self, **kwargs):
        """
        Description
        ----------
        This method plots how the baseline firing rate changes over
        time for all clusters that were significant for at least one
        1-D feature. [1] The first plot considers six example clusters
        from all animals that specific brain area was recorded from. In
        each session (light1, weight, light2), spikes are allocated into
        10 second bins and smoothed with a Gaussian (sigma=1). They are
        then concatenated into a single array and a rolling mean (size=50) is
        calculated over the whole window.
        ----------

        Parameters
        ----------
        **kwargs (dictionary)
        condensing_bin (int)
            The size of the bin for calculating rates (in seconds); defaults to 10.
        smooth_sd (int)
            The SD of the smoothing window; defaults to 1 (bin).
        rolling_average_window (int)
            The size of the rolling mean window; defaults to 50 (bins).
        ----------

        Returns
        ----------
        baseline_change_examples (plot)
            A plot with cluster examples for firing rate changes over sessions.
        ----------
        """

        condensing_bin = kwargs['condensing_bin'] if 'condensing_bin' in kwargs.keys() and type(kwargs['condensing_bin']) == int else 10
        smooth_sd = kwargs['smooth_sd'] if 'smooth_sd' in kwargs.keys() and type(kwargs['smooth_sd']) == int else 1
        rolling_average_window = kwargs['rolling_average_window'] if 'rolling_average_window' in kwargs.keys() and type(kwargs['rolling_average_window']) == int else 50

        activity_dict = {}
        for animal in self.baseline_dict.keys():
            activity_dict[animal] = {}
            for cl_id in self.baseline_dict[animal]['cl_ids']:
                activity_dict[animal][cl_id] = {'light1': 0, self.middle_session_type: 0, 'light2': 0}
                for idx, session_name in enumerate(['light1', self.middle_session_type, 'light2']):
                    file_id, \
                    activity_dictionary, \
                    purged_spikes_dictionary = Spikes(input_file=self.baseline_dict[animal]['files'][idx]).convert_activity_to_frames_with_shuffles(get_clusters=cl_id,
                                                                                                                                                    to_shuffle=False,
                                                                                                                                                    condense_arr=True,
                                                                                                                                                    condense_bin_ms=int(1000*condensing_bin))
                    activity_dict[animal][cl_id][session_name] = gaussian_filter1d(input=activity_dictionary[cl_id]['activity'].todense(), sigma=smooth_sd) / condensing_bin

        plot_dict = {}
        labels = {}
        borders = {}
        for animal in activity_dict.keys():
            plot_dict[animal] = {}
            labels[animal] = {}
            borders[animal] = [0, 0, 0]
            for cl_idx, cl_id in enumerate(activity_dict[animal].keys()):
                concatenated_activity = np.concatenate((activity_dict[animal][cl_id]['light1'],
                                                        activity_dict[animal][cl_id][self.middle_session_type],
                                                        activity_dict[animal][cl_id]['light2']))
                labels[animal][cl_id] = concatenated_activity.max()
                smoothed_activity = uniform_filter1d(concatenated_activity, size=rolling_average_window)
                plot_dict[animal][cl_id] = smoothed_activity / smoothed_activity.max()
                if cl_idx == 0:
                    borders[animal][0] = activity_dict[animal][cl_id]['light1'].shape[0]
                    borders[animal][1] = activity_dict[animal][cl_id]['light1'].shape[0] \
                                         + activity_dict[animal][cl_id][self.middle_session_type].shape[0]
                    borders[animal][2] = activity_dict[animal][cl_id]['light1'].shape[0] \
                                         + activity_dict[animal][cl_id][self.middle_session_type].shape[0]\
                                         + activity_dict[animal][cl_id]['light2'].shape[0]

        row_num = len(self.baseline_dict.keys())
        fig, ax = plt.subplots(nrows=row_num, ncols=1, figsize=(6.4, row_num*4.8))
        for animal_idx, animal in enumerate(self.baseline_dict.keys()):
            ax = plt.subplot(row_num, 1, animal_idx+1)
            for cl_id in sorted(labels[animal], key=labels[animal].get, reverse=True):
                ax.plot(plot_dict[animal][cl_id], '-',
                        color='#000000',
                        alpha=labels[animal][cl_id] / 100 + .05,
                        label=f'{labels[animal][cl_id]} spikes/s')
            ax.legend(frameon=False)
            ax.set_title(f'Rat #{animal}')
            for bo_idx, border in enumerate(borders[animal]):
                if bo_idx < 2:
                    ax.axvline(x=border, ls='-.', color='#000000', alpha=.25)
            ax.set_xticks([borders[animal][0] // 2,
                           (borders[animal][1]+borders[animal][0]) // 2,
                           (borders[animal][2]+borders[animal][1]) // 2])
            ax.set_xticklabels(['light1', self.middle_session_type, 'light2'])
            ax.tick_params(axis='both', which='both', length=0)
            ax.set_ylabel('Order of recordings')
            ax.set_ylabel('Peak normalized activity')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
        if self.save_fig:
            if os.path.exists(self.save_dir):
                fig.savefig(f'{self.save_dir}{os.sep}baseline_change_examples.{self.fig_format}', dpi=300)
            else:
                print("Specified save directory doesn't exist. Try again.")
                sys.exit()
        plt.show()

        # statistics on baseline rate change
        with open(self.weight_json_file, 'r') as json_file:
            data = json.load(json_file)

        baseline_rates = {'light1': [], self.middle_session_type: [], 'light2': []}
        for cl_num in data.keys():
            baseline_rates['light1'].append(data[cl_num]['baseline_firing_rates']['light1'])
            baseline_rates['light2'].append(data[cl_num]['baseline_firing_rates']['light2'])
            baseline_rates[self.middle_session_type].append(data[cl_num]['baseline_firing_rates'][self.middle_session_type])

        # fig2, ax2 = plt.subplots(nrows=1, ncols=1)
        # ax2 = plt.subplot(1, 1, 1)
        # hist_l1, edges = np.histogram(a=baseline_rates['light1'], bins=np.linspace(0, 40, 50))
        # hist_l2, edges2 = np.histogram(a=baseline_rates['light2'], bins=np.linspace(0, 40, 50))
        # middle_session, edges2 = np.histogram(a=baseline_rates[self.middle_session_type], bins=np.linspace(0, 40, 50))
        # bin_centers = 0.5*(edges[1:]+edges[:-1])
        # plt.plot(bin_centers, hist_l1, c='#000000', alpha=.35, label='light1')
        # plt.plot(bin_centers, hist_l2, c='#000000', alpha=.45, label='light2')
        # plt.plot(bin_centers, middle_session, c='#000000', alpha=1, label=self.middle_session_type)
        # plt.legend()
        # plt.title('Baseline firing rate distributions')
        # plt.xlabel('Baseline firing rate (spikes/s)')
        # plt.ylabel('Unit count')
        # if self.save_fig:
        #     if os.path.exists(self.save_dir):
        #         fig2.savefig(f'{self.save_dir}{os.sep}firing_rate_distributions.{self.fig_format}', dpi=300)
        #     else:
        #         print("Specified save directory doesn't exist. Try again.")
        #         sys.exit()
        # plt.show()

        # fig2, ax2 = plt.subplots(nrows=1, ncols=1)
        # ax2 = plt.subplot(1, 1, 1)
        # ax2.scatter(x=[gauss(.2, .025) for x in range(len(baseline_rates['light1']))], y=baseline_rates['light1'],
        #             color='#000000', alpha=.15, s=10)
        # ax2.boxplot(x=baseline_rates['light1'], positions=[.35], notch=True, sym='', widths=.1)
        # ax2.scatter(x=[gauss(.6, .025) for x in range(len(baseline_rates[self.middle_session_type]))], y=baseline_rates[self.middle_session_type],
        #             color='#000000', alpha=.75, s=10)
        # ax2.boxplot(x=baseline_rates[self.middle_session_type], positions=[.75], notch=True, sym='', widths=.1)
        # ax2.scatter(x=[gauss(1., .025) for x in range(len(baseline_rates['light2']))], y=baseline_rates['light2'],
        #             color='#000000', alpha=.15, s=10)
        # ax2.boxplot(x=baseline_rates['light2'], positions=[1.15], notch=True, sym='', widths=.1)
        # ax2.set_xticks([.275, .675, 1.075])
        # ax2.set_xticklabels(['light1', self.middle_session_type, 'light2'])
        # ax2.set_ylabel('Firing rate (spikes/s)')
        # ax2.set_yscale('log')
        # ax2.spines['top'].set_visible(False)
        # ax2.spines['right'].set_visible(False)
        # if self.save_fig:
        #     if os.path.exists(self.save_dir):
        #         fig2.savefig(f'{self.save_dir}{os.sep}baseline_change_statistics1.{self.fig_format}', dpi=300)
        #     else:
        #         print("Specified save directory doesn't exist. Try again.")
        #         sys.exit()
        # plt.show()

        diff_light1_weight = np.diff(np.array([baseline_rates[self.middle_session_type], baseline_rates['light1']]), axis=0).ravel()
        diff_weight_light2 = np.diff(np.array([baseline_rates['light2'], baseline_rates[self.middle_session_type]]), axis=0).ravel()
        diff_light1_light2 = np.diff(np.array([baseline_rates['light2'], baseline_rates['light1']]), axis=0).ravel()

        shuffled = np.zeros((3, 1000))
        for sh in tqdm(range(1000)):
            for idx, n in enumerate([(self.middle_session_type, 'light1'), ('light2', self.middle_session_type), ('light2', 'light1')]):
                joint_arr = np.array([baseline_rates[n[0]], baseline_rates[n[1]]])
                for col in range(joint_arr.shape[1]):
                    np.random.shuffle(joint_arr[:, col])
                    shuffled[idx, sh] = np.mean(np.diff(joint_arr, axis=0))

        fig3, ax3 = plt.subplots(nrows=1, ncols=3, figsize=(6.4*3, 4.8))
        ax31 = plt.subplot(1, 3, 1)
        hist_n, hist_bins, hist_patches = ax31.hist(diff_light1_weight, bins=np.linspace(-7.5, 7.5, 50), histtype='stepfilled', color='#FFFFFF', edgecolor='#000000')
        ax31.axvline(x=0, ls='-.', color='#000000', alpha=.25)
        ax31.plot(diff_light1_weight.mean(), 12, marker='o', color='#000000')
        ax31.set_xlabel(f'light1 - {self.middle_session_type} (spikes/s)')
        ax31.set_ylabel('Number of units')
        p_value = 1 - scipy.stats.norm.cdf((diff_light1_weight.mean() - shuffled[0, :].mean()) / shuffled[0, :].std())
        ax31.text(x=4, y=480, s=f'p={p_value:.2e}')
        axins1 = inset_axes(ax31, width='40%', height='30%', loc=2)
        axins1.hist(shuffled[0, :], bins=np.linspace(-.15, .15, 20), histtype='stepfilled', color='#000000', alpha=.25)
        axins1.axvline(x=np.nanpercentile(shuffled[0, :], 99.5), color='#000000', ls='-.', alpha=.5)
        axins1.plot(diff_light1_weight.mean(), 10, marker='o', color='#000000')
        axins1.set_xticks([-.1, 0, .1, .2])
        axins1.set_yticks([])
        ax32 = plt.subplot(1, 3, 2)
        ax32.hist(diff_weight_light2, bins=hist_bins, histtype='stepfilled', color='#FFFFFF', edgecolor='#000000')
        ax32.axvline(x=0, ls='-.', color='#000000', alpha=.25)
        ax32.plot(diff_weight_light2.mean(), 12, marker='o', color='#000000')
        ax32.set_xlabel(f'{self.middle_session_type} - light2 (spikes/s)')
        p_value_2 = 1 - scipy.stats.norm.cdf((diff_weight_light2.mean() - shuffled[1, :].mean()) / shuffled[1, :].std())
        ax32.text(x=6, y=490, s=f'p={p_value_2:.2f}')
        axins2 = inset_axes(ax32, width='40%', height='30%', loc=2)
        axins2.hist(shuffled[1, :], bins=np.linspace(-.15, .15, 20), histtype='stepfilled', color='#000000', alpha=.25)
        axins2.plot(diff_weight_light2.mean(), 10, marker='o', color='#000000')
        axins2.set_xticks([-.1, 0, .1])
        axins2.set_yticks([])
        ax33 = plt.subplot(1, 3, 3)
        ax33.hist(diff_light1_light2, bins=hist_bins, histtype='stepfilled', color='#FFFFFF', edgecolor='#000000')
        ax33.axvline(x=0, ls='-.', color='#000000', alpha=.25)
        ax33.plot(diff_light1_light2.mean(), 10, marker='o', color='#000000')
        ax33.set_xlabel('light1 - light2 (spikes/s)')
        p_value_3 = 1 - scipy.stats.norm.cdf((diff_light1_light2.mean() - shuffled[2, :].mean()) / shuffled[2, :].std())
        ax33.text(x=4, y=420, s=f'p={p_value_3:.2e}')
        axins3 = inset_axes(ax33, width='40%', height='30%', loc=2)
        axins3.hist(shuffled[2, :], bins=np.linspace(-.15, .15, 20), histtype='stepfilled', color='#000000', alpha=.25)
        axins3.axvline(x=np.nanpercentile(shuffled[2, :], 99.5), color='#000000', ls='-.', alpha=.5)
        axins3.plot(diff_light1_light2.mean(), 7, marker='o', color='#000000')
        axins3.set_xticks([-.1, 0, .1, .2])
        axins3.set_yticks([])
        if self.save_fig:
            if os.path.exists(self.save_dir):
                fig3.savefig(f'{self.save_dir}{os.sep}baseline_change_statistics2.{self.fig_format}', dpi=300)
            else:
                print("Specified save directory doesn't exist. Try again.")
                sys.exit()
        plt.show()

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
        hist_max (int)
            The ymax value for the shuffled histograms; defaults to 275.
        hist_step (int)
            The step for y-values on the shuffled histogramsl defaults to 50.
        ----------

        Returns
        ----------
        weight_feature_distributions (plot)
            A plot with weight peak rate differences distributions.
        ----------
        """

        hist_max = kwargs['hist_max'] if 'hist_max' in kwargs.keys() and type(kwargs['hist_max']) == int else 275
        hist_step = kwargs['hist_step'] if 'hist_step' in kwargs.keys() and type(kwargs['hist_step']) == int else 50

        weight_dict = extract_json_data(json_file=self.weight_json_file,
                                        weight=True,
                                        features=self.chosen_features,
                                        peak_min=self.peak_min,
                                        rate_stability_bound=self.rate_stability_bound,
                                        der=self.der,
                                        ref_dict=self.ref_dict,
                                        middle_session=self.middle_session_type)

        shuffled_dict = make_shuffled_distributions(weight_dict=weight_dict, ref_dict=self.ref_dict,
                                                    middle_session=self.middle_session_type)

        fig = plt.figure(figsize=(12, 20))
        gs_left = [.075, .405, .735]
        gs_right = [.33, .66, .98]
        for gs_idx, gs in enumerate(['peaks', 'information_rates', 'correlations']):
            gs1 = fig.add_gridspec(nrows=12, ncols=3, left=gs_left[gs_idx],
                                   right=gs_right[gs_idx], wspace=.1, hspace=.5)
            ax1 = fig.add_subplot(gs1[:2, :])
            ax2 = fig.add_subplot(gs1[6:8, :])
            ax3 = fig.add_subplot(gs1[3:5, :])
            ax4 = fig.add_subplot(gs1[9:11, :])
            for chosen_feature in self.chosen_features:
                if 'head' in chosen_feature or 'Head' in chosen_feature:
                    axes_list = [ax1, ax2]
                else:
                    axes_list = [ax3, ax4]
                chosen_feature_der = f'{chosen_feature}_{self.der}_der'
                feature_color = [val for key, val in Ratemap.feature_colors.items() if key in chosen_feature][0]
                for ax, specific_feature in zip(axes_list, [chosen_feature, chosen_feature_der]):
                    if gs != 'correlations':
                        light_larger = np.array(weight_dict[specific_feature][gs][self.ref_dict['ref_session']]) \
                                               > np.array(weight_dict[specific_feature][gs][self.middle_session_type])
                        weight_larger = ~light_larger
                        ax.scatter(x=np.array(weight_dict[specific_feature][gs][self.ref_dict['ref_session']])[light_larger],
                                   y=np.array(weight_dict[specific_feature][gs][self.middle_session_type])[light_larger],
                                   color=feature_color, alpha=1, s=10)
                        ax.scatter(x=np.array(weight_dict[specific_feature][gs][self.ref_dict['ref_session']])[weight_larger],
                                   y=np.array(weight_dict[specific_feature][gs][self.middle_session_type])[weight_larger],
                                   color=feature_color, alpha=1, s=10)
                    else:
                        light_larger = np.array(weight_dict[chosen_feature][gs]['lights']) \
                                               > np.array(weight_dict[chosen_feature][gs]['{}-{}'.format(self.ref_dict['ref_session'], self.middle_session_type)])
                        weight_larger = ~light_larger
                        ax.scatter(x=np.array(weight_dict[chosen_feature][gs]['lights'])[light_larger],
                                   y=np.array(weight_dict[chosen_feature][gs]['{}-{}'.format(self.ref_dict['ref_session'], self.middle_session_type)])[light_larger],
                                   color=feature_color, alpha=1, s=10)
                        ax.scatter(x=np.array(weight_dict[chosen_feature][gs]['lights'])[weight_larger],
                                   y=np.array(weight_dict[chosen_feature][gs]['{}-{}'.format(self.ref_dict['ref_session'], self.middle_session_type)])[weight_larger],
                                   color=feature_color, alpha=1, s=10)
            for ax_idx, ax in enumerate([ax1, ax2, ax3, ax4]):
                if ax_idx == 0:
                    ax.set_title(gs.replace('_', ' '))
                if gs == 'peaks':
                    ax.plot([.1, 100], [.1, 100], ls='-.', lw=.5, color='#000000')
                    ax.set_xlim(.1, 100)
                    ax.set_xscale('log')
                    ax.set_ylim(.1, 100)
                    ax.set_yscale('log')
                    ax.tick_params(axis='both', which='major', length=0, labelsize=8, pad=.5)
                    ax.set_xlabel(f'L{self.light_session} peak rate (spikes/s)', labelpad=.1)
                    ax.set_ylabel(f'{self.middle_session_type} rate at L{self.light_session} peak (spikes/s)', labelpad=.1)
                elif gs == 'information_rates':
                    ax.plot([.001, 1], [.001, 1], ls='-.', lw=.5, color='#000000')
                    ax.tick_params(axis='both', which='major', length=0, labelsize=8, pad=.75)
                    ax.set_xlim(.001, 1)
                    ax.set_xscale('log')
                    ax.set_ylim(.001, 1)
                    ax.set_yscale('log')
                    ax.set_xlabel(f'L{self.light_session} info rate (bits/spike)', labelpad=.1)
                    ax.set_ylabel(f'{self.middle_session_type} info rate (bits/spike)', labelpad=.1)
                else:
                    ax.plot([-1.1, 1.1], [-1.1, 1.1], ls='-.', lw=.5, color='#000000')
                    ax.set_xlim(-1.1, 1.1)
                    ax.set_ylim(-1.1, 1.1)
                    ax.set_yticks([-1, -.5, 0, .5, 1])
                    ax.tick_params(axis='both', which='major', length=1, labelsize=8, pad=.75)
                    ax.set_xlabel('L1-L2 correlation', labelpad=.1)
                    ax.set_ylabel(f'{self.middle_session_type}-L2 correlation', labelpad=.1)

            head_col = 0
            back_col = 0
            for chosen_feature in self.chosen_features:
                if 'head' in chosen_feature or 'Head' in chosen_feature:
                    row_list = [2, 8]
                    col = head_col
                else:
                    row_list = [5, 11]
                    col = back_col
                chosen_feature_der = f'{chosen_feature}_{self.der}_der'
                feature_color = [val for key, val in Ratemap.feature_colors.items() if key in chosen_feature][0]
                for row, specific_feature in zip(row_list, [chosen_feature, chosen_feature_der]):
                    ax = fig.add_subplot(gs1[row, col])
                    ax.hist(shuffled_dict[specific_feature][gs]['null_differences'], bins=10,
                            histtype='stepfilled', color='#808080', edgecolor='#000000', alpha=.5)
                    ax.axvline(x=np.nanpercentile(shuffled_dict[specific_feature][gs]['null_differences'], 0.5), color='#000000', ls='-.', lw=.5)
                    ax.axvline(x=np.nanpercentile(shuffled_dict[specific_feature][gs]['null_differences'], 99.5), color='#000000', ls='-.', lw=.5)
                    ax.set_ylim(ymax=hist_max)
                    ax.plot(shuffled_dict[specific_feature][gs]['true_difference'], 0+.05*hist_max, marker='o', color=feature_color, markersize=5)
                    if col < 1:
                        ax.set_ylabel('Shuffled count', labelpad=.1, fontsize=6)
                        ax.set_yticks(np.arange(0, hist_max, hist_step))
                    elif col == 1:
                        if gs == 'peaks':
                            ax.set_xlabel(f'L{self.light_session} - {self.middle_session_type} difference (spikes/s)', labelpad=.5, fontsize=6)
                        elif gs == 'information_rates':
                            ax.set_xlabel(f'L{self.light_session} - {self.middle_session_type} difference (bits/spike)', labelpad=.5, fontsize=6)
                        else:
                            ax.set_xlabel(f'L{self.light_session} - {self.middle_session_type} difference (rho)', labelpad=.5, fontsize=6)
                        ax.set_yticklabels([])
                    else:
                        ax.set_yticklabels([])
                    ax.tick_params(axis='both', which='both', labelsize=5)
                if 'head' in chosen_feature or 'Head' in chosen_feature:
                    head_col += 1
                else:
                    back_col += 1
        if self.save_fig:
            if os.path.exists(self.save_dir):
                fig.savefig(f'{self.save_dir}{os.sep}weight_features_comparisons.{self.fig_format}', dpi=300)
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
        ----------

        Returns
        ----------
        weight_statistics (plot)
            A plot with weight peak rate differences statistics.
        ----------
        """

        weight_dict = extract_json_data(json_file=self.weight_json_file,
                                        weight=True,
                                        features=self.chosen_features,
                                        peak_min=self.peak_min,
                                        rate_stability_bound=self.rate_stability_bound,
                                        der=self.der,
                                        ref_dict=self.ref_dict,
                                        middle_session=self.middle_session_type)

        shuffled_dict = make_shuffled_distributions(weight_dict=weight_dict, ref_dict=self.ref_dict,
                                                    middle_session=self.middle_session_type)

        weight_stats_dict={'features': {'peaks': [], 'information_rates': [], 'correlations': [],
                                        'z_peaks': [], 'z_information_rates': [], 'z_correlations': [],
                                        'cl_n': [], 'names': [], 'feature_colors': []},
                           'features_der': {'peaks': [], 'information_rates': [], 'correlations': [],
                                            'z_peaks': [], 'z_information_rates': [], 'z_correlations': [],
                                            'cl_n': [], 'names': [], 'feature_colors': []}}

        for feature in weight_dict.keys():
            if 'der' not in feature:
                weight_stats_dict['features']['peaks'].append(shuffled_dict[feature]['peaks']['p-value'])
                weight_stats_dict['features']['information_rates'].append(shuffled_dict[feature]['information_rates']['p-value'])
                weight_stats_dict['features']['correlations'].append(shuffled_dict[feature]['correlations']['p-value'])
                weight_stats_dict['features']['z_peaks'].append(shuffled_dict[feature]['peaks']['z-value'])
                weight_stats_dict['features']['z_information_rates'].append(shuffled_dict[feature]['information_rates']['z-value'])
                weight_stats_dict['features']['z_correlations'].append(shuffled_dict[feature]['correlations']['z-value'])
                weight_stats_dict['features']['cl_n'].append(len(weight_dict[feature]['peaks'][self.middle_session_type]))
                weight_stats_dict['features']['names'].append(feature)
                weight_stats_dict['features']['feature_colors'].append([val for key, val in Ratemap.feature_colors.items() if key in feature][0])
            else:
                weight_stats_dict['features_der']['peaks'].append(shuffled_dict[feature]['peaks']['p-value'])
                weight_stats_dict['features_der']['information_rates'].append(shuffled_dict[feature]['information_rates']['p-value'])
                weight_stats_dict['features_der']['correlations'].append(shuffled_dict[feature]['correlations']['p-value'])
                weight_stats_dict['features_der']['z_peaks'].append(shuffled_dict[feature]['peaks']['z-value'])
                weight_stats_dict['features_der']['z_information_rates'].append(shuffled_dict[feature]['information_rates']['z-value'])
                weight_stats_dict['features_der']['z_correlations'].append(shuffled_dict[feature]['correlations']['z-value'])
                weight_stats_dict['features_der']['cl_n'].append(len(weight_dict[feature]['peaks'][self.middle_session_type]))
                weight_stats_dict['features_der']['names'].append(feature)
                weight_stats_dict['features_der']['feature_colors'].append([val for key, val in Ratemap.feature_colors.items() if key in feature][0])

        print(weight_stats_dict)

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(30, 5))
        ax1 = plt.subplot(111)
        left_point, middle_point, right_point = 2, 3, 4
        cmap=plt.cm.seismic_r
        normal_range = plt.Normalize(vmin=-4, vmax=0)
        for idx, feature in enumerate(weight_stats_dict['features']['names']):
            probabilities=np.log10([weight_stats_dict['features']['peaks'][idx],
                                    weight_stats_dict['features']['information_rates'][idx],
                                    weight_stats_dict['features']['correlations'][idx],
                                    weight_stats_dict['features_der']['peaks'][idx],
                                    weight_stats_dict['features_der']['information_rates'][idx],
                                    weight_stats_dict['features_der']['correlations'][idx]])
            colors = plt.cm.seismic_r(normal_range(probabilities))
            for y_idx, y_point in enumerate([1, 3]):
                if y_idx == 0:
                    co = colors[3:]
                else:
                    co = colors[:3]
                p = patches.Polygon(xy=np.array([[left_point, y_point], [right_point, y_point], [middle_point, (2*np.sqrt(3)/6)+y_point]]),
                                    closed=True,
                                    ec='#000000',
                                    fc=co[0])
                p2 = patches.Polygon(xy=np.array([[left_point, y_point], [middle_point, (2*np.sqrt(3)/2)+y_point], [middle_point, (2*np.sqrt(3)/6)+y_point]]),
                                     closed=True,
                                     ec='#000000',
                                     fc=co[1])
                p3 = patches.Polygon(xy=np.array([[right_point, y_point], [middle_point, (2*np.sqrt(3)/2)+y_point], [middle_point, (2*np.sqrt(3)/6)+y_point]]),
                                     closed=True,
                                     ec='#000000',
                                     fc=co[2])
                ax1.add_patch(p)
                ax1.add_patch(p2)
                ax1.add_patch(p3)
                ax1.text(x=left_point+.3, y=5.5, s=feature.replace('_', ' '), fontsize=8)
                ax1.text(x=left_point+.5, y=5, s='n_clusters={}'.format(weight_stats_dict['features']['cl_n'][idx]), fontsize=8)
                ax1.text(x=left_point+.5, y=.5, s='n_clusters={}'.format(weight_stats_dict['features_der']['cl_n'][idx]), fontsize=8)
            left_point += 3
            middle_point += 3
            right_point += 3
        ax1.text(x=.6, y=4, s='Feature', fontsize=8)
        ax1.text(x=.6, y=2, s='Derivative', fontsize=8)
        ax1.set_xlim(0, 35)
        ax1.set_ylim(0, 6)
        ax1.spines['bottom'].set_visible(False)
        ax1.spines['left'].set_visible(False)
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.set_xticks([])
        ax1.set_yticks([])
        cax, _ = cbar.make_axes(ax1)
        color_bar = cbar.ColorbarBase(cax, cmap=cmap, norm=normal_range)
        color_bar.set_label('log$_{10}$(p-value)')
        color_bar.ax.tick_params(size=0)
        if self.save_fig:
            if os.path.exists(self.save_dir):
                fig.savefig(f'{self.save_dir}{os.sep}weight_features_statistics.{self.fig_format}', dpi=300)
            else:
                print("Specified save directory doesn't exist. Try again.")
                sys.exit()
        plt.show()
