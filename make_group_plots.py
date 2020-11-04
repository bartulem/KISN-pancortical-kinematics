# -*- coding: utf-8 -*-

"""

@author: bartulem

Make group plots.

"""

from neural_activity import Spikes
from select_clusters import ClusterFinder
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import wilcoxon
from scipy.stats import sem


class PlotGroupResults:
    def __init__(self, session_list=[], cluster_groups_dir='', sp_profiles_csv='',
                 save_fig=False, fig_format='png', save_dir='/home/bartulm/Downloads',
                 decoding_dir='', animal_ids={'frank': '26473', 'johnjohn': '26471', 'kavorka': '26525'}):
        self.session_list = session_list
        self.cluster_groups_dir = cluster_groups_dir
        self.sp_profiles_csv = sp_profiles_csv
        self.save_fig = save_fig
        self.fig_format = fig_format
        self.save_dir = save_dir
        self.decoding_dir = decoding_dir
        self.animal_ids = animal_ids

    def sound_stim_summary(self, **kwargs):
        """
        Description
        ----------
        This method plots the sound stimulation effect for a group of cells (can be across
        different animals).
        ----------

        Parameters
        ----------
        **kwargs (dictionary)
        relevant_areas (list)
            Areas of interest; defaults to ['A'].
        relevant_cluster_types (str)
            Cluster types to consider; defaults to 'good'.
        bin_size_ms (int)
            The bin size of the PETH; defaults to 50 (ms).
        window_size (int/float)
            The unilateral window size; defaults to 10 (seconds).
        mod_idx_time (int / float)
            The time before/after sound stim to calculate the index in; defaults to 500 (ms).
        smooth (bool)
            Smooth PETHs; defaults to False.
        smooth_sd (int)
            The SD of the smoothing window; defaults to 1 (bin).
        critical_p_value (float)
            The p_value below something is considered statistically significant; defaults to 0.01
        get_most_modulated (bool)
            Print the five most modulated (suppressed and excited) clusters; defaults to False.
        to_plot (bool)
            Yey or ney on the plotting; defaults to False.
        profile_colors (dict)
            What colors to use for each spiking profile; defaults to {'RS': '#698B69', 'FS': '#9BCD9B'}.
        ----------

        Returns
        ----------
        sound_stim_scatter (fig)
            A scatter plot of the sound stimulation effects.
        ----------
        """

        relevant_areas = kwargs['relevant_areas'] if 'relevant_areas' in kwargs.keys() and type(kwargs['relevant_areas']) == list else ['A']
        relevant_cluster_types = kwargs['relevant_cluster_types'] if 'relevant_cluster_types' in kwargs.keys() and type(kwargs['relevant_cluster_types']) == str else 'good'
        bin_size_ms = kwargs['bin_size_ms'] if 'bin_size_ms' in kwargs.keys() and type(kwargs['bin_size_ms']) == int else 50
        window_size = kwargs['window_size'] if 'window_size' in kwargs.keys() and (type(kwargs['window_size']) == int or type(kwargs['window_size']) == float) else 10
        mod_idx_time = kwargs['mod_idx_time'] if 'mod_idx_time' in kwargs.keys() and (type(kwargs['mod_idx_time']) == int or type(kwargs['mod_idx_time']) == float) else 500
        smooth = kwargs['smooth'] if 'smooth' in kwargs.keys() and type(kwargs['smooth']) == bool else False
        smooth_sd = kwargs['smooth_sd'] if 'smooth_sd' in kwargs.keys() and type(kwargs['smooth_sd']) == int else 1
        critical_p_value = kwargs['critical_p_value'] if 'critical_p_value' in kwargs.keys() and type(kwargs['critical_p_value']) == float else .01
        get_most_modulated = kwargs['get_most_modulated'] if 'get_most_modulated' in kwargs.keys() and type(kwargs['get_most_modulated']) == bool else False
        to_plot = kwargs['to_plot'] if 'to_plot' in kwargs.keys() and type(kwargs['to_plot']) == bool else False
        profile_colors = kwargs['profile_colors'] if 'profile_colors' in kwargs.keys() and type(kwargs['profile_colors']) == dict else {'RS': '#698B69', 'FS': '#9BCD9B'}

        if not os.path.exists(self.cluster_groups_dir):
            print(f"Invalid location for directory {self.cluster_groups_dir}. Please try again.")
            sys.exit()

        # get PETH data for chosen clusters in designated sessions
        sound_stim_data = {}
        if len(self.session_list) > 0:
            for one_session in self.session_list:
                if os.path.exists(one_session):
                    relevant_session_clusters = ClusterFinder(session=one_session,
                                                              cluster_groups_dir=self.cluster_groups_dir).get_desired_clusters(filter_by_area=relevant_areas,
                                                                                                                               filter_by_cluster_type=relevant_cluster_types)
                    session_name, peth = Spikes(input_file=one_session).get_peths(get_clusters=relevant_session_clusters,
                                                                                  bin_size_ms=bin_size_ms,
                                                                                  window_size=window_size,
                                                                                  smooth=smooth,
                                                                                  smooth_sd=smooth_sd)
                    sound_stim_data[session_name] = peth
                else:
                    print(f"Invalid location for file {one_session}. Please try again.")
                    sys.exit()
        else:
            print("No session provided.")
            sys.exit()

        # prepare the arrays for plotting and calculate all the statistics
        total_num_clusters = np.sum([len(sound_stim_data[session].keys()) for session in sound_stim_data.keys()])
        plot_array = np.zeros((total_num_clusters, 2*int(round(window_size / (bin_size_ms / 1e3)))))
        statistics_dict = {}
        cell_id = 0
        for session in sound_stim_data.keys():
            for cluster in sound_stim_data[session].keys():
                # # get mean PETH for plotting
                all_trials = sound_stim_data[session][cluster]['peth']
                averaged_trials = np.nanmean(all_trials, axis=0)

                # normalize each average by its peak
                plot_array[cell_id, :] = averaged_trials / np.max(averaged_trials)

                # # get all the details for the statistics dict
                statistics_dict[cell_id] = {}

                # get session and cluster id
                statistics_dict[cell_id]['session'] = session
                statistics_dict[cell_id]['cell_id'] = cluster

                # get sound modulation index
                zero_bin = averaged_trials.shape[0] // 2
                bins_to_skip = mod_idx_time // bin_size_ms
                sound_bin_start = zero_bin + bins_to_skip
                sound_bin_end = sound_bin_start + bins_to_skip
                pre_bin_end = zero_bin - bins_to_skip
                pre_bin_start = pre_bin_end - bins_to_skip
                statistics_dict[cell_id]['sound_modulation_index'] = (averaged_trials[sound_bin_start:sound_bin_end].mean() - averaged_trials[pre_bin_start:pre_bin_end].mean()) / \
                                                                     (averaged_trials[sound_bin_start:sound_bin_end].mean() + averaged_trials[pre_bin_start:pre_bin_end].mean())

                # get statistical significance (no sound vs. sound)
                trials_array = np.zeros((all_trials.shape[0], 2))
                for trial in range(all_trials.shape[0]):
                    trials_array[trial, :] = [all_trials[trial, pre_bin_start:pre_bin_end].mean(), all_trials[trial, sound_bin_start:sound_bin_end].mean()]
                statistics_dict[cell_id]['p_value'] = wilcoxon(x=trials_array[:, 0], y=trials_array[:, 1], zero_method='zsplit')[1]

                cell_id += 1

        # get total number of clusters in the dataset
        total_cluster_number = len(statistics_dict.keys())

        # get cluster profiles
        if not os.path.exists(self.sp_profiles_csv):
            print(f"Invalid location for file {self.sp_profiles_csv}. Please try again.")
            sys.exit()
        else:
            profile_data = pd.read_csv(self.sp_profiles_csv)

        # separate significantly suppressed and sound excited clusters
        modulated_clusters = {'suppressed': {}, 'excited': {}}
        count_dict = {'sign_excited_rs': 0,
                      'sign_excited_fs': 0,
                      'sign_suppressed_rs': 0,
                      'sign_suppressed_fs': 0,
                      'ns_rs': 0,
                      'ns_fs': 0}

        for cluster in statistics_dict.keys():
            session_id = statistics_dict[cluster]['session']
            file_animal = [animal for animal in ClusterFinder.probe_site_areas.keys() if animal in session_id][0]
            file_bank = [bank for bank in ['distal', 'intermediate'] if bank in session_id][0]
            file_date = session_id[session_id.find('20')-4:session_id.find('20')+2]
            for idx, row in profile_data.iterrows():
                if row[0] == f'{file_animal}_{file_date}_{file_bank}' and row[1] == statistics_dict[cluster]['cell_id']:
                    cl_profile = row[-1]
                    break
            if statistics_dict[cluster]['sound_modulation_index'] < 0 and statistics_dict[cluster]['p_value'] < critical_p_value:
                modulated_clusters['suppressed'][cluster] = statistics_dict[cluster]
                if cl_profile == 'RS':
                    count_dict['sign_suppressed_rs'] += 1
                else:
                    count_dict['sign_suppressed_fs'] += 1
                """if statistics_dict[cluster]['sound_modulation_index'] < -.5 and cl_profile == 'FS':
                    print(statistics_dict[cluster]['session'], statistics_dict[cluster]['cell_id'], statistics_dict[cluster]['sound_modulation_index'], cl_profile)"""
            elif statistics_dict[cluster]['sound_modulation_index'] > 0 and statistics_dict[cluster]['p_value'] < critical_p_value:
                modulated_clusters['excited'][cluster] = statistics_dict[cluster]
                if cl_profile == 'RS':
                    count_dict['sign_excited_rs'] += 1
                else:
                    count_dict['sign_excited_fs'] += 1
                """if statistics_dict[cluster]['sound_modulation_index'] > .5 and cl_profile == 'FS':
                    print(statistics_dict[cluster]['session'], statistics_dict[cluster]['cell_id'], statistics_dict[cluster]['sound_modulation_index'], cl_profile)"""
            elif statistics_dict[cluster]['p_value'] >= critical_p_value:
                if cl_profile == 'RS':
                    count_dict['ns_rs'] += 1
                else:
                    count_dict['ns_fs'] += 1

        # order clusters in each category separately
        cluster_order_suppressed = [item[0] for item in sorted(modulated_clusters['suppressed'].items(), key=lambda i: i[1]['sound_modulation_index'])]
        cluster_order_excited = [item[0] for item in sorted(modulated_clusters['excited'].items(), key=lambda i: i[1]['sound_modulation_index'], reverse=True)]

        # find most modulated cells
        if get_most_modulated:
            print(f"There are {total_cluster_number} clusters in this dataset, and these are the category counts: {count_dict}")
            for idx in range(20):
                print(f"Number {idx+1} on the suppressed list: {statistics_dict[cluster_order_suppressed[idx]]['session']}, "
                      f"{statistics_dict[cluster_order_suppressed[idx]]['cell_id']}, SMI: {statistics_dict[cluster_order_suppressed[idx]]['sound_modulation_index']}")
                print(f"Number {idx+1} on the excited list: {statistics_dict[cluster_order_excited[idx]]['session']}, "
                      f"{statistics_dict[cluster_order_excited[idx]]['cell_id']}, SMI: {statistics_dict[cluster_order_excited[idx]]['sound_modulation_index']}")

        # re-order cluster array by sound modulation index (from lowest to highest value and vice-versa for excited clusters)
        plot_array_ordered_suppressed = plot_array.take(indices=cluster_order_suppressed, axis=0)
        plot_array_ordered_excited = plot_array.take(indices=cluster_order_excited, axis=0)

        # plot
        if to_plot:
            # make group mean activity plot
            fig = plt.figure(figsize=(8, 6), dpi=300, tight_layout=True)
            ax1 = fig.add_subplot(121, label='1')
            ax1.imshow(plot_array_ordered_suppressed, aspect='auto', vmin=0, vmax=1, cmap='cividis')
            ax2 = fig.add_subplot(121, label='2', frame_on=False)
            ax2.plot(range(plot_array_ordered_suppressed.shape[1]), plot_array_ordered_suppressed.mean(axis=0), ls='-', lw=3, c='#1E90FF')
            ax2.set_xlim(0, 400)
            ax2.set_xticks([])
            ax2.set_yticks([])
            ax3 = fig.add_subplot(122, label='3')
            im = ax3.imshow(plot_array_ordered_excited, aspect='auto', vmin=0, vmax=1, cmap='cividis')
            ax4 = fig.add_subplot(122, label='4', frame_on=False)
            ax4.plot(range(plot_array_ordered_excited.shape[1]), plot_array_ordered_excited.mean(axis=0), ls='-', lw=3, c='#FF6347')
            ax4.set_xlim(0, 400)
            ax4.set_xticks([])
            ax4.set_yticks([])
            cb_ax = fig.add_axes([0.9, 0.05, 0.01, 0.3])
            cbar = fig.colorbar(im, orientation='vertical', cax=cb_ax, shrink=.3)
            cbar.set_label('Normalized activity')
            cbar.ax.tick_params(size=0)
            ax1.set_xticks(np.arange(0, 401, 100))
            ax3.set_xticks(np.arange(0, 401, 100))
            ax1.set_xticklabels([-10, -5, 0, 5, 10])
            ax3.set_xticklabels([-10, -5, 0, 5, 10])
            ax1.set_xlabel('Time relative to sound onset (s)')
            ax3.set_xlabel('Time relative to sound onset (s)')
            ax1.tick_params(axis='y', length=0)
            ax3.tick_params(axis='y', length=0)
            ax1.set_ylabel('Cell number')
            for side in ['right', 'top', 'left', 'bottom']:
                ax1.spines[side].set_visible(False)
                ax3.spines[side].set_visible(False)
            if self.save_fig:
                if os.path.exists(self.save_dir):
                    fig.savefig(f'{self.save_dir}{os.sep}sound_peth_group.{self.fig_format}')
                else:
                    print("Specified save directory doesn't exist. Try again.")
                    sys.exit()
            plt.show()

            # make pie chart
            size = .3
            labels = ['RS', 'FS']
            inner_colors = ['#1E90FF', '#FF6347', '#DEDEDE']*2
            outer_colors = [profile_colors['RS'], profile_colors['FS']]
            pie_values = np.array([[count_dict['sign_suppressed_rs'], count_dict['sign_excited_rs'], count_dict['ns_rs']],
                                   [count_dict['sign_suppressed_fs'], count_dict['sign_excited_fs'], count_dict['ns_fs']]])

            fig2, ax5 = plt.subplots(nrows=1, ncols=1, figsize=(8, 6), dpi=300)
            ax5.pie(pie_values.sum(axis=1), radius=1, colors=outer_colors, shadow=False,
                    autopct='%1.1f%%', labels=labels, wedgeprops=dict(width=size, edgecolor='#FFFFFF'))
            ax5.pie(pie_values.flatten(), radius=1-size, colors=inner_colors,
                    shadow=False, wedgeprops=dict(width=size, edgecolor='#FFFFFF'))
            ax5.set(aspect="equal", title='Sound modulated cells summary`')
            if self.save_fig:
                if os.path.exists(self.save_dir):
                    fig2.savefig(f'{self.save_dir}{os.sep}sound_modulation_summary.{self.fig_format}')
                else:
                    print("Specified save directory doesn't exist. Try again.")
                    sys.exit()
            plt.show()

            # make SMI histograms
            smi = [statistics_dict[cluster]['sound_modulation_index'] for cluster in statistics_dict.keys()]
            smi_neg = [statistics_dict[cluster]['sound_modulation_index'] for cluster in statistics_dict.keys()
                       if (statistics_dict[cluster]['sound_modulation_index'] < 0 and statistics_dict[cluster]['p_value'] < .01)]
            smi_pos = [statistics_dict[cluster]['sound_modulation_index'] for cluster in statistics_dict.keys()
                       if (statistics_dict[cluster]['sound_modulation_index'] > 0 and statistics_dict[cluster]['p_value'] < .01)]
            fig3 = plt.figure(figsize=(8, 6), dpi=300)
            bins = np.linspace(-1, 1, 20)
            ax6 = fig3.add_subplot(111, label='6')
            ax6.hist(smi, bins=bins, color='#DEDEDE', alpha=.6, edgecolor='#000000')
            ax6.hist(smi_neg, bins=bins, color='#1E90FF', alpha=.6)
            ax6.hist(smi_pos, bins=bins, color='#FF6347', alpha=.6)
            ax6.set_xlabel('Sound modulation index')
            ax6.set_ylabel('Number of cells')
            for side in ['right', 'top']:
                ax6.spines[side].set_visible(False)
            if self.save_fig:
                if os.path.exists(self.save_dir):
                    fig3.savefig(f'{self.save_dir}{os.sep}sound_modulation_distribution.{self.fig_format}')
                else:
                    print("Specified save directory doesn't exist. Try again.")
                    sys.exit()
            plt.show()

    def decoding_summary(self, **kwargs):

        file_dict = {'data': {'A': [], 'V': []}, 'shuffled': {'A': [], 'V': []}}
        if not os.path.exists(self.decoding_dir):
            print(f"Invalid location for directory {self.decoding_dir}. Please try again.")
            sys.exit()
        else:
            for decoding_file_name in os.listdir(self.decoding_dir):
                if 'shuffled' in decoding_file_name:
                    if 'A' in decoding_file_name:
                        file_dict['shuffled']['A'].append(decoding_file_name)
                    else:
                        file_dict['shuffled']['V'].append(decoding_file_name)
                else:
                    if 'A' in decoding_file_name:
                        file_dict['data']['A'].append(decoding_file_name)
                    else:
                        file_dict['data']['V'].append(decoding_file_name)

        # sort dict by file name
        for data_type in file_dict.keys():
            for data_area in file_dict[data_type].keys():
                file_dict[data_type][data_area].sort()

        # load the data
        decoding_data = {'data': {'A': {}, 'V': {}}, 'shuffled': {'A': {}, 'V': {}}}
        for data_type in decoding_data.keys():
            for data_area in decoding_data[data_type].keys():
                for file_idx, one_file in enumerate(file_dict[data_type][data_area]):
                    decoding_data[data_type][data_area][list(self.animal_ids.keys())[file_idx]] = np.load(f'{self.decoding_dir}{os.sep}{one_file}')

        # get data to plot
        plot_data = {'A': {'decoding_accuracy': {'mean': {}, 'sem': {}}, 'shuffled': np.array([[1000., 0.]]*5)},
                     'V': {'decoding_accuracy': {'mean': {}, 'sem': {}}, 'shuffled': np.array([[1000., 0.]]*5)}}
        for area in decoding_data['data']:
            for animal in decoding_data['data'][area].keys():
                plot_data[area]['decoding_accuracy']['mean'][animal] = decoding_data['data'][area][animal].mean(axis=1)
                plot_data[area]['decoding_accuracy']['sem'][animal] = sem(decoding_data['data'][area][animal], axis=1)
                down_percentiles = np.percentile(decoding_data['shuffled'][area][animal], q=.5, axis=1)
                for d_idx, d_per in enumerate(down_percentiles):
                    if d_per < plot_data[area]['shuffled'][d_idx, 0]:
                        plot_data[area]['shuffled'][d_idx, 0] = d_per
                up_percentiles = np.percentile(decoding_data['shuffled'][area][animal], q=99.5, axis=1)
                for u_idx, u_per in enumerate(up_percentiles):
                    if u_per > plot_data[area]['shuffled'][u_idx, 1]:
                        plot_data[area]['shuffled'][u_idx, 1] = u_per

        # plot
        x_values = np.array([5, 10, 20, 50, 100])
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 5), dpi=300)
        ax[0].errorbar(x=x_values, y=plot_data['A']['decoding_accuracy']['mean']['kavorka'], yerr=plot_data['A']['decoding_accuracy']['sem']['kavorka']*3,
                     color='#000000', fmt='-o', label=f"#{self.animal_ids['kavorka']}")
        ax[0].errorbar(x=x_values, y=plot_data['A']['decoding_accuracy']['mean']['frank'], yerr=plot_data['A']['decoding_accuracy']['sem']['frank']*3,
                     color='#000000', fmt='-^', label=f"#{self.animal_ids['frank']}")
        ax[0].errorbar(x=x_values, y=plot_data['A']['decoding_accuracy']['mean']['johnjohn'], yerr=plot_data['A']['decoding_accuracy']['sem']['johnjohn']*3,
                     color='#000000', fmt='-s', label=f"#{self.animal_ids['johnjohn']}")
        ax[0].fill_between(x=x_values, y1=plot_data['A']['shuffled'][:, 0], y2=plot_data['A']['shuffled'][:, 1], color='grey', alpha=.25)
        ax[0].set_ylim(.5, 1)
        ax[0].legend()
        ax[0].set_title('Decoding of sound stim by A cells')
        ax[0].set_xlabel('Number of cells')
        ax[0].set_ylabel('Decoding accuracy')

        ax[1].errorbar(x=x_values, y=plot_data['V']['decoding_accuracy']['mean']['kavorka'], yerr=plot_data['V']['decoding_accuracy']['sem']['kavorka']*3,
                     color='#000000', fmt='-o', label=f"#{self.animal_ids['kavorka']}")
        ax[1].errorbar(x=x_values, y=plot_data['V']['decoding_accuracy']['mean']['frank'], yerr=plot_data['V']['decoding_accuracy']['sem']['frank']*3,
                     color='#000000', fmt='-^', label=f"#{self.animal_ids['frank']}")
        ax[1].errorbar(x=x_values, y=plot_data['V']['decoding_accuracy']['mean']['johnjohn'], yerr=plot_data['V']['decoding_accuracy']['sem']['johnjohn']*3,
                     color='#000000', fmt='-s', label=f"#{self.animal_ids['johnjohn']}")
        ax[1].fill_between(x=x_values, y1=plot_data['V']['shuffled'][:, 0], y2=plot_data['V']['shuffled'][:, 1], color='grey', alpha=.25)
        ax[1].set_ylim(.5, 1)
        ax[1].legend()
        ax[1].set_title('Decoding of sound stim by V cells')
        ax[1].set_xlabel('Number of cells')
        ax[1].set_ylabel('Decoding accuracy')
        plt.show()
