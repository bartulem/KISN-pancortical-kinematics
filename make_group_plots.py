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


class PlotGroupResults:
    def __init__(self, session_list=[], cluster_groups_dir=0, sp_profiles_csv=0):
        self.session_list = session_list
        self.cluster_groups_dir = cluster_groups_dir
        self.sp_profiles_csv = sp_profiles_csv

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
        profile_colors (dict)
            What colors to use for each profile in the scatter plot; defaults to {'RS': '#B0B0B0', 'FS': '#000000'}.
        save_fig (bool)
            Save the figure or not; defaults to False.
        fig_format (str)
            The format of the figure; defaults to 'png'.
        save_dir (bool)
            Directory to save the figure in; defaults to '/home/bartulm/Downloads'.
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
        profile_colors = kwargs['profile_colors'] if 'profile_colors' in kwargs.keys() and type(kwargs['profile_colors']) == dict else {'RS': '#B0B0B0', 'FS': '#000000'}
        save_fig = kwargs['save_fig'] if 'save_fig' in kwargs.keys() and type(kwargs['save_fig']) == bool else False
        fig_format = kwargs['fig_format'] if 'fig_format' in kwargs.keys() and type(kwargs['fig_format']) == str else 'png'
        save_dir = kwargs['save_dir'] if 'save_dir' in kwargs.keys() and type(kwargs['save_dir']) == str else '/home/bartulm/Downloads'

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
                averaged_trials = all_trials.mean(axis=0)

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

        # order cells by sound modulation index
        cluster_order = [item[0] for item in sorted(statistics_dict.items(), key=lambda i: i[1]['sound_modulation_index'])]

        # re-order cluster array by sound modulation index (from lowest to highest value)
        plot_array_ordered = plot_array.take(indices=cluster_order, axis=0)

        # get SMIs and p_values and order them according to the cluster order
        all_smi = []
        all_pval = []
        for cell in statistics_dict.keys():
            all_smi.append(statistics_dict[cell]['sound_modulation_index'])
            all_pval.append(statistics_dict[cell]['p_value'])
        all_smi_ordered = [all_smi[i] for i in cluster_order]
        all_pval_ordered = [all_pval[i] for i in cluster_order]

        # get cluster profiles
        if not os.path.exists(self.sp_profiles_csv):
            print(f"Invalid location for file {self.sp_profiles_csv}. Please try again.")
            sys.exit()
        else:
            profile_data = pd.read_csv(self.sp_profiles_csv)

        # plot
        fig = plt.figure(figsize=(4, 10), dpi=300)
        ax1 = fig.add_subplot(111)
        im = ax1.imshow(plot_array_ordered, aspect='auto', vmin=0, vmax=1, cmap='cividis')
        for cell_idx in range(plot_array_ordered.shape[0]):
            # find if cluster is suppressed/excited by sound
            if all_smi_ordered[cell_idx] > 0 and all_pval_ordered[cell_idx] < 0.01:
                significance_color = '#FF6347'
            elif all_smi_ordered[cell_idx] < 0 and all_pval_ordered[cell_idx] < 0.01:
                significance_color = '#1E90FF'
            else:
                significance_color = '#C9C9C9'
            ax1.plot(405, cell_idx, 'o', ms=1.5, c=significance_color)

            # get animal name, bank id and date of session
            session_id = statistics_dict[cell_idx]['session']
            file_animal = [animal for animal in ClusterFinder.probe_site_areas.keys() if animal in session_id][0]
            file_bank = [bank for bank in ['distal', 'intermediate'] if bank in session_id][0]
            file_date = session_id[session_id.find('20')-4:session_id.find('20')+2]

            # find if cluster is RS or FS
            for idx, row in profile_data.iterrows():
                if row[0] == f'{file_animal}_{file_date}_{file_bank}' and row[1] == statistics_dict[cell_idx]['cell_id']:
                    cl_profile = row[-1]
                    break
            ax1.plot(415, cell_idx, 'o', ms=1.5, c=profile_colors[cl_profile])
        ax1.set_xticks(np.arange(0, 401, 100))
        ax1.set_xticklabels([-10, -5, 0, 5, 10])
        ax1.set_xlabel('Time relative to sound onset (s)')
        ax1.tick_params(axis='y', length=0)
        ax1.set_ylabel('Cell number')
        cbar = fig.colorbar(im, shrink=.2)
        cbar.set_label('Normalized activity')
        ax1.spines['right'].set_visible(False)
        ax1.spines['top'].set_visible(False)
        ax1.spines['left'].set_visible(False)
        ax1.spines['bottom'].set_visible(False)
        cbar.ax.tick_params(size=0)
        if save_fig:
            if os.path.exists(save_dir):
                fig.savefig(f'{save_dir}{os.sep}sound_peth_group.{fig_format}')
            else:
                print("Specified save directory doesn't exist. Try again.")
                sys.exit()
        plt.show()
