# -*- coding: utf-8 -*-

"""

@author: bartulem

Make group plots.

"""

import io
import os
import sys
import re
import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from scipy.stats import wilcoxon
from scipy.stats import sem
from scipy.stats import pearsonr
import decode_events
from sessions2load import Session
from neural_activity import Spikes
from neural_activity import gaussian_smoothing
from select_clusters import ClusterFinder
from define_spiking_profile import get_cluster_spiking_profiles


class PlotGroupResults:
    def __init__(self, session_list=[], cluster_groups_dir='', sp_profiles_csv='',
                 save_fig=False, fig_format='png', save_dir='/home/bartulm/Downloads',
                 decoding_dir='', animal_ids={'frank': '26473', 'johnjohn': '26471', 'kavorka': '26525'},
                 relevant_areas=['A'], relevant_cluster_types='good',
                 bin_size_ms=50, window_size=10, smooth=False, smooth_sd=1, to_plot=False,
                 input_012_list=[], pkl_load_dir='', critical_p_value=.01,
                 profile_colors={'RS': '#698B69', 'FS': '#9BCD9B'}, modulation_indices_dir='',
                 all_animals_012={}):
        self.session_list = session_list
        self.cluster_groups_dir = cluster_groups_dir
        self.sp_profiles_csv = sp_profiles_csv
        self.save_fig = save_fig
        self.fig_format = fig_format
        self.save_dir = save_dir
        self.decoding_dir = decoding_dir
        self.animal_ids = animal_ids
        self.relevant_areas = relevant_areas
        self.relevant_cluster_types = relevant_cluster_types
        self.bin_size_ms = bin_size_ms
        self.window_size = window_size
        self.smooth = smooth
        self.smooth_sd = smooth_sd
        self.to_plot = to_plot
        self.input_012_list = input_012_list
        self.pkl_load_dir = pkl_load_dir
        self.critical_p_value = critical_p_value
        self.profile_colors = profile_colors
        self.modulation_indices_dir = modulation_indices_dir
        self.all_animals_012 = all_animals_012

    def sound_modulation_summary(self, **kwargs):
        """
        Description
        ----------
        This method plots the sound stimulation effect for a group of cells (can be across
        different animals). PETHs were smoothed with a Gaussian of 1 bin width.
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
        mean_activity_plot (fig)
            A "snake" plot of the sound stimulation effect.
        pie_chart (fig)
            A pie chart breakdown of sound modulation effects for RS and FS cells.
        SMI_histogram (fig)
            A histogram of the SMIs.
        ----------
        """

        mod_idx_time = kwargs['mod_idx_time'] if 'mod_idx_time' in kwargs.keys() and (type(kwargs['mod_idx_time']) == int or type(kwargs['mod_idx_time']) == float) else 500
        get_most_modulated = kwargs['get_most_modulated'] if 'get_most_modulated' in kwargs.keys() and type(kwargs['get_most_modulated']) == bool else False

        if not os.path.exists(self.cluster_groups_dir):
            print(f"Invalid location for directory {self.cluster_groups_dir}. Please try again.")
            sys.exit()

        # get PETH data for chosen clusters in designated sessions
        sound_stim_data = {}
        if len(self.session_list) > 0:
            for one_session in self.session_list:
                if os.path.exists(one_session):
                    relevant_session_clusters = ClusterFinder(session=one_session,
                                                              cluster_groups_dir=self.cluster_groups_dir).get_desired_clusters(filter_by_area=self.relevant_areas,
                                                                                                                               filter_by_cluster_type=self.relevant_cluster_types)
                    session_name, peth = Spikes(input_file=one_session).get_peths(get_clusters=relevant_session_clusters,
                                                                                  bin_size_ms=self.bin_size_ms,
                                                                                  window_size=self.window_size,
                                                                                  smooth=self.smooth,
                                                                                  smooth_sd=self.smooth_sd)
                    sound_stim_data[session_name] = peth
                else:
                    print(f"Invalid location for file {one_session}. Please try again.")
                    sys.exit()
        else:
            print("No session provided.")
            sys.exit()

        # prepare the arrays for plotting and calculate all the statistics
        total_num_clusters = np.sum([len(sound_stim_data[session].keys()) for session in sound_stim_data.keys()])
        plot_array = np.zeros((total_num_clusters, 2 * int(round(self.window_size / (self.bin_size_ms / 1e3)))))
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
                bins_to_skip = mod_idx_time // self.bin_size_ms
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

        significance_dict = {}
        for cluster in statistics_dict.keys():
            session_id = statistics_dict[cluster]['session']
            file_animal = [animal for animal in ClusterFinder.probe_site_areas.keys() if animal in session_id][0]
            file_bank = [bank for bank in ['distal', 'intermediate'] if bank in session_id][0]
            file_date = session_id[session_id.find('20') - 4:session_id.find('20') + 2]
            if file_animal not in significance_dict.keys():
                significance_dict[file_animal] = {}
            for idx, row in profile_data.iterrows():
                if row[0] == f'{file_animal}_{file_date}_{file_bank}' and row[1] == statistics_dict[cluster]['cell_id']:
                    cl_profile = row[-1]
                    break
            if statistics_dict[cluster]['sound_modulation_index'] < 0 and statistics_dict[cluster]['p_value'] < self.critical_p_value:
                modulated_clusters['suppressed'][cluster] = statistics_dict[cluster]
                significance_dict[file_animal][statistics_dict[cluster]['cell_id']] = cl_profile
                if cl_profile == 'RS':
                    count_dict['sign_suppressed_rs'] += 1
                else:
                    count_dict['sign_suppressed_fs'] += 1
                """if statistics_dict[cluster]['sound_modulation_index'] < -.5 and cl_profile == 'FS':
                    print(statistics_dict[cluster]['session'], statistics_dict[cluster]['cell_id'], statistics_dict[cluster]['sound_modulation_index'], cl_profile)"""
            elif statistics_dict[cluster]['sound_modulation_index'] > 0 and statistics_dict[cluster]['p_value'] < self.critical_p_value:
                modulated_clusters['excited'][cluster] = statistics_dict[cluster]
                significance_dict[file_animal][statistics_dict[cluster]['cell_id']] = cl_profile
                if cl_profile == 'RS':
                    count_dict['sign_excited_rs'] += 1
                else:
                    count_dict['sign_excited_fs'] += 1
                """if statistics_dict[cluster]['sound_modulation_index'] > .5 and cl_profile == 'FS':
                    print(statistics_dict[cluster]['session'], statistics_dict[cluster]['cell_id'], statistics_dict[cluster]['sound_modulation_index'], cl_profile)"""
            elif statistics_dict[cluster]['p_value'] >= self.critical_p_value:
                if cl_profile == 'RS':
                    count_dict['ns_rs'] += 1
                else:
                    count_dict['ns_fs'] += 1

        print(count_dict)

        if False:
            with io.open(f'smi_significant_{self.relevant_areas[0]}.json', 'w', encoding='utf-8') as mi_file:
                mi_file.write(json.dumps(significance_dict, ensure_ascii=False, indent=4))

        # order clusters in each category separately
        cluster_order_suppressed = [item[0] for item in sorted(modulated_clusters['suppressed'].items(), key=lambda i: i[1]['sound_modulation_index'])]
        cluster_order_excited = [item[0] for item in sorted(modulated_clusters['excited'].items(), key=lambda i: i[1]['sound_modulation_index'], reverse=True)]

        # find most modulated cells
        if get_most_modulated:
            print(f"There are {total_cluster_number} clusters in this dataset, and these are the category counts: {count_dict}")
            for idx in range(20):
                print(f"Number {idx + 1} on the suppressed list: {statistics_dict[cluster_order_suppressed[idx]]['session']}, "
                      f"{statistics_dict[cluster_order_suppressed[idx]]['cell_id']}, SMI: {statistics_dict[cluster_order_suppressed[idx]]['sound_modulation_index']}")
                print(f"Number {idx + 1} on the excited list: {statistics_dict[cluster_order_excited[idx]]['session']}, "
                      f"{statistics_dict[cluster_order_excited[idx]]['cell_id']}, SMI: {statistics_dict[cluster_order_excited[idx]]['sound_modulation_index']}")

        # re-order cluster array by sound modulation index (from lowest to highest value and vice-versa for excited clusters)
        plot_array_ordered_suppressed = plot_array.take(indices=cluster_order_suppressed, axis=0)
        plot_array_ordered_excited = plot_array.take(indices=cluster_order_excited, axis=0)

        # plot
        if self.to_plot:
            # make group mean activity plot
            fig = plt.figure(figsize=(8, 6), dpi=300, tight_layout=True)
            ax1 = fig.add_subplot(121, label='1')
            ax1.imshow(plot_array_ordered_suppressed, aspect='auto', vmin=0, vmax=1, cmap='cividis')
            ax2 = fig.add_subplot(121, label='2', frame_on=False)
            ax2.plot(range(plot_array_ordered_suppressed.shape[1]), plot_array_ordered_suppressed.mean(axis=0), ls='-', lw=3, c='#00008B')
            ax2.set_xlim(0, 400)
            ax2.set_xticks([])
            ax2.set_yticks([])
            ax3 = fig.add_subplot(122, label='3')
            im = ax3.imshow(plot_array_ordered_excited, aspect='auto', vmin=0, vmax=1, cmap='cividis')
            ax4 = fig.add_subplot(122, label='4', frame_on=False)
            ax4.plot(range(plot_array_ordered_excited.shape[1]), plot_array_ordered_excited.mean(axis=0), ls='-', lw=3, c='#EEC900')
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
            inner_colors = ['#00008B', '#EEC900', '#DEDEDE'] * 2
            outer_colors = [self.profile_colors['RS'], self.profile_colors['FS']]
            pie_values = np.array([[count_dict['sign_suppressed_rs'], count_dict['sign_excited_rs'], count_dict['ns_rs']],
                                   [count_dict['sign_suppressed_fs'], count_dict['sign_excited_fs'], count_dict['ns_fs']]])

            fig2, ax5 = plt.subplots(nrows=1, ncols=1, figsize=(8, 6), dpi=300)
            ax5.pie(pie_values.sum(axis=1), radius=1, colors=outer_colors, shadow=False,
                    autopct='%1.1f%%', labels=labels, wedgeprops=dict(width=size, edgecolor='#FFFFFF'))
            ax5.pie(pie_values.flatten(), radius=1 - size, colors=inner_colors,
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
            ax6.hist(smi_neg, bins=bins, color='#00008B', alpha=.6)
            ax6.hist(smi_pos, bins=bins, color='#EEC900', alpha=.6)
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

    def luminance_modulation_summary(self, **kwargs):
        """
        Description
        ----------
        This method plots the luminance modulation effect for a group of cells (can be across
        different animals). PETHs were smoothed with a Gaussian of 1 bins width.
        ----------

        Parameters
        ----------
        **kwargs (dictionary)
        to_calculate (bool)
            Calculate luminance modulation across files; defaults to False.
        decode_what (str)
            The modulation of interest; defaults to 'luminance'.
        speed_threshold_low (int/float)
            Value above which variable should be considered; defaults to 0.
        speed_threshold_high (int/float)
            Value below which variable should not be considered; defaults to 5.
        speed_min_seq_duration (int/float)
            The minimum duration for chosen sequences; defaults to 2 (seconds).
        ----------

        Returns
        ----------
        mean_activity_plot (fig)
            A "snake" plot of the sound stimulation effect.
        pie_chart (fig)
            A pie chart breakdown of sound modulation effects for RS and FS cells.
        SMI_histogram (fig)
            A histogram of the SMIs.
        ----------
        """

        to_calculate = kwargs['to_calculate'] if 'to_calculate' in kwargs.keys() and type(kwargs['to_calculate']) == bool else False
        decode_what = kwargs['decode_what'] if 'decode_what' in kwargs.keys() and type(kwargs['decode_what']) == str else 'luminance'
        speed_threshold_high = kwargs['speed_threshold_high'] if 'speed_threshold_high' in kwargs.keys() and (type(kwargs['speed_threshold_high']) == int or type(kwargs['speed_threshold_high']) == float) else 5.
        speed_threshold_low = kwargs['speed_threshold_low'] if 'speed_threshold_low' in kwargs.keys() and (type(kwargs['speed_threshold_low']) == int or type(kwargs['speed_threshold_low']) == float) else 0.
        speed_min_seq_duration = kwargs['speed_min_seq_duration'] if 'speed_min_seq_duration' in kwargs.keys() \
                                                                     and (type(kwargs['speed_min_seq_duration']) == int or type(kwargs['speed_min_seq_duration']) == float) else 2.

        if to_calculate:
            # get discontinuous PETH data for chosen clusters in designated sessions
            luminance_modulation_data = {}
            for three_sessions in self.input_012_list:
                # get details of the three sessions
                file_animal = [name for name in ClusterFinder.probe_site_areas.keys() if name in three_sessions[0]][0]
                file_bank = [bank for bank in ['distal', 'intermediate'] if bank in three_sessions[0]][0]
                get_date_idx = [date.start() for date in re.finditer('20', three_sessions[0])][-1]
                file_date = three_sessions[0][get_date_idx - 4:get_date_idx + 2]

                # get relevant clusters
                all_clusters, chosen_clusters, extra_chosen_clusters, cluster_dict = decode_events.choose_012_clusters(the_input_012=three_sessions,
                                                                                                                       cl_gr_dir=self.cluster_groups_dir,
                                                                                                                       sp_prof_csv=self.sp_profiles_csv,
                                                                                                                       cl_areas=self.relevant_areas,
                                                                                                                       cl_type=self.relevant_cluster_types,
                                                                                                                       dec_type=decode_what)

                # get discontinuous PETHs
                discontinuous_peths = Spikes(input_012=three_sessions,
                                             cluster_groups_dir=self.cluster_groups_dir,
                                             sp_profiles_csv=self.sp_profiles_csv).get_discontinuous_peths(get_clusters=all_clusters,
                                                                                                           cluster_type=self.relevant_cluster_types,
                                                                                                           cluster_areas=self.relevant_areas,
                                                                                                           discontinuous_raster=False,
                                                                                                           to_smooth=self.smooth,
                                                                                                           smooth_sd=self.smooth_sd,
                                                                                                           speed_threshold_high=speed_threshold_high,
                                                                                                           speed_threshold_low=speed_threshold_low,
                                                                                                           speed_min_seq_duration=speed_min_seq_duration,
                                                                                                           bin_size_ms=self.bin_size_ms,
                                                                                                           window_size=self.window_size)

                luminance_modulation_data[f'{file_animal}_{file_date}_{file_bank}'] = discontinuous_peths

            # prepare the arrays for plotting and calculate all the statistics
            total_cluster_num = np.sum([len(luminance_modulation_data[session].keys()) for session in luminance_modulation_data.keys()])
            statistics_dict = {'plot_array': np.zeros((total_cluster_num, int(round(self.window_size / (self.bin_size_ms / 1e3)))))}
            cell_id = 0
            for session in luminance_modulation_data.keys():
                for cluster in luminance_modulation_data[session].keys():
                    # # get mean PETH for plotting
                    all_trials = luminance_modulation_data[session][cluster]['discontinuous_peth']
                    averaged_trials = np.nanmean(all_trials, axis=0)

                    # # get all the details for the statistics dict
                    statistics_dict[cell_id] = {}

                    # get session and cluster id
                    statistics_dict[cell_id]['session'] = session
                    statistics_dict[cell_id]['cell_id'] = cluster

                    # normalize each average by its peak
                    statistics_dict['plot_array'][cell_id, :] = averaged_trials / np.max(averaged_trials)

                    statistics_dict[cell_id]['luminance_modulation_index'] = (averaged_trials[40:80].mean() - averaged_trials[:40].mean()) / \
                                                                             (averaged_trials[:40].mean() + averaged_trials[40:80].mean())

                    trials_array = np.zeros((all_trials.shape[0], 2))
                    for trial in range(all_trials.shape[0]):
                        trials_array[trial, :] = [all_trials[trial, :40].mean()-all_trials[trial, 40:80].mean(),  all_trials[trial, :40].mean()-all_trials[trial, 80:].mean()]
                    statistics_dict[cell_id]['p_value'] = wilcoxon(x=trials_array[:, 0], y=trials_array[:, 1], zero_method='zsplit')[1]

                    cell_id += 1

            # save statistics dict as .pkl file
            with open(f'{self.save_dir}{os.sep}luminance_modulation_{self.relevant_areas[0]}_data.pkl', 'wb') as pickle_file:
                pickle.dump(statistics_dict, pickle_file)

        if self.to_plot:

            # load pickle with luminance info
            with open(f'{self.pkl_load_dir}{os.sep}luminance_modulation_{self.relevant_areas[0]}_data.pkl', 'rb') as pickle_file:
                statistics_dict = pickle.load(pickle_file)

            # get total number of clusters in the dataset
            total_cluster_number = len(statistics_dict.keys())-1

            # get plot array and delete it from the dict
            plot_arr = statistics_dict['plot_array']
            del statistics_dict['plot_array']

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

            significance_dict = {'kavorka': {'distal': {}, 'intermediate': {}}, 'johnjohn': {'distal': {}, 'intermediate': {}}, 'frank': {'distal': {}, 'intermediate': {}}}
            for cluster in tqdm(statistics_dict.keys()):
                session_id = statistics_dict[cluster]['session']
                file_animal = [animal for animal in ClusterFinder.probe_site_areas.keys() if animal in session_id][0]
                file_bank = [bank for bank in ['distal', 'intermediate'] if bank in session_id][0]
                get_date_idx = [date.start() for date in re.finditer('20', session_id)][-1]
                file_date = session_id[get_date_idx-4:get_date_idx+2]
                for idx, row in profile_data.iterrows():
                    if row[0] == f'{file_animal}_{file_date}_{file_bank}' and row[1] == statistics_dict[cluster]['cell_id']:
                        cl_profile = row[-1]
                        break
                if statistics_dict[cluster][f'{decode_what}_modulation_index'] < 0 and statistics_dict[cluster]['p_value'] < self.critical_p_value:
                    modulated_clusters['suppressed'][cluster] = statistics_dict[cluster]
                    significance_dict[file_animal][file_bank][statistics_dict[cluster]['cell_id']] = cl_profile
                    if cl_profile == 'RS':
                        count_dict['sign_suppressed_rs'] += 1
                    else:
                        count_dict['sign_suppressed_fs'] += 1
                elif statistics_dict[cluster][f'{decode_what}_modulation_index'] > 0 and statistics_dict[cluster]['p_value'] < self.critical_p_value:
                    modulated_clusters['excited'][cluster] = statistics_dict[cluster]
                    significance_dict[file_animal][file_bank][statistics_dict[cluster]['cell_id']] = cl_profile
                    if cl_profile == 'RS':
                        count_dict['sign_excited_rs'] += 1
                    else:
                        count_dict['sign_excited_fs'] += 1
                elif statistics_dict[cluster]['p_value'] >= self.critical_p_value:
                    if cl_profile == 'RS':
                        count_dict['ns_rs'] += 1
                    else:
                        count_dict['ns_fs'] += 1

            if True:
                with io.open(f'lmi_significant_{self.relevant_areas[0]}.json', 'w', encoding='utf-8') as mi_file:
                    mi_file.write(json.dumps(significance_dict, ensure_ascii=False, indent=4))

            print(count_dict)

            # order clusters in each category separately
            cluster_order_suppressed = [item[0] for item in sorted(modulated_clusters['suppressed'].items(), key=lambda i: i[1][f'{decode_what}_modulation_index'])]
            cluster_order_excited = [item[0] for item in sorted(modulated_clusters['excited'].items(), key=lambda i: i[1][f'{decode_what}_modulation_index'], reverse=True)]

            # re-order cluster array by sound modulation index (from lowest to highest value and vice-versa for excited clusters)
            plot_array_ordered_suppressed = plot_arr.take(indices=cluster_order_suppressed, axis=0)
            plot_array_ordered_excited = plot_arr.take(indices=cluster_order_excited, axis=0)

            # make group mean activity plot
            fig = plt.figure(figsize=(8, 6), dpi=300, tight_layout=True)
            ax1 = fig.add_subplot(121, label='1')
            ax1.imshow(plot_array_ordered_suppressed, aspect='auto', vmin=0, vmax=1, cmap='cividis')
            ax2 = fig.add_subplot(121, label='2', frame_on=False)
            ax2.plot(range(plot_array_ordered_suppressed.shape[1]), plot_array_ordered_suppressed.mean(axis=0), ls='-', lw=3, c='#00008B')
            ax2.set_xlim(0, 120)
            ax2.set_xticks([])
            ax2.set_yticks([])
            ax3 = fig.add_subplot(122, label='3')
            im = ax3.imshow(plot_array_ordered_excited, aspect='auto', vmin=0, vmax=1, cmap='cividis')
            ax4 = fig.add_subplot(122, label='4', frame_on=False)
            ax4.plot(range(plot_array_ordered_excited.shape[1]), plot_array_ordered_excited.mean(axis=0), ls='-', lw=3, c='#EEC900')
            ax4.set_xlim(0, 120)
            ax4.set_xticks([])
            ax4.set_yticks([])
            cb_ax = fig.add_axes([0.9, 0.05, 0.01, 0.3])
            cbar = fig.colorbar(im, orientation='vertical', cax=cb_ax, shrink=.3)
            cbar.set_label('Normalized activity')
            cbar.ax.tick_params(size=0)
            ax1.set_xticks(np.arange(0, 121, 20))
            ax3.set_xticks(np.arange(0, 121, 20))
            ax1.set_xticklabels(np.arange(0, 7, 1))
            ax3.set_xticklabels(np.arange(0, 7, 1))
            ax1.set_xlabel('light-dark-light (s)')
            ax3.set_xlabel('light-dark-light (s)')
            ax1.tick_params(axis='y', length=0)
            ax3.tick_params(axis='y', length=0)
            ax1.set_ylabel('Cell number')
            for side in ['right', 'top', 'left', 'bottom']:
                ax1.spines[side].set_visible(False)
                ax3.spines[side].set_visible(False)
            if self.save_fig:
                if os.path.exists(self.save_dir):
                    fig.savefig(f'{self.save_dir}{os.sep}{decode_what}_peth_group.{self.fig_format}')
                else:
                    print("Specified save directory doesn't exist. Try again.")
                    sys.exit()
            plt.show()

            # make pie chart
            size = .3
            labels = ['RS', 'FS']
            inner_colors = ['#00008B', '#EEC900', '#DEDEDE'] * 2
            outer_colors = [self.profile_colors['RS'], self.profile_colors['FS']]
            pie_values = np.array([[count_dict['sign_suppressed_rs'], count_dict['sign_excited_rs'], count_dict['ns_rs']],
                                   [count_dict['sign_suppressed_fs'], count_dict['sign_excited_fs'], count_dict['ns_fs']]])

            fig2, ax5 = plt.subplots(nrows=1, ncols=1, figsize=(8, 6), dpi=300)
            ax5.pie(pie_values.sum(axis=1), radius=1, colors=outer_colors, shadow=False,
                    autopct='%1.1f%%', labels=labels, wedgeprops=dict(width=size, edgecolor='#FFFFFF'))
            ax5.pie(pie_values.flatten(), radius=1 - size, colors=inner_colors,
                    shadow=False, wedgeprops=dict(width=size, edgecolor='#FFFFFF'))
            ax5.set(aspect="equal", title='Luminance modulated cells summary`')
            if self.save_fig:
                if os.path.exists(self.save_dir):
                    fig2.savefig(f'{self.save_dir}{os.sep}{decode_what}_modulation_summary.{self.fig_format}')
                else:
                    print("Specified save directory doesn't exist. Try again.")
                    sys.exit()
            plt.show()

            # make SMI histograms
            smi = [statistics_dict[cluster]['luminance_modulation_index'] for cluster in statistics_dict.keys()]
            smi_neg = [statistics_dict[cluster]['luminance_modulation_index'] for cluster in statistics_dict.keys()
                       if (statistics_dict[cluster]['luminance_modulation_index'] < 0 and statistics_dict[cluster]['p_value'] < self.critical_p_value)]
            smi_pos = [statistics_dict[cluster]['luminance_modulation_index'] for cluster in statistics_dict.keys()
                       if (statistics_dict[cluster]['luminance_modulation_index'] > 0 and statistics_dict[cluster]['p_value'] < self.critical_p_value)]
            fig3 = plt.figure(figsize=(8, 6), dpi=300)
            bins = np.linspace(-1, 1, 20)
            ax6 = fig3.add_subplot(111, label='6')
            ax6.hist(smi, bins=bins, color='#DEDEDE', alpha=.6, edgecolor='#000000')
            ax6.hist(smi_neg, bins=bins, color='#00008B', alpha=.6)
            ax6.hist(smi_pos, bins=bins, color='#EEC900', alpha=.6)
            ax6.set_xlabel('Luminance modulation index')
            ax6.set_ylabel('Number of cells')
            for side in ['right', 'top']:
                ax6.spines[side].set_visible(False)
            if self.save_fig:
                if os.path.exists(self.save_dir):
                    fig3.savefig(f'{self.save_dir}{os.sep}{decode_what}_modulation_distribution.{self.fig_format}')
                else:
                    print("Specified save directory doesn't exist. Try again.")
                    sys.exit()
            plt.show()

    def decoding_summary(self, **kwargs):
        """
        Description
        ----------
        This method plots the event (sound, luminance, etc.) decoding accuracy separately
        for each animal. The lines representing animals represent the means of decoding
        accuracy across the 10 obtained runs for each number of clusters on the x-axis
        (which is, by default, [5, 10, 20, 50, 100]). The vertical lines represent 3*SEM
        at each of these points. The grey shaded area represent the results for 99% of the
        shuffled data.
        ----------

        Parameters
        ----------
        **kwargs (dictionary)
        x_values_arr (np.ndarray)
            An array of numbers of cells to decode with; defaults to np.array([5, 10, 20, 50, 100]).
        decoding_event (str)
            Decoding event for figure title; defaults to 'sound stimulation'.
        z_value_sem (float)
            The z-value for the SEM calculation; defaults to 2.58 (3 SD).
        ----------

        Returns
        ----------
        decoding_accuracy (fig)
            A plot of decoding accuracy across A and V cortices for a particular event.
        ----------
        """

        x_values_arr = kwargs['x_values_arr'] if 'x_values_arr' in kwargs.keys() and type(kwargs['x_values_arr']) == np.ndarray else np.array([5, 10, 20, 50, 100])
        decoding_event = kwargs['decoding_event'] if 'decoding_event' in kwargs.keys() and type(kwargs['decoding_event']) == str else 'sound stimulation'
        z_value_sem = kwargs['z_value_sem'] if 'z_value_sem' in kwargs.keys() and type(kwargs['z_value_sem']) == float else 2.58

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
        plot_data = {'A': {'decoding_accuracy': {'mean': {}, 'sem': {}}, 'shuffled': np.array([[1000., 0.]] * 5)},
                     'V': {'decoding_accuracy': {'mean': {}, 'sem': {}}, 'shuffled': np.array([[1000., 0.]] * 5)}}
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
        x_values = x_values_arr
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 5), dpi=300, tight_layout=True)
        ax[0].errorbar(x=x_values, y=plot_data['A']['decoding_accuracy']['mean']['kavorka'], yerr=plot_data['A']['decoding_accuracy']['sem']['kavorka'] * z_value_sem,
                       color='#000000', fmt='-o', label=f"#{self.animal_ids['kavorka']}")
        ax[0].errorbar(x=x_values, y=plot_data['A']['decoding_accuracy']['mean']['frank'], yerr=plot_data['A']['decoding_accuracy']['sem']['frank'] * z_value_sem,
                       color='#000000', fmt='-^', label=f"#{self.animal_ids['frank']}")
        ax[0].errorbar(x=x_values, y=plot_data['A']['decoding_accuracy']['mean']['johnjohn'], yerr=plot_data['A']['decoding_accuracy']['sem']['johnjohn'] * z_value_sem,
                       color='#000000', fmt='-s', label=f"#{self.animal_ids['johnjohn']}")
        ax[0].fill_between(x=x_values, y1=plot_data['A']['shuffled'][:, 0], y2=plot_data['A']['shuffled'][:, 1], color='grey', alpha=.25)
        ax[0].set_ylim(.3, 1)
        ax[0].set_xlim(0)
        ax[0].legend()
        ax[0].set_title('A1 units')
        ax[0].set_xlabel('Number of units')
        ax[0].set_ylabel('Decoding accuracy')

        ax[1].errorbar(x=x_values, y=plot_data['V']['decoding_accuracy']['mean']['kavorka'], yerr=plot_data['V']['decoding_accuracy']['sem']['kavorka'] * z_value_sem,
                       color='#000000', fmt='-o', label=f"#{self.animal_ids['kavorka']}")
        ax[1].errorbar(x=x_values, y=plot_data['V']['decoding_accuracy']['mean']['frank'], yerr=plot_data['V']['decoding_accuracy']['sem']['frank'] * z_value_sem,
                       color='#000000', fmt='-^', label=f"#{self.animal_ids['frank']}")
        ax[1].errorbar(x=x_values, y=plot_data['V']['decoding_accuracy']['mean']['johnjohn'], yerr=plot_data['V']['decoding_accuracy']['sem']['johnjohn'] * z_value_sem,
                       color='#000000', fmt='-s', label=f"#{self.animal_ids['johnjohn']}")
        ax[1].fill_between(x=x_values, y1=plot_data['V']['shuffled'][:, 0], y2=plot_data['V']['shuffled'][:, 1], color='#808080', alpha=.25)
        ax[1].set_ylim(.3, 1)
        ax[1].set_xlim(0)
        ax[1].legend()
        ax[1].set_title('V units')
        ax[1].set_xlabel('Number of units')
        ax[1].set_ylabel('Decoding accuracy')
        if self.save_fig:
            if os.path.exists(self.save_dir):
                fig.savefig(f'{self.save_dir}{os.sep}{decoding_event}_decoding_accuracy.{self.fig_format}')
            else:
                print("Specified save directory doesn't exist. Try again.")
                sys.exit()
        plt.show()

    def modulation_along_probe(self, **kwargs):
        """
        Description
        ----------
        This method plots sound and luminance modulation significant units with respect
        to their position along the probe. It sums all the significantly modulated units
        (suppressed or excited) at their respective peak channels and normalizes their
        counts by the maximum number of units at any channel.
        ----------

        Parameters
        ----------
        **kwargs (dictionary)
        cmap_smi (str)
            The colormap for SMI; defaults to 'Blues'.
        cmap_lmi (str)
            The colormap for LMI; defaults to 'Reds'.
        ----------

        Returns
        ----------
        modulation_along_probe (fig)
            A plot of SMI and LMI significant unit concentration along probe.
        ----------
        """

        cmap_smi = kwargs['cmap_smi'] if 'cmap_smi' in kwargs.keys() and type(kwargs['cmap_smi']) == str else 'Blues'
        cmap_lmi = kwargs['cmap_lmi'] if 'cmap_lmi' in kwargs.keys() and type(kwargs['cmap_lmi']) == str else 'Reds'

        data = {}
        for file in os.listdir(self.modulation_indices_dir):
            with open(f'{self.modulation_indices_dir}{os.sep}{file}') as json_file:
                temp_data = json.load(json_file)
            index_type = 'smi' if 'smi' in file else 'lmi'
            brain_area = 'V' if 'V' in file else 'A'
            data[f'{index_type}_{brain_area}'] = temp_data

        for animal in ['frank', 'johnjohn', 'kavorka']:
            plot_modulation_data = {'smi': list(data['smi_A'][animal].keys()) + list(data['smi_V'][animal].keys()),
                                    'lmi_distal': list(data['lmi_A'][animal].keys()) + list(data['lmi_V'][animal]['distal'].keys()),
                                    'lmi_intermediate': list(data['lmi_V'][animal]['intermediate'].keys())}

            plot_modulation_arrays = {'smi_probe_arr': np.zeros((384, 2)),
                                      'lmi_probe_arr': np.zeros((384, 2))}

            for data_type in plot_modulation_data.keys():
                index_type = 'smi' if 'smi' in data_type else 'lmi'
                bank = 'intermediate' if 'intermediate' in data_type else 'distal'

                for item in plot_modulation_data[data_type]:
                    if bank == 'distal':
                        ch = int(item[item.index('ch')+2:])
                    else:
                        ch = int(item[item.index('ch')+2:]) + 384
                    modulo = ch % 2
                    row = ch // 2
                    if modulo == 0:
                        col = 0
                    else:
                        col = 1
                    plot_modulation_arrays[f'{index_type}_probe_arr'][row, col] += 1

            reduction_factor = 2

            reduced_plot_modulation_arrays = {'smi_probe_arr': np.zeros((384 // reduction_factor, 1)),
                                              'lmi_probe_arr': np.zeros((384 // reduction_factor, 1))}

            for arr_name in plot_modulation_arrays:
                for rr_idx, reduced_row in enumerate(range(0, 384, reduction_factor)):
                    reduced_plot_modulation_arrays[arr_name][rr_idx, :] = plot_modulation_arrays[arr_name][reduced_row:reduced_row+reduction_factor, :].sum()

            for arr_name in reduced_plot_modulation_arrays:
                smoothed_arr = gaussian_smoothing(array=reduced_plot_modulation_arrays[arr_name],
                                                  sigma=3,
                                                  axis=0)
                reduced_plot_modulation_arrays[arr_name] = smoothed_arr / smoothed_arr.max()

            fig = plt.figure(figsize=(2, 8))
            ax = fig.add_subplot(121)
            im = ax.imshow(reduced_plot_modulation_arrays['smi_probe_arr'], aspect='auto', vmin=0, vmax=1, cmap=cmap_smi, alpha=1, origin='lower')
            ax2 = fig.add_subplot(122)
            im2 = ax2.imshow(reduced_plot_modulation_arrays['lmi_probe_arr'], aspect='auto', vmin=0, vmax=1, cmap=cmap_lmi, alpha=1, origin='lower')
            """cbar = fig.colorbar(im, orientation='vertical', shrink=.3)
            cbar.ax.tick_params(size=0)"""
            cbar2 = fig.colorbar(im2, orientation='vertical', shrink=.3)
            cbar2.ax.tick_params(size=0)
            if self.save_fig:
                if os.path.exists(self.save_dir):
                    fig.savefig(f'{self.save_dir}{os.sep}{animal}_modulation_along_probe.{self.fig_format}', dpi=300)
                else:
                    print("Specified save directory doesn't exist. Try again.")
                    sys.exit()
            plt.show()

    def light_dark_fr_correlations(self, **kwargs):
        """
        Description
        ----------
        This method plots the firing rate distribution changes across three different
        sessions and the correlation distribution of population vectors from session 3
        to the population averages of session 1 and session 2.
        ----------

        Parameters
        ----------
        **kwargs (dictionary)
        get_cl_profiles (bool)
            Get profiles (RS / FS) of clusters; defaults to False.
        total_fr_correlations (int)
            Total number of frames to correlate with; defaults to 1e4.
        ----------

        Returns
        ----------
        spike_count_distributions (fig)
            A plot of spike count distributions for the specified 3 files.
        ----------
        """

        get_cl_profiles = kwargs['get_cl_profiles'] if 'get_cl_profiles' in kwargs.keys() and type(kwargs['get_cl_profiles']) == bool else False
        total_fr_correlations = kwargs['total_fr_correlations'] if 'total_fr_correlations' in kwargs.keys() and type(kwargs['total_fr_correlations']) == int else 10000

        clusters_across_sessions = {}
        all_common_clusters = {}
        for animal in self.all_animals_012.keys():
            clusters_across_sessions[animal] = {0: [], 1: [], 2: []}
            for session_id, session in enumerate(self.all_animals_012[animal]):
                clusters_across_sessions[animal][session_id] = ClusterFinder(session=session,
                                                                             cluster_groups_dir=self.cluster_groups_dir,
                                                                             sp_profiles_csv=self.sp_profiles_csv).get_desired_clusters(filter_by_cluster_type=self.relevant_cluster_types,
                                                                                                                                        filter_by_area=self.relevant_areas)

            all_common_clusters[animal] = list(set(clusters_across_sessions[animal][0]).intersection(clusters_across_sessions[animal][1], clusters_across_sessions[animal][2]))

        print(len(all_common_clusters['kavorka']), len(all_common_clusters['frank']), len(all_common_clusters['johnjohn']))

        activity_across_sessions = {}
        for animal in self.all_animals_012.keys():
            activity_across_sessions[animal] = {0: {}, 1: {}, 2: {}}
            for session_id, session in enumerate(self.all_animals_012[animal]):
                the_session, activity_dictionary, purged_spikes_dict = Spikes(input_file=session).convert_activity_to_frames_with_shuffles(get_clusters=all_common_clusters[animal],
                                                                                                                                           to_shuffle=False,
                                                                                                                                           condense_arr=True)
                activity_across_sessions[animal][session_id] = activity_dictionary

        if get_cl_profiles:
            cluster_profiles = {}
            for animal in self.all_animals_012.keys():
                file_bank = [bank for bank in ['distal', 'intermediate'] if bank in self.all_animals_012[animal][0]][0]
                get_date_idx = [date.start() for date in re.finditer('20', self.all_animals_012[animal][0])][-1]
                file_date = self.all_animals_012[animal][0][get_date_idx-4:get_date_idx+2]
                cluster_profiles[animal] = get_cluster_spiking_profiles(cluster_list=all_common_clusters[animal], recording_day=f'{animal}_{file_date}_{file_bank}', sp_profiles_csv=self.sp_profiles_csv)

        activity_arrays = {}
        for animal in self.all_animals_012.keys():
            zero_ses_name, zero_extracted_frame_info = Session(session=self.all_animals_012[animal][0]).data_loader(extract_variables=['total_frame_num'])
            first_ses_name, first_extracted_frame_info = Session(session=self.all_animals_012[animal][1]).data_loader(extract_variables=['total_frame_num'])
            second_ses_name, second_extracted_frame_info = Session(session=self.all_animals_012[animal][2]).data_loader(extract_variables=['total_frame_num'])
            min_total_frame_num = np.array([zero_extracted_frame_info['total_frame_num'],
                                            first_extracted_frame_info['total_frame_num'],
                                            second_extracted_frame_info['total_frame_num']]).min() // int(120. * (100 / 1e3))

            # make spike count distributions figure
            activity_arrays[animal] = {0: np.zeros((min_total_frame_num, len(all_common_clusters[animal]))),
                                       1: np.zeros((min_total_frame_num, len(all_common_clusters[animal]))),
                                       2: np.zeros((min_total_frame_num, len(all_common_clusters[animal])))}
            row_num = np.ceil(np.sqrt(len(all_common_clusters[animal]))).astype(np.int32)
            col_num = np.ceil(np.sqrt(len(all_common_clusters[animal]))).astype(np.int32)
            fig, ax = plt.subplots(nrows=row_num, ncols=col_num, figsize=(15, 15))
            bins = np.arange(0, 10, 1)
            bin_centers = np.array([0.5 * (bins[i] + bins[i + 1]) for i in range(len(bins) - 1)])
            for cl_idx, cl in enumerate(all_common_clusters[animal]):
                if True:
                    if get_cl_profiles:
                        if cluster_profiles[cl] == 'RS':
                            profile_color = '#698B69'
                        else:
                            profile_color = '#9BCD9B'
                    activity_0 = activity_across_sessions[animal][0][cl]['activity'][:min_total_frame_num].todense().astype(np.float32)
                    activity_arrays[animal][0][:, cl_idx] = activity_0
                    activity_1 = activity_across_sessions[animal][1][cl]['activity'][:min_total_frame_num].todense().astype(np.float32)
                    activity_arrays[animal][1][:, cl_idx] = activity_1
                    activity_2 = activity_across_sessions[animal][2][cl]['activity'][:min_total_frame_num].todense().astype(np.float32)
                    activity_arrays[animal][2][:, cl_idx] = activity_2
                    data_entries_1, bins_1 = np.histogram(activity_0, bins=bins)
                    data_entries_2, bins_2 = np.histogram(activity_2, bins=bins)
                    data_entries_d, bins_d = np.histogram(activity_1, bins=bins)
                    ax = plt.subplot(row_num, col_num, cl_idx+1)
                    ax.plot(bin_centers, data_entries_d, color='#00008B', linewidth=1.5, alpha=.75)
                    ax.plot(bin_centers, data_entries_1, color='#EEC900', linewidth=1.5, alpha=.75)
                    ax.plot(bin_centers, data_entries_2, color='#CD950C', linewidth=1.5, alpha=.75)
                    ax.set_xticks([])
                    ax.set_yticks([])
                    if get_cl_profiles:
                        if cluster_profiles[cl] == 'FS':
                            for side in ['bottom', 'top', 'right', 'left']:
                                ax.spines[side].set_linewidth(4)
                                ax.spines[side].set_color(profile_color)
                    ax.set_title(cl[6:12], fontdict={'fontweight': 'bold', 'fontsize': 8})
            plt.tight_layout()
            if self.save_fig:
                if os.path.exists(self.save_dir):
                    fig.savefig(f'{self.save_dir}{os.sep}spike_count_distributions_{animal}_{self.relevant_areas[0]}.{self.fig_format}', dpi=300)
                else:
                    print("Specified save directory doesn't exist. Try again.")
                    sys.exit()
            plt.show()

        # make population vector correlation plot
        for animal_idx, animal in enumerate(self.all_animals_012.keys()):
            correlations_0 = np.zeros(total_fr_correlations)
            correlations_1 = np.zeros(total_fr_correlations)
            for frame in tqdm(range(total_fr_correlations)):
                correlations_0[frame] = pearsonr(activity_arrays[animal][2][frame, :], activity_arrays[animal][0].mean(axis=0))[0]
                correlations_1[frame] = pearsonr(activity_arrays[animal][2][frame, :], activity_arrays[animal][1].mean(axis=0))[0]

            bins2 = np.linspace(-0.1, 1, 100)
            fig2 = plt.figure(figsize=(5, 5))
            ax2 = fig2.add_subplot(111)
            ax2.hist(correlations_1, bins2, density=True, alpha=0.5, label='Dark', color='#00008B')
            ax2.hist(correlations_0, bins2, density=True, alpha=0.5, label='Light 1', color='#EEC900')
            ax2.legend(loc='upper left')
            ax2.set_xlabel('Correlation')
            ax2.set_ylabel('Probability density')
            ax2.set_title(animal)
            if self.save_fig:
                if os.path.exists(self.save_dir):
                    fig2.savefig(f'{self.save_dir}{os.sep}population_vector_correlations_{animal}_{self.relevant_areas[0]}.{self.fig_format}', dpi=300)
                else:
                    print("Specified save directory doesn't exist. Try again.")
                    sys.exit()
            plt.show()

