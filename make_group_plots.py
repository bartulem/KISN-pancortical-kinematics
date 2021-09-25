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
import seaborn as sns
from mpl_toolkits.mplot3d.axes3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d.proj3d import proj_transform
import pandas as pd
from tqdm import tqdm
from random import gauss
from scipy.stats import wilcoxon
from scipy.stats import sem
from scipy.stats import pearsonr
from scipy.stats import mannwhitneyu
import decode_events
import sessions2load
import make_ratemaps
import neural_activity
import select_clusters
import define_spiking_profile

class Arrow3D(FancyArrowPatch):

    def __init__(self, x, y, z, dx, dy, dz, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._xyz = (x, y, z)
        self._dxdydz = (dx, dy, dz)

    def draw(self, renderer):
        x1, y1, z1 = self._xyz
        dx, dy, dz = self._dxdydz
        x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)

        xs, ys, zs = proj_transform((x1, x2), (y1, y2), (z1, z2), self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        super().draw(renderer)

def _arrow3D(ax, x, y, z, dx, dy, dz, *args, **kwargs):
    arrow = Arrow3D(x, y, z, dx, dy, dz, *args, **kwargs)
    ax.add_artist(arrow)

setattr(Axes3D, 'arrow3D', _arrow3D)


class PlotGroupResults:

    mi_colors = {'excited': '#EEC900', 'suppressed': '#00008B', 'ns': '#DEDEDE'}

    tuning_categories = {0: '#232323',  # unclassified
                         1: '#C8D92B',  # position
                         2: '#777831',  # self-motion, speeds
                         3: '#CC2128',  # ego head position
                         4: '#E28586',  # ego head movement
                         5: '#6C2265',  # allo head position
                         6: '#B799C8',  # allo head movement
                         7: '#1B6AA0',  # back position
                         8: '#569CB2',  # back movement
                         9: '#F47927',  # neck_elevation
                         10: '#FAAD73'}  # neck movement

    feature_colors = {'Unclassified': '#232323',
                      'Ego3_Head_roll_1st_der': '#F38D9F',
                      'Ego3_Head_azimuth_1st_der': '#F27170',
                      'Ego3_Head_pitch_1st_der': '#EE4E6F',
                      'Ego3_Head_roll': '#ED2A2C',
                      'Ego3_Head_azimuth': '#CA2026',
                      'Ego3_Head_pitch': '#AC2023',
                      'Ego2_head_roll_1st_der': '#BE88BB',
                      'Allo_head_direction_1st_der': '#8D64AA',
                      'Ego2_head_pitch_1st_der': '#C897C4',
                      'Ego2_head_roll': '#8D3A95',
                      'Allo_head_direction': '#8A267E',
                      'Ego2_head_pitch': '#6F3894',
                      'Back_azimuth_1st_der': '#86D5F5',
                      'Back_pitch_1st_der': '#5C8ECA',
                      'Back_azimuth': '#2977B6',
                      'Back_pitch': '#15489D',
                      'Neck_elevation': '#F37827',
                      'Neck_elevation_1st_der': '#F9AD74',
                      'Position': '#C8D92B',
                      'Body_direction': '#64BC62',
                      'Body_direction_1st_der': '#91C38F',
                      'Speeds': '#14A049',
                      'Self_motion': '#665E27',
                       np.nan: '#000000'}

    def __init__(self, session_list=[], cluster_groups_dir='', sp_profiles_csv='',
                 save_fig=False, fig_format='png', save_dir='',
                 decoding_dir='', animal_ids=None,
                 relevant_areas=None, relevant_cluster_types='good',
                 bin_size_ms=50, window_size=10, smooth=False, smooth_sd=1, to_plot=False,
                 input_012_list=[], pkl_load_dir='', critical_p_value=.01,
                 profile_colors=None, modulation_indices_dir='',
                 all_animals_012={}, tuning_peaks_file='', occ_file='', cch_summary_file='',
                 md_distances_file=''):
        if relevant_areas is None:
            relevant_areas = ['A']
        if animal_ids is None:
            animal_ids = {'frank': '26473', 'johnjohn': '26471', 'kavorka': '26525',
                          'roy': '26472', 'bruno': '26148', 'jacopo': '26504', 'crazyjoe': '26507'}
        if profile_colors is None:
            profile_colors = {'RS': '#698B69', 'FS': '#9BCD9B'}
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
        self.tuning_peaks_file = tuning_peaks_file
        self.occ_file = occ_file
        self.cch_summary_file = cch_summary_file
        self.md_distances_file = md_distances_file

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
                    relevant_session_clusters = select_clusters.ClusterFinder(session=one_session,
                                                                              cluster_groups_dir=self.cluster_groups_dir).get_desired_clusters(filter_by_area=self.relevant_areas,
                                                                                                                                               filter_by_cluster_type=self.relevant_cluster_types)
                    session_name, peth = neural_activity.Spikes(input_file=one_session).get_peths(get_clusters=relevant_session_clusters,
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
            file_animal = [animal for animal in select_clusters.ClusterFinder.probe_site_areas.keys() if animal in session_id][0]
            file_bank = [bank for bank in ['distal', 'intermediate'] if bank in session_id][0]
            file_date = session_id[session_id.find('20') - 4:session_id.find('20') + 2]
            if file_animal not in significance_dict.keys():
                significance_dict[file_animal] = {}
            for idx, row in profile_data.iterrows():
                if row[0] == f'{file_animal}_{file_date}_{file_bank}' and row[1] == statistics_dict[cluster]['cell_id']:
                    cl_profile = row[7]
                    cl_row = idx
                    break

            # save to profile data .csv
            profile_data.iloc[cl_row, 8] = statistics_dict[cluster]['sound_modulation_index']
            profile_data.iloc[cl_row, 9] = statistics_dict[cluster]['p_value']

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

        # save SMI-filled dataframe to .csv file
        profile_data.to_csv(path_or_buf=f'{self.sp_profiles_csv}', sep=';', index=False)

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
        LMI_histogram (fig)
            A histogram of the LMIs.
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
            for three_sessions in tqdm(self.input_012_list):
                # get details of the three sessions
                file_animal = [name for name in select_clusters.ClusterFinder.probe_site_areas.keys() if name in three_sessions[0]][0]
                file_bank = [bank for bank in ['distal', 'intermediate'] if bank in three_sessions[0]][0]
                get_date_idx = [date.start() for date in re.finditer('20', three_sessions[0])][-1]
                file_date = three_sessions[0][get_date_idx - 4:get_date_idx + 2]

                # get relevant clusters
                all_clusters, chosen_clusters, extra_chosen_clusters, cluster_dict = decode_events.choose_012_clusters(the_input_012=three_sessions,
                                                                                                                       cl_gr_dir=self.cluster_groups_dir,
                                                                                                                       sp_prof_csv=self.sp_profiles_csv,
                                                                                                                       cl_areas=self.relevant_areas,
                                                                                                                       cl_type=self.relevant_cluster_types,
                                                                                                                       dec_type=decode_what,
                                                                                                                       desired_profiles=True)

                # get discontinuous PETHs
                discontinuous_peths = neural_activity.Spikes(input_012=three_sessions,
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
                    pseudo_trials_array = np.zeros((all_trials.shape[0], 2))
                    for trial in range(all_trials.shape[0]):
                        trials_array[trial, :] = [all_trials[trial, :40].mean(), all_trials[trial, 40:80].mean()]
                        pseudo_trials_array[trial, :] = [all_trials[trial, :40].mean(), all_trials[trial, 80:].mean()]
                    statistics_dict[cell_id]['p_value'] = wilcoxon(x=trials_array[:, 0], y=trials_array[:, 1], zero_method='zsplit')[1]
                    statistics_dict[cell_id]['p_value_check'] = wilcoxon(x=pseudo_trials_array[:, 0], y=pseudo_trials_array[:, 1], zero_method='zsplit')[1]

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

            significance_dict = {'crazyjoe': {'distal': {}, 'intermediate': {}}, 'jacopo': {'distal': {}, 'intermediate': {}}, 'roy': {'distal': {}, 'intermediate': {}}}
            for cluster in tqdm(statistics_dict.keys()):
                session_id = statistics_dict[cluster]['session']
                file_animal = [animal for animal in select_clusters.ClusterFinder.probe_site_areas.keys() if animal in session_id][0]
                file_bank = [bank for bank in ['distal', 'intermediate'] if bank in session_id][0]
                get_date_idx = [date.start() for date in re.finditer('20', session_id)][-1]
                file_date = session_id[get_date_idx-4:get_date_idx+2]
                for idx, row in profile_data.iterrows():
                    if row[0] == f'{file_animal}_{file_date}_{file_bank}' and row[1] == statistics_dict[cluster]['cell_id']:
                        cl_profile = row[7]
                        cl_row = idx
                        break

                # save to profile data .csv
                profile_data.iloc[cl_row, 10] = statistics_dict[cluster]['luminance_modulation_index']
                profile_data.iloc[cl_row, 11] = statistics_dict[cluster]['p_value']
                profile_data.iloc[cl_row, 12] = statistics_dict[cluster]['p_value_check']

                if statistics_dict[cluster][f'{decode_what}_modulation_index'] < 0 and statistics_dict[cluster]['p_value'] < self.critical_p_value < statistics_dict[cluster]['p_value_check']:
                    modulated_clusters['suppressed'][cluster] = statistics_dict[cluster]
                    significance_dict[file_animal][file_bank][statistics_dict[cluster]['cell_id']] = cl_profile
                    if cl_profile == 'RS':
                        count_dict['sign_suppressed_rs'] += 1
                    else:
                        count_dict['sign_suppressed_fs'] += 1
                elif statistics_dict[cluster][f'{decode_what}_modulation_index'] > 0 and statistics_dict[cluster]['p_value'] < self.critical_p_value < statistics_dict[cluster]['p_value_check']:
                    modulated_clusters['excited'][cluster] = statistics_dict[cluster]
                    significance_dict[file_animal][file_bank][statistics_dict[cluster]['cell_id']] = cl_profile
                    if cl_profile == 'RS':
                        count_dict['sign_excited_rs'] += 1
                    else:
                        count_dict['sign_excited_fs'] += 1
                elif statistics_dict[cluster]['p_value'] >= self.critical_p_value or \
                        (statistics_dict[cluster]['p_value'] < self.critical_p_value and statistics_dict[cluster]['p_value_check'] < self.critical_p_value):
                    if cl_profile == 'RS':
                        count_dict['ns_rs'] += 1
                    else:
                        count_dict['ns_fs'] += 1

            # save LMI-filled dataframe to .csv file
            profile_data.to_csv(path_or_buf=f'{self.sp_profiles_csv}', sep=';', index=False)

            if True:
                with io.open(f'/home/bartulm/Downloads/lmi_significant_{self.relevant_areas[0]}.json', 'w', encoding='utf-8') as mi_file:
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
        areas (list)
            The brain areas decoding was performed on; defaults to ['A', 'V'].
        animals_1 (list)
            Animals for the first brain area; defaults to ['kavorka', 'frank', 'johnjohn']
        animals_2 (list)
            Animals for the first brain area; defaults to ['kavorka', 'frank', 'johnjohn']
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
        areas = kwargs['areas'] if 'areas' in kwargs.keys() and type(kwargs['areas']) == list else ['A', 'V']
        animals_1 = kwargs['animals_1'] if 'animals_1' in kwargs.keys() and type(kwargs['animals_1']) == list else ['kavorka', 'frank', 'johnjohn']
        animals_2 = kwargs['animals_2'] if 'animals_2' in kwargs.keys() and type(kwargs['animals_2']) == list else ['kavorka', 'frank', 'johnjohn']

        file_dict = {'data': {areas[0]: [], areas[1]: []}, 'shuffled': {areas[0]: [], areas[1]: []}}
        if not os.path.exists(self.decoding_dir):
            print(f"Invalid location for directory {self.decoding_dir}. Please try again.")
            sys.exit()
        else:
            for decoding_file_name in os.listdir(self.decoding_dir):
                if 'shuffled' in decoding_file_name:
                    if areas[0] in decoding_file_name:
                        file_dict['shuffled'][areas[0]].append(decoding_file_name)
                    else:
                        file_dict['shuffled'][areas[1]].append(decoding_file_name)
                else:
                    if areas[0] in decoding_file_name:
                        file_dict['data'][areas[0]].append(decoding_file_name)
                    else:
                        file_dict['data'][areas[1]].append(decoding_file_name)

        # sort dict by file name
        for data_type in file_dict.keys():
            for data_area in file_dict[data_type].keys():
                file_dict[data_type][data_area].sort()

        # load the data
        decoding_data = {'data': {areas[0]: {}, areas[1]: {}}, 'shuffled': {areas[0]: {}, areas[1]: {}}}
        for data_type in decoding_data.keys():
            for data_area in decoding_data[data_type].keys():
                for one_file in file_dict[data_type][data_area]:
                    animal_name = [animal for animal in self.animal_ids.keys() if animal in one_file][0]
                    decoding_data[data_type][data_area][animal_name] = np.load(f'{self.decoding_dir}{os.sep}{one_file}')

        # get data to plot
        plot_data = {areas[0]: {'decoding_accuracy': {'mean': {}, 'sem': {}}, 'shuffled': np.array([[1000., 0.]] * 5)},
                     areas[1]: {'decoding_accuracy': {'mean': {}, 'sem': {}}, 'shuffled': np.array([[1000., 0.]] * 5)}}
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
        ax[0].errorbar(x=x_values, y=plot_data[areas[0]]['decoding_accuracy']['mean'][animals_1[0]], yerr=plot_data[areas[0]]['decoding_accuracy']['sem'][animals_1[0]] * z_value_sem,
                       color='#000000', fmt='-o', label=f"#{self.animal_ids[animals_1[0]]}")
        ax[0].errorbar(x=x_values, y=plot_data[areas[0]]['decoding_accuracy']['mean'][animals_1[1]], yerr=plot_data[areas[0]]['decoding_accuracy']['sem'][animals_1[1]] * z_value_sem,
                       color='#000000', fmt='-^', label=f"#{self.animal_ids[animals_1[1]]}")
        ax[0].errorbar(x=x_values, y=plot_data[areas[0]]['decoding_accuracy']['mean'][animals_1[2]], yerr=plot_data[areas[0]]['decoding_accuracy']['sem'][animals_1[2]] * z_value_sem,
                       color='#000000', fmt='-s', label=f"#{self.animal_ids[animals_1[2]]}")
        ax[0].fill_between(x=x_values, y1=plot_data[areas[0]]['shuffled'][:, 0], y2=plot_data[areas[0]]['shuffled'][:, 1], color='grey', alpha=.25)
        ax[0].set_ylim(.3, 1)
        ax[0].set_xlim(0)
        ax[0].legend()
        ax[0].set_title(f'{areas[0]} units')
        ax[0].set_xlabel('Number of units')
        ax[0].set_ylabel('Decoding accuracy')

        ax[1].errorbar(x=x_values, y=plot_data[areas[1]]['decoding_accuracy']['mean'][animals_2[0]], yerr=plot_data[areas[1]]['decoding_accuracy']['sem'][animals_2[0]] * z_value_sem,
                       color='#000000', fmt='-o', label=f"#{self.animal_ids[animals_2[0]]}")
        ax[1].errorbar(x=x_values, y=plot_data[areas[1]]['decoding_accuracy']['mean'][animals_2[1]], yerr=plot_data[areas[1]]['decoding_accuracy']['sem'][animals_2[1]] * z_value_sem,
                       color='#000000', fmt='-^', label=f"#{self.animal_ids[animals_2[1]]}")
        ax[1].errorbar(x=x_values, y=plot_data[areas[1]]['decoding_accuracy']['mean'][animals_2[2]], yerr=plot_data[areas[1]]['decoding_accuracy']['sem'][animals_2[2]] * z_value_sem,
                       color='#000000', fmt='-s', label=f"#{self.animal_ids[animals_2[2]]}")
        ax[1].fill_between(x=x_values, y1=plot_data[areas[1]]['shuffled'][:, 0], y2=plot_data[areas[1]]['shuffled'][:, 1], color='#808080', alpha=.25)
        ax[1].set_ylim(.3, 1)
        ax[1].set_xlim(0)
        ax[1].legend()
        ax[1].set_title(f'{areas[1]} units')
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
                                    'lmi_distal': list(data['lmi_A'][animal]['distal'].keys()) + list(data['lmi_V'][animal]['distal'].keys()),
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
                smoothed_arr = neural_activity.gaussian_smoothing(array=reduced_plot_modulation_arrays[arr_name],
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
                clusters_across_sessions[animal][session_id] = select_clusters.ClusterFinder(session=session,
                                                                                             cluster_groups_dir=self.cluster_groups_dir,
                                                                                             sp_profiles_csv=self.sp_profiles_csv).get_desired_clusters(filter_by_cluster_type=self.relevant_cluster_types,
                                                                                                                                                        filter_by_area=self.relevant_areas)

            all_common_clusters[animal] = list(set(clusters_across_sessions[animal][0]).intersection(clusters_across_sessions[animal][1], clusters_across_sessions[animal][2]))

        print(len(all_common_clusters['kavorka']), len(all_common_clusters['frank']), len(all_common_clusters['johnjohn']))

        activity_across_sessions = {}
        for animal in self.all_animals_012.keys():
            activity_across_sessions[animal] = {0: {}, 1: {}, 2: {}}
            for session_id, session in enumerate(self.all_animals_012[animal]):
                the_session, activity_dictionary, purged_spikes_dict = neural_activity.Spikes(input_file=session).convert_activity_to_frames_with_shuffles(get_clusters=all_common_clusters[animal],
                                                                                                                                                           to_shuffle=False,
                                                                                                                                                           condense_arr=True)
                activity_across_sessions[animal][session_id] = activity_dictionary

        if get_cl_profiles:
            cluster_profiles = {}
            for animal in self.all_animals_012.keys():
                file_bank = [bank for bank in ['distal', 'intermediate'] if bank in self.all_animals_012[animal][0]][0]
                get_date_idx = [date.start() for date in re.finditer('20', self.all_animals_012[animal][0])][-1]
                file_date = self.all_animals_012[animal][0][get_date_idx-4:get_date_idx+2]
                cluster_profiles[animal] = define_spiking_profile.get_cluster_spiking_profiles(cluster_list=all_common_clusters[animal],
                                                                                               recording_day=f'{animal}_{file_date}_{file_bank}',
                                                                                               sp_profiles_csv=self.sp_profiles_csv)

        activity_arrays = {}
        for animal in self.all_animals_012.keys():
            zero_ses_name, zero_extracted_frame_info = sessions2load.Session(session=self.all_animals_012[animal][0]).data_loader(extract_variables=['total_frame_num'])
            first_ses_name, first_extracted_frame_info = sessions2load.Session(session=self.all_animals_012[animal][1]).data_loader(extract_variables=['total_frame_num'])
            second_ses_name, second_extracted_frame_info = sessions2load.Session(session=self.all_animals_012[animal][2]).data_loader(extract_variables=['total_frame_num'])
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

    def tuning_peaks_by_occ(self, **kwargs):
        """
        Description
        ----------
        This method plots tuning peaks by occupancy for a given brain area.
        ----------

        Parameters
        ----------
        **kwargs (dictionary)
        chosen_variables (list)
            Variables of interest; defaults to ['Ego3_Head_pitch', 'Ego3_Head_roll', 'Ego3_Head_azimuth',
                                                'Back_pitch', 'Back_azimuth', 'Neck_elevation'].
        ----------

        Returns
        ----------
        tuning_peaks_occ (fig)
            A plot of the relationship between tuning peaks and occupancy.
        ----------
        """

        chosen_variables = kwargs['chosen_variables'] if 'chosen_variables' in kwargs.keys() \
                                                         and type(kwargs['chosen_variables']) == list else ['Ego3_Head_pitch', 'Ego3_Head_roll', 'Ego3_Head_azimuth',
                                                                                                            'Back_pitch', 'Back_azimuth', 'Neck_elevation']

        # load occupancy file and get data
        if os.path.exists(self.occ_file):
            with open(self.occ_file) as occ_file:
                occ_json = json.load(occ_file)
        else:
            print("The occupancy file doesn't exist. Try again.")
            sys.exit()

        occ_data = {}
        x_data = {}
        for var in chosen_variables:
            occ_data[var] = np.array(occ_json[var]['occ']) / np.sum(occ_json[var]['occ']) * 100
            x_data[var] = np.array(occ_json[var]['xvals'])

        # load tuning peaks file and get data
        if os.path.exists(self.tuning_peaks_file):
            with open(self.tuning_peaks_file) as tp_file:
                tp_json = json.load(tp_file)
        else:
            print("The tuning peaks file doesn't exist. Try again.")
            sys.exit()

        tp_data_raw = {var: [] for var in chosen_variables}
        for cl in tp_json.keys():
            for var in chosen_variables:
                if var in tp_json[cl]['features'].keys():
                    tp_data_raw[var].append(tp_json[cl]['features'][var])

        counted_tp_data = {var: {x_val: tp_data_raw[var].count(x_val) for x_val in x_data[var]} for var in chosen_variables}

        tp_data = {}
        for var in counted_tp_data.keys():
            one_var = np.array(list(counted_tp_data[var].values()))
            tp_data[var] = one_var / one_var.sum() * 100

        # plot results
        fig, ax = plt.subplots(nrows=1, ncols=len(chosen_variables),
                               figsize=(6.4*len(chosen_variables), 4.8))
        for var_idx, var in enumerate(chosen_variables):
            feature_color = [val for key, val in make_ratemaps.Ratemap.feature_colors.items() if key in var][0]
            ax = plt.subplot(1, len(chosen_variables), var_idx+1)
            ax.set_title(var)
            bool_arr = occ_data[var] > 1.
            occ = occ_data[var][bool_arr]
            tp = tp_data[var][bool_arr]
            ax.scatter(x=occ, y=tp, color=feature_color, alpha=1, s=30)
            ax.axvline(x=int(np.ceil(occ.max()))/2, ls='-.', color='#000000')
            ax.set_xlabel('occupancy (% total)')
            ax.set_xlim(0, int(np.ceil(occ.max())))
            ax.set_xticks([0, int(np.ceil(occ.max()))])
            ax.set_ylabel('cells tuned (%)')
            ax.set_ylim(0, int(np.ceil(tp.max())))
            ax.set_yticks([0, int(np.ceil(tp.max()))])
        if self.save_fig:
            if os.path.exists(self.save_dir):
                fig.savefig(f'{self.save_dir}{os.sep}tuning_peaks_by_occ_{self.relevant_areas[0]}.{self.fig_format}', dpi=300)
            else:
                print("Specified save directory doesn't exist. Try again.")
                sys.exit()
        plt.show()

    def plot_cch_summary(self, **kwargs):
        """
        Description
        ----------
        This method plots the CCH summary results for every brain area.
        ----------

        Parameters
        ----------
        **kwargs (dictionary)

        ----------

        Returns
        ----------
        cch_summary (fig)
            A plot summarizing the CCH results across brain areas.
        ----------
        """

        # load the data
        with open(self.cch_summary_file, 'r') as summary_file:
            plotting_dict = json.load(summary_file)

        area_rats = {'VV': ['kavorka', 'johnjohn', 'frank'],
                     'AA': ['kavorka', 'johnjohn', 'frank'],
                     'MM': ['roy', 'crazyjoe', 'jacopo'],
                     'SS': ['roy', 'crazyjoe', 'jacopo']}

        # point_3d_dict = {}
        # for area_area in plotting_dict.keys():
        #     if area_area not in point_3d_dict.keys():
        #         point_3d_dict[area_area] = {}
        #     for animal in area_rats[area_area]:
        #         if animal not in point_3d_dict[area_area].keys():
        #             point_3d_dict[area_area][animal] = {'X': [], 'Y': [], 'Z': []}
        #         for a_s in plotting_dict[area_area][animal].keys():
        #             for cl in plotting_dict[area_area][animal][a_s]['clusters'].keys():
        #                 point_3d_dict[area_area][animal]['X'].append(plotting_dict[area_area][animal][a_s]['clusters'][cl]['XYZ'][0])
        #                 point_3d_dict[area_area][animal]['Y'].append(plotting_dict[area_area][animal][a_s]['clusters'][cl]['XYZ'][1])
        #                 point_3d_dict[area_area][animal]['Z'].append(plotting_dict[area_area][animal][a_s]['clusters'][cl]['XYZ'][2])
        #
        # line_3d_dict = {}
        # for area_area in plotting_dict.keys():
        #     if area_area not in line_3d_dict.keys():
        #         line_3d_dict[area_area] = {}
        #     for animal in area_rats[area_area]:
        #         if animal not in line_3d_dict[area_area].keys():
        #             line_3d_dict[area_area][animal] = {}
        #         for a_s in plotting_dict[area_area][animal].keys():
        #             for pair_idx, pair in enumerate(plotting_dict[area_area][animal][a_s]['pairs']):
        #                 cl1, cl2 = pair.split('-')
        #                 direction = plotting_dict[area_area][animal][a_s]['directionality'][pair_idx]
        #                 if direction == 1:
        #                     line_3d_dict[area_area][animal][f'pair_{pair_idx}'] = [plotting_dict[area_area][animal][a_s]['clusters'][cl1]['XYZ'],
        #                                                                            plotting_dict[area_area][animal][a_s]['clusters'][cl2]['XYZ'],
        #                                                                            plotting_dict[area_area][animal][a_s]['strength'][pair_idx]]
        #                 else:
        #                     line_3d_dict[area_area][animal][f'pair_{pair_idx}'] = [plotting_dict[area_area][animal][a_s]['clusters'][cl2]['XYZ'],
        #                                                                            plotting_dict[area_area][animal][a_s]['clusters'][cl1]['XYZ'],
        #                                                                            plotting_dict[area_area][animal][a_s]['strength'][pair_idx]]
        # fig = plt.figure(figsize=(4, 12), dpi=300)
        # gs = fig.add_gridspec(9, 3)
        # gs.update(wspace=1.5, hspace=1.5)
        # ax1 = fig.add_subplot(gs[:3, :3], projection='3d')
        # for pair in line_3d_dict['VV']['frank'].keys():
        #     pair_data = line_3d_dict['VV']['frank'][pair]
        #     ax1.arrow3D(pair_data[0][0], pair_data[0][1], pair_data[0][2],
        #                 pair_data[1][0]-pair_data[0][0], pair_data[1][1]-pair_data[0][1], pair_data[1][2]-pair_data[0][2],
        #                 mutation_scale=5,
        #                 arrowstyle='-|>',
        #                 ls='-',
        #                 lw=.5+pair_data[2],
        #                 color='#000000')
        # ax1.scatter(point_3d_dict['VV']['frank']['X'], point_3d_dict['VV']['frank']['Y'], point_3d_dict['VV']['frank']['Z'], color='#FFFFFF', ec='#000000', alpha=1)
        # ax1.view_init(elev=20, azim=120)
        # ax1.set_title(f"#{self.animal_ids['frank']}", pad=0)
        # ax1.set_xlabel('AP (mm)')
        # ax1.set_ylabel('ML (mm)')
        # ax1.set_zlabel('DV (mm)')
        # ax2 = fig.add_subplot(gs[3:6, :3], projection='3d')
        # for pair in line_3d_dict['VV']['kavorka'].keys():
        #     pair_data = line_3d_dict['VV']['kavorka'][pair]
        #     ax2.arrow3D(pair_data[0][0], pair_data[0][1], pair_data[0][2],
        #                 pair_data[1][0]-pair_data[0][0], pair_data[1][1]-pair_data[0][1], pair_data[1][2]-pair_data[0][2],
        #                 mutation_scale=5,
        #                 arrowstyle='-|>',
        #                 ls='-',
        #                 lw=.5+pair_data[2],
        #                 color='#000000')
        # ax2.plot3D([-5.85, -5.86],
        #            [6.0, 5.75],
        #            [-2, -2], ls='-', color='#000000', lw=.5+.05)
        # ax2.text(-5.86, 5.75, -2.05, s='5%',  size=8, color='#000000')
        # ax2.plot3D([-5.85, -5.86],
        #            [6.0, 5.75],
        #            [-2.2, -2.2], ls='-', color='#000000', lw=.5+.5)
        # ax2.text(-5.86, 5.75, -2.25, s='50%',  size=8, color='#000000')
        # ax2.scatter(point_3d_dict['VV']['kavorka']['X'], point_3d_dict['VV']['kavorka']['Y'], point_3d_dict['VV']['kavorka']['Z'], color='#FFFFFF', ec='#000000', alpha=1)
        # ax2.view_init(elev=20, azim=120)
        # ax2.set_title(f"#{self.animal_ids['kavorka']}", pad=0)
        # ax2.set_xlabel('AP (mm)')
        # ax2.set_ylabel('ML (mm)')
        # ax2.set_zlabel('DV (mm)')
        # ax3 = fig.add_subplot(gs[6:9, :3], projection='3d')
        # for pair in line_3d_dict['VV']['johnjohn'].keys():
        #     pair_data = line_3d_dict['VV']['johnjohn'][pair]
        #     ax3.arrow3D(pair_data[0][0], pair_data[0][1], pair_data[0][2],
        #                 pair_data[1][0]-pair_data[0][0], pair_data[1][1]-pair_data[0][1], pair_data[1][2]-pair_data[0][2],
        #                 mutation_scale=5,
        #                 arrowstyle='-|>',
        #                 ls='-',
        #                 lw=.5+pair_data[2],
        #                 color='#000000')
        # ax3.scatter(point_3d_dict['VV']['johnjohn']['X'], point_3d_dict['VV']['johnjohn']['Y'], point_3d_dict['VV']['johnjohn']['Z'], color='#FFFFFF', ec='#000000', alpha=1)
        # ax3.view_init(elev=20, azim=120)
        # ax3.set_title(f"#{self.animal_ids['johnjohn']}", pad=0)
        # ax3.set_xlabel('AP (mm)')
        # ax3.set_ylabel('ML (mm)')
        # ax3.set_zlabel('DV (mm)')
        # plt.show()

        # fig2, f_ax = plt.subplots(1, 1, figsize=(7, 6), dpi=400)
        # sns.regplot(np.log10(plotting_dict['VV']['distances']), np.log10(plotting_dict['VV']['strength']), color='#000000', scatter_kws={'alpha':.3})
        # f_ax.set_xlabel('log$_{10}$pair distance (mm)')
        # f_ax.set_xlim(-3, .5)
        # f_ax.set_xticks([-2.5, -2, -1.5, -1, -.5, 0])
        # f_ax.set_ylabel('log$_{10}$synapse strength (A.U.)')
        # f_ax.set_ylim(-2.75, -.25)
        # f_ax.text(x=-2.85, y=-.55, s='r=-.1 (p=.07)', fontsize=16)
        # h_bins = [1.4, 1.8, 2.2, 2.6, 3, 3.4, 3.8, 4.2]
        # inset_axes = fig2.add_axes(rect=[.65, .65, .22, .20])
        # inset_axes.hist(plotting_dict['VV']['timing'], bins=h_bins, color='#000000', alpha=.3)
        # inset_axes.plot(np.median(plotting_dict['VV']['timing']), 5, marker='o', ms=5, c='#000000')
        # inset_axes.set_xticks([1.6, 2, 2.4, 2.8, 3.2, 3.6, 4])
        # inset_axes.set_xticklabels([1.6, 2, 2.4, 2.8, 3.2, 3.6, 4], fontsize=6)
        # inset_axes.set_xlabel('CC offset (ms)')
        # inset_axes.set_yticks([0, 50, 100, 150])
        # inset_axes.set_yticklabels([0, 50, 100, 150], fontsize=6)
        # inset_axes.set_ylabel('Number of pairs')
        # plt.show()

        fig3 = plt.figure(figsize=(5, 6), dpi=400)
        gs = fig3.add_gridspec(3, 4)
        gs.update(hspace=.5)
        ff_ax1 = fig3.add_subplot(gs[0, 0])
        ff_ax1.bar(x=[0, 1], height=[plotting_dict['VV']['profile']['RS'], plotting_dict['VV']['profile']['FS']], width=.9, color=['#698B69', '#9BCD9B'])
        ff_ax1.set_xticks([0, 1])
        ff_ax1.set_xticklabels(['RS', 'FS'], fontsize=8)
        ff_ax1.set_xlabel('profile', fontsize=10)
        ff_ax1.set_yticks([0, 125, 250])
        ff_ax1.set_yticklabels([0, 125, 250], fontsize=8)
        ff_ax1.set_ylabel('cell #', fontsize=10)
        ff_ax2 = fig3.add_subplot(gs[1, 0])
        ff_ax2.bar(x=[0, 1, 2], height=[plotting_dict['VV']['SMI']['excited'], plotting_dict['VV']['SMI']['suppressed'], plotting_dict['VV']['SMI']['ns']], width=.9, color=['#EEC900', '#00008B', '#DEDEDE'])
        ff_ax2.set_xticks([0, 1, 2])
        ff_ax2.set_xticklabels(['exc', 'sup', 'ns'], fontsize=8)
        ff_ax2.set_xlabel('SM', fontsize=10)
        ff_ax2.set_yticks([0, 100, 200, 300])
        ff_ax2.set_yticklabels([0, 100, 200, 300], fontsize=8)
        ff_ax2.set_ylabel('cell #', fontsize=10)
        ff_ax3 = fig3.add_subplot(gs[2, 0])
        ff_ax3.bar(x=[0, 1, 2], height=[plotting_dict['VV']['LMI']['excited'], plotting_dict['VV']['LMI']['suppressed'], plotting_dict['VV']['LMI']['ns']], width=.9, color=['#EEC900', '#00008B', '#DEDEDE'])
        ff_ax3.set_xticks([0, 1, 2])
        ff_ax3.set_xticklabels(['exc', 'sup', 'ns'], fontsize=8)
        ff_ax3.set_xlabel('LM', fontsize=10)
        ff_ax3.set_yticks([0, 125, 250])
        ff_ax3.set_yticklabels([0, 125, 250], fontsize=8)
        ff_ax3.set_ylabel('cell #', fontsize=10)
        ff_ax4 = fig3.add_subplot(gs[:, 1:4])
        ff_ax4.bar(x=list(range(24)), height=[plotting_dict['VV']['behavior']['null'],
                                              plotting_dict['VV']['behavior']['Ego3_Head_roll_1st_der'],
                                              plotting_dict['VV']['behavior']['Ego3_Head_azimuth_1st_der'],
                                              plotting_dict['VV']['behavior']['Ego3_Head_pitch_1st_der'],
                                              plotting_dict['VV']['behavior']['Ego3_Head_roll'],
                                              plotting_dict['VV']['behavior']['Ego3_Head_azimuth'],
                                              plotting_dict['VV']['behavior']['Ego3_Head_pitch'],
                                              plotting_dict['VV']['behavior']['Ego2_head_roll_1st_der'],
                                              plotting_dict['VV']['behavior']['Allo_head_direction_1st_der'],
                                              plotting_dict['VV']['behavior']['Ego2_head_pitch_1st_der'],
                                              plotting_dict['VV']['behavior']['Ego2_head_roll'],
                                              plotting_dict['VV']['behavior']['Allo_head_direction'],
                                              plotting_dict['VV']['behavior']['Ego2_head_pitch'],
                                              plotting_dict['VV']['behavior']['Back_azimuth_1st_der'],
                                              plotting_dict['VV']['behavior']['Back_pitch_1st_der'],
                                              plotting_dict['VV']['behavior']['Back_azimuth'],
                                              plotting_dict['VV']['behavior']['Back_pitch'],
                                              plotting_dict['VV']['behavior']['Neck_elevation'],
                                              plotting_dict['VV']['behavior']['Neck_elevation_1st_der'],
                                              plotting_dict['VV']['behavior']['Position'],
                                              plotting_dict['VV']['behavior']['Body_direction'],
                                              plotting_dict['VV']['behavior']['Body_direction_1st_der'],
                                              plotting_dict['VV']['behavior']['Speeds'],
                                              plotting_dict['VV']['behavior']['Self_motion']], width=.9, color=[self.feature_colors[key] for key in self.feature_colors.keys()])
        ff_ax4.yaxis.tick_right()
        ff_ax4.yaxis.set_label_position('right')
        ff_ax4.set_ylabel('cell #', fontsize=10)
        ff_ax4.set_xlabel('Behavioral tuning', fontsize=10)
        plt.tight_layout()
        plt.show()

    def plot_cch_functional(self, **kwargs):
        """
        Description
        ----------
        This method plots the CCH summary results for every brain area.
        ----------

        Parameters
        ----------
        **kwargs (dictionary)
        umap_embedding_file (str)
            File location of UMAP embedding file.
        plot_raw_umap (bool)
            Plots raw UMAP results with nothing else; defaults to False.
        plot_connected_cl (bool)
            Plots connected cluster pairs in all areas; defaults to False.
        filter_unclassified (bool)
            Remove GLM 'unclassified' clusters from the plot; defaults to False.
        plot_sm (bool)
            Use sensory modulation colors to plot; defaults to False.
        sm (str)
            Sensory modulation of interest: 'sound' or 'luminance'; defaults to 'sound.
        df_pca_columns (list / bool)
            Columns of the spiking profile csv file to be used for dimensionality reduction;
            defaults to ['SMI', 'pSMI', 'LMI', 'pLMI', 'pLMIcheck',
                         'B Speeds', 'C Body_direction', 'C Body_direction_1st_der',
                         'D Allo_head_direction', 'D Allo_head_direction_1st_der',
                         'G Neck_elevation', 'G Neck_elevation_1st_der', 'K Ego3_Head_roll',
                         'K Ego3_Head_roll_1st_der', 'L Ego3_Head_pitch', 'L Ego3_Head_pitch_1st_der',
                         'M Ego3_Head_azimuth',	'M Ego3_Head_azimuth_1st_der', 'N Back_pitch',
                         'N Back_pitch_1st_der', 'O Back_azimuth', 'O Back_azimuth_1st_der',
                         'P Ego2_head_roll', 'P Ego2_head_roll_1st_der', 'Q Ego2_head_pitch',
                         'Q Ego2_head_pitch_1st_der', 'Z Position', 'Z Self_motion'].
        ----------

        Returns
        ----------
        cch_function (fig)
            A plot summarizing the CCH results across brain areas.
        ---
        """
        umap_embedding_file = kwargs['umap_embedding_file'] if 'umap_embedding_file' in kwargs.keys() and type(kwargs['umap_embedding_file']) == str else ''
        plot_raw_umap = kwargs['plot_raw_umap'] if 'plot_raw_umap' in kwargs.keys() and type(kwargs['plot_raw_umap']) == bool else False
        plot_connected_cl = kwargs['plot_connected_cl'] if 'plot_connected_cl' in kwargs.keys() and type(kwargs['plot_connected_cl']) == bool else False
        filter_unclassified = kwargs['filter_unclassified'] if 'filter_unclassified' in kwargs.keys() and type(kwargs['filter_unclassified']) == bool else False
        plot_sm = kwargs['plot_sm'] if 'plot_sm' in kwargs.keys() and type(kwargs['plot_sm']) == bool else False
        sm = kwargs['sm'] if 'sm' in kwargs.keys() and kwargs['sm'] in ['sound', 'luminance'] else 'sound'
        df_pca_columns = kwargs['df_pca_columns'] if 'df_pca_columns' in kwargs.keys() and \
                                                     type(kwargs['df_pca_columns']) == list else ['SMI', 'pSMI', 'LMI', 'pLMI', 'pLMIcheck',
                                                                                                  'B Speeds', 'C Body_direction', 'C Body_direction_1st_der',
                                                                                                  'D Allo_head_direction', 'D Allo_head_direction_1st_der',
                                                                                                  'G Neck_elevation', 'G Neck_elevation_1st_der', 'K Ego3_Head_roll',
                                                                                                  'K Ego3_Head_roll_1st_der', 'L Ego3_Head_pitch', 'L Ego3_Head_pitch_1st_der',
                                                                                                  'M Ego3_Head_azimuth', 'M Ego3_Head_azimuth_1st_der', 'N Back_pitch',
                                                                                                  'N Back_pitch_1st_der', 'O Back_azimuth', 'O Back_azimuth_1st_der',
                                                                                                  'P Ego2_head_roll', 'P Ego2_head_roll_1st_der', 'Q Ego2_head_pitch',
                                                                                                  'Q Ego2_head_pitch_1st_der', 'Z Position', 'Z Self_motion']

        # load the data
        umap_data = np.load(umap_embedding_file)
        spc = pd.read_csv(self.sp_profiles_csv)
        with open(self.cch_summary_file, 'r') as summary_file:
            synaptic_data = json.load(summary_file)
        with open(self.md_distances_file, 'r') as md_file:
            md_distances_data = json.load(md_file)

        # screen for first covariate nan values, so they can be excluded
        non_nan_idx_list = spc.loc[~pd.isnull(spc.loc[:, 'first_covariate'])].index.tolist()

        # get colors for functional features
        color_list = [self.tuning_categories[spc.loc[i, 'category']] for i in non_nan_idx_list]

        if plot_sm:
            color_list = []
            for i in range(spc.shape[0]):
                if i in non_nan_idx_list:
                    if sm == 'sound':
                        if spc.loc[i, 'pSMI'] < .05 and spc.loc[i, 'SMI'] < 0:
                            color_list.append(self.mi_colors['suppressed'])
                        elif spc.loc[i, 'pSMI'] < .05 and spc.loc[i, 'SMI'] > 0:
                            color_list.append(self.mi_colors['excited'])
                        else:
                            color_list.append(self.mi_colors['ns'])
                    elif sm == 'luminance':
                        if spc.loc[i, 'pLMI'] < .05 < spc.loc[i, 'pLMIcheck'] and spc.loc[i, 'LMI'] < 0:
                            color_list.append(self.mi_colors['suppressed'])
                        elif spc.loc[i, 'pLMI'] < .05 < spc.loc[i, 'pLMIcheck'] and spc.loc[i, 'LMI'] > 0:
                            color_list.append(self.mi_colors['excited'])
                        else:
                            color_list.append(self.mi_colors['ns'])

        if filter_unclassified:
            unclassified_idx_lst = []
            for c_idx, c in enumerate(color_list):
                if c == '#232323':
                    unclassified_idx_lst.append(c_idx)
            umap_data = np.delete(arr=umap_data, obj=unclassified_idx_lst, axis=0)
            try:
                while True:
                    color_list.remove('#232323')
            except ValueError:
                pass

        # plot raw UMAP results
        if plot_raw_umap:
            fig = plt.figure(dpi=400)
            ax = fig.add_subplot()
            ax.scatter(umap_data[:, 0], umap_data[:, 1], s=10, c=color_list, alpha=.5)
            # ax.set_title('Sound modulation')
            ax.set_xlabel('UMAP 1')
            ax.set_ylabel('UMAP 2')
            plt.show()

        pl_dict = {'VV': {'points': [], 'pairs': [], 'strength': [], 'type': []}, 'AA': {'points': [], 'pairs': [], 'strength': [], 'type': []},
                   'MM': {'points': [], 'pairs': [], 'strength': [], 'type': []}, 'SS': {'points': [], 'pairs': [], 'strength': [], 'type': []}}

        for area in synaptic_data.keys():
            if area == 'VV' or area == 'AA':
                animal_list = ['kavorka', 'johnjohn', 'frank']
            else:
                animal_list = ['jacopo', 'crazyjoe', 'roy']
            for animal in animal_list:
                for animal_session in synaptic_data[area][animal].keys():
                    for pair_idx, pair in enumerate(synaptic_data[area][animal][animal_session]['pairs']):
                        cl1, cl2 = pair.split('-')
                        spc_pos1 = spc[(spc['cluster_id'] == cl1) & (spc['session_id'] == animal_session)].index.tolist()[0]
                        spc_pos2 = spc[(spc['cluster_id'] == cl2) & (spc['session_id'] == animal_session)].index.tolist()[0]
                        if spc_pos1 in non_nan_idx_list and spc_pos2 in non_nan_idx_list:
                            pos1 = non_nan_idx_list.index(spc_pos1)
                            pos2 = non_nan_idx_list.index(spc_pos2)
                            if pos1 not in pl_dict[area]['points']:
                                pl_dict[area]['points'].append(pos1)
                            if pos2 not in pl_dict[area]['points']:
                                pl_dict[area]['points'].append(pos2)
                            if synaptic_data[area][animal][animal_session]['directionality'][pair_idx] < 0:
                                pl_dict[area]['pairs'].append((pos1, pos2))
                            else:
                                pl_dict[area]['pairs'].append((pos2, pos1))
                            pl_dict[area]['strength'].append(synaptic_data[area][animal][animal_session]['strength'][pair_idx])
                            if synaptic_data[area][animal][animal_session]['type'][pair_idx] == 'excitatory':
                                pl_dict[area]['type'].append('-')
                            elif synaptic_data[area][animal][animal_session]['type'][pair_idx] == 'inhibitory':
                                pl_dict[area]['type'].append('-.')

        # plot
        if plot_connected_cl:
            area_colors = {'VV': '#E79791', 'AA': '#5F847F', 'MM': '#EEB849', 'SS': '#7396C0'}
            variables_dict = {'eu_distances': {'VV': [], 'AA': [], 'MM': [], 'SS': []}, 'synapse_strength': {'VV': [], 'AA': [], 'MM': [], 'SS': []}}
            fig, ax = plt.subplots(2, 2, dpi=500)
            for sub_idx, subplot in enumerate(['VV', 'AA', 'MM', 'SS']):
                ax = plt.subplot(2, 2, sub_idx+1)
                ax.scatter(umap_data[pl_dict[subplot]['points'], 0], umap_data[pl_dict[subplot]['points'], 1], s=2, c=area_colors[subplot], alpha=.5)
                ax.set_title(f'{subplot} connections')
                ax.set_xlabel('UMAP 1', fontsize=10)
                ax.set_ylabel('UMAP 2', fontsize=10)
                if subplot == 'VV' or subplot == 'AA':
                    ax.set_xlim(-1, 12)
                    ax.set_xticks([2, 4, 6, 8, 10])
                    ax.set_ylim(-2, 11)
                    ax.set_yticks([0, 2, 4, 6, 8, 10])
                else:
                    if subplot == 'MM':
                        ax.set_xlim(-1, 9)
                        ax.set_xticks([0, 2, 4, 6, 8])
                    else:
                        ax.set_xlim(1, 9)
                        ax.set_xticks([2, 4, 6, 8])
                    ax.set_ylim(-2, 9)
                    ax.set_yticks([0, 2, 4, 6, 8])
                for con_idx, connection in enumerate(pl_dict[subplot]['pairs']):
                    variables_dict['eu_distances'][subplot] = md_distances_data[subplot]['md_distance']
                    variables_dict['synapse_strength'][subplot].append(pl_dict[subplot]['strength'][con_idx])
                    ax.plot([umap_data[connection[0], 0], umap_data[connection[1], 0]], [umap_data[connection[0], 1], umap_data[connection[1], 1]],
                            lw=pl_dict[subplot]['strength'][con_idx]*3, ls=pl_dict[subplot]['type'][con_idx], c=area_colors[subplot])
            plt.tight_layout()
            plt.show()

            for variable in variables_dict.keys():
                fig2, ax2 = plt.subplots(1, 1, dpi=500)
                xs = [[gauss(0.25*(ind+1), 0.015) for x in range(len(variables_dict[variable][area]))] for ind, area in enumerate(variables_dict[variable].keys())]
                for sub_idx, subplot in enumerate(['VV', 'AA', 'MM', 'SS']):
                    ax2.scatter(xs[sub_idx], variables_dict[variable][subplot], s=10, color=area_colors[subplot], alpha=.5)
                    parts = ax2.violinplot(dataset=variables_dict[variable][subplot], positions=[np.mean(xs[sub_idx])+.1], vert=True, widths=.1, showmeans=False, showmedians=False, showextrema=False)
                    for pc in parts['bodies']:
                        pc.set_facecolor(area_colors[subplot])
                        pc.set_edgecolor('#000000')
                        pc.set_alpha(.4)
                    quartile1, median, quartile3 = np.percentile(variables_dict[variable][subplot], [25, 50, 75])
                    ax2.scatter([np.mean(xs[sub_idx])+.1], median, marker='o', color='#FFFFFF', s=20, zorder=3)
                    ax2.vlines([np.mean(xs[sub_idx])+.1], quartile1, quartile3, color='#000000', linestyle='-', lw=2)
                if variable == 'synapse_strength':
                    ax2.set_ylabel('Synapse strength')
                else:
                    ax2.set_ylabel('Euclidean distance in "functional space"')
                ax2.set_yscale('log')
                ax2.set_xticks([])
                plt.show()

                print('VV', 'AA', mannwhitneyu(variables_dict[variable]['VV'], variables_dict[variable]['AA']))
                print('VV', 'MM', mannwhitneyu(variables_dict[variable]['VV'], variables_dict[variable]['MM']))
                print('VV', 'SS', mannwhitneyu(variables_dict[variable]['VV'], variables_dict[variable]['SS']))
                print('AA', 'SS', mannwhitneyu(variables_dict[variable]['AA'], variables_dict[variable]['SS']))
                print('AA', 'MM', mannwhitneyu(variables_dict[variable]['AA'], variables_dict[variable]['MM']))
                print('MM', 'SS', mannwhitneyu(variables_dict[variable]['MM'], variables_dict[variable]['SS']))

            fig3, ax3 = plt.subplots(2, 2, dpi=500)
            for sub_idx, subplot in enumerate(['VV', 'AA', 'MM', 'SS']):
                ax3 = plt.subplot(2, 2, sub_idx+1)
                sns.regplot(np.log10(variables_dict['eu_distances'][subplot]), np.log10(variables_dict['synapse_strength'][subplot]), color=area_colors[subplot], scatter_kws={'alpha':.5})
                print(subplot, pearsonr(np.log10(variables_dict['eu_distances'][subplot]), np.log10(variables_dict['synapse_strength'][subplot])),
                      pearsonr(variables_dict['eu_distances'][subplot], variables_dict['synapse_strength'][subplot]))
                ax3.set_title(subplot)
                ax3.set_xlabel('log$_{10}$functional distances (A.U.)')
                ax3.set_ylabel('log$_{10}$synapse strength (A.U.)')
                if subplot == 'VV' or subplot == 'AA':
                    ax3.set_xlim(-.6, 1.6)
                    ax3.set_xticks([-.5, 0, .5, 1, 1.5])
                    ax3.set_ylim(-2.75, -.15)
                    ax3.set_yticks([-2.5, -2, -1.5, -1, -.5])
                elif subplot == 'MM':
                    ax3.set_xlim(-.4, 1.6)
                    ax3.set_xticks([0, .5, 1, 1.5])
                    ax3.set_ylim(-2.6, -.4)
                    ax3.set_yticks([-2.5, -2, -1.5, -1, -.5])
                else:
                    ax3.set_xlim(-.6, 1.1)
                    ax3.set_xticks([-.5, 0, .5, 1])
                    ax3.set_ylim(-2.6, -.9)
                    ax3.set_yticks([-2.5, -2, -1.5, -1])
            plt.tight_layout()
            plt.show()




