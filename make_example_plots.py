# -*- coding: utf-8 -*-

"""

@author: bartulem

Make example plots.

"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colorbar as cbar
from scipy.stats import sem
import neural_activity
import select_clusters

plt.rcParams['font.sans-serif'] = ['Helvetica']


class PlotExamples:
    animal_ids = {'frank': '26473', 'johnjohn': '26471', 'kavorka': '26525',
                  'roy': '26472', 'bruno': '26148', 'jacopo': '26504', 'crazyjoe': '26507'}

    elev_azimuth_3d = {'frank': (40, 60), 'johnjohn': (40, 120), 'kavorka': (10, 120),
                       'roy': (10, 120), 'jacopo': (10, 120), 'crazyjoe': (40, 0)}

    area_rats = {'VV': ['kavorka', 'johnjohn', 'frank'],
                 'AA': ['kavorka', 'johnjohn', 'frank'],
                 'AV': ['kavorka', 'johnjohn', 'frank'],
                 'MM': ['roy', 'crazyjoe', 'jacopo'],
                 'SS': ['roy', 'crazyjoe', 'jacopo'],
                 'MS': ['roy', 'crazyjoe', 'jacopo']}

    areas_by_rat = {'kavorka': ['A', 'V'],
                    'johnjohn': ['A', 'V'],
                    'frank': ['A', 'V'],
                    'roy': ['M', 'S'],
                    'crazyjoe': ['M', 'S'],
                    'jacopo': ['M', 'S']}

    def __init__(self, session=0, cluster_name='', kilosort_output_dir='',
                 save_fig=False, fig_format='png', save_dir='',
                 input_012=['', '', ''], cl_brain_area='', cch_data_dir='',
                 area_file='', cell_pair_id='', sp_profiles_csv='', cluster_groups_dir='',
                 profile_colors=None, area_colors=None, cch_summary_file='',
                 data_file_dir=''):
        self.session = session
        self.input_012 = input_012
        self.cluster_name = cluster_name
        self.kilosort_output_dir = kilosort_output_dir
        self.save_fig = save_fig
        self.fig_format = fig_format
        self.save_dir = save_dir
        self.cl_brain_area = cl_brain_area
        self.cch_data_dir = cch_data_dir
        self.area_file = area_file
        self.cell_pair_id = cell_pair_id
        self.sp_profiles_csv = sp_profiles_csv
        self.cluster_groups_dir = cluster_groups_dir
        self.data_file_dir = data_file_dir
        if profile_colors is None:
            self.profile_colors = {'RS': '#698B69', 'FS': '#9BCD9B'}
        if area_colors is None:
            self.area_colors = {'V': '#E79791', 'A': '#5F847F', 'M': '#EEB849', 'S': '#7396C0'}
        self.cch_summary_file = cch_summary_file

    def plot_peth(self, **kwargs):
        """
        Description
        ----------
        This method creates a PETH plot for a given cluster based on the onsets of the white
        noise stimulation. It plots the specific raster for that cluster, a behavioral PETH
        for a chosen feature and the trial-averaged PETH for that cluster.
        ----------

        Parameters
        ----------
        **kwargs (dictionary)
        peth_color (str)
            Color of the PETH; defaults to '#000000'.
        raster_color (str)
            Color of the raster; defaults to '#808080'.
        beh_color (str)
            Color of the behavior; defaults to '#008000'.
        ----------

        Returns
        ----------
        peth (fig)
            A peri-event time histogram.
        ----------
        """

        peth_color = kwargs['peth_color'] if 'peth_color' in kwargs.keys() and type(kwargs['peth_color']) == str else '#000000'
        raster_color = kwargs['raster_color'] if 'raster_color' in kwargs.keys() and type(kwargs['raster_color']) == str else '#808080'
        beh_color = kwargs['beh_color'] if 'beh_color' in kwargs.keys() and type(kwargs['beh_color']) == str else '#008000'

        # get peth, raster and behavior data
        session_name, peth, raster, peth_beh = neural_activity.Spikes(input_file=self.session).get_peths(get_clusters=[self.cluster_name],
                                                                                                         smooth=True,
                                                                                                         smooth_sd=3,
                                                                                                         raster=True,
                                                                                                         beh_raster='speeds')

        # get means/SEMs for spikes/behavior
        peth_mean = np.nanmean(peth[self.cluster_name]['peth'], axis=0)
        peth_sem = sem(a=peth[self.cluster_name]['peth'], axis=0, nan_policy='omit')
        beh_mean = np.nanmean(peth_beh, axis=0)
        beh_sem = sem(a=peth_beh, axis=0, nan_policy='omit')

        # plot figure
        fig = plt.figure(figsize=(5, 5), dpi=300, tight_layout=True)
        ax1 = fig.add_subplot(111, label='1')
        ax2 = fig.add_subplot(111, label='2', frame_on=False)
        # ax3 = fig.add_subplot(111, label='3', frame_on=False)
        ax1.eventplot(raster[self.cluster_name], colors=raster_color, lineoffsets=1, linelengths=1, linewidths=.1)
        ax2.plot(range(peth[self.cluster_name]['peth'].shape[1]), peth_mean, color=peth_color)
        ax2.fill_between(range(peth[self.cluster_name]['peth'].shape[1]), peth_mean + peth_sem, peth_mean - peth_sem, color=peth_color, alpha=.3)
        ax2.axvspan(200, 300, alpha=0.2, color=raster_color)
        # ax3.plot(range(peth_beh.shape[1]), beh_mean, color=beh_color)
        # ax3.fill_between(range(peth_beh.shape[1]), beh_mean+beh_sem, beh_mean-beh_sem, color=beh_color, alpha=.3)
        ax1.set_xticks([])
        # ax1.set_yticks([])
        ax1.set_ylabel('Trial number')
        ax1.set_ylim(1)
        ax2.set_xticks(np.arange(0, 401, 50))
        ax2.set_xticklabels(np.arange(-10, 10.1, 2.5))
        ax2.set_xlabel('Time relative to sound onset (s)')
        ax2.yaxis.tick_right()
        ax2.yaxis.set_label_position('right')
        ax2.set_ylabel('Firing rate (spikes/s)')
        # ax3.set_xticks([])
        # ax3.yaxis.tick_left()
        # ax3.yaxis.set_label_position('left')
        # ax3.tick_params(axis='y', colors=beh_color)
        # ax3.set_ylabel('Speed (cm/s)', color=beh_color)
        if self.save_fig:
            if os.path.exists(self.save_dir):
                fig.savefig(f'{self.save_dir}{os.sep}{self.cluster_name}_wh_noise_peth.{self.fig_format}')
            else:
                print("The specified save directory doesn't exist. Try again.")
                sys.exit()
        plt.show()

    def plot_discontinuous_peth(self, **kwargs):
        """
        Description
        ----------
        This method creates a discontinuous PETH plot for a given cluster across
        three different recording sessions.
        ----------

        Parameters
        ----------
        **kwargs (dictionary)
        peth_color (str)
            Color of the PETH; defaults to '#000000'.
        raster_color (str)
            Color of the discontinuous_raster; defaults to '#808080'.
        ----------

        Returns
        ----------
        discontinuous_peth (fig)
            A discontinuous peri-event time histogram.
        ----------
        """

        peth_color = kwargs['peth_color'] if 'peth_color' in kwargs.keys() and type(kwargs['peth_color']) == str else '#000000'
        raster_color = kwargs['raster_color'] if 'raster_color' in kwargs.keys() and type(kwargs['raster_color']) == str else '#808080'

        # get discontinuous_peth and discontinuous_raster
        discontinuous_peth, discontinuous_raster = neural_activity.Spikes(input_012=self.input_012).get_discontinuous_peths(get_clusters=[self.cluster_name],
                                                                                                                            cluster_type='good',
                                                                                                                            cluster_areas=[self.cl_brain_area],
                                                                                                                            discontinuous_raster=True,
                                                                                                                            to_smooth=True,
                                                                                                                            smooth_sd=3)

        # get means/SEMs for spikes/behavior
        discontinuous_peth_mean = np.nanmean(discontinuous_peth[self.cluster_name]['discontinuous_peth'], axis=0)
        peth_sem = sem(a=discontinuous_peth[self.cluster_name]['discontinuous_peth'], axis=0, nan_policy='omit')

        # plot figure
        fig = plt.figure(figsize=(5, 5), dpi=300, tight_layout=True)
        ax1 = fig.add_subplot(111, label='1')
        ax2 = fig.add_subplot(111, label='2', frame_on=False)
        ax1.eventplot(discontinuous_raster[self.cluster_name], colors=raster_color, lineoffsets=1, linelengths=1, linewidths=.1)
        ax2.plot(range(discontinuous_peth[self.cluster_name]['discontinuous_peth'].shape[1]), discontinuous_peth_mean, color=peth_color)
        ax2.fill_between(range(discontinuous_peth[self.cluster_name]['discontinuous_peth'].shape[1]), discontinuous_peth_mean + peth_sem, discontinuous_peth_mean - peth_sem, color=peth_color,
                         alpha=.3)
        ax2.axvspan(40, 80, alpha=0.2, color=raster_color)
        ax1.set_xlim(0, 6)
        ax1.set_xticks([])
        ax1.set_ylabel('Trial number')
        # ax1.set_ylim(1)
        ax2.set_xticks([])
        ax2.set_xticks(np.arange(0, 121, 20))
        ax2.set_xticklabels(np.arange(0, 7, 1))
        ax2.set_xlabel('light-dark-light (s)')
        ax2.yaxis.tick_right()
        ax2.yaxis.set_label_position('right')
        ax2.set_ylabel('Firing rate (spikes/s)')
        if self.save_fig:
            if os.path.exists(self.save_dir):
                fig.savefig(f'{self.save_dir}{os.sep}{self.cluster_name}_discontinuous_peth.{self.fig_format}')
            else:
                print("The specified save directory doesn't exist. Try again.")
                sys.exit()
        plt.show()

    def plot_spiking_profile(self, **kwargs):
        """
        Description
        ----------
        This method plots the waveform on the peak amplitude channel and several adjacent
        ones.
        ----------

        Parameters
        ----------
        **kwargs (dictionary)
        unit_id (int)
            Unit number in Kilosort output files; defaults to 0.
        peak_ch (int)
            Peak channel of the chosen unit; defaults to 2.
        n_channels (int)
            Number of relevant channels around the peak channel; defaults to 2.
        spike_num (int)
            Maximum number os spikes to take; defaults to 1000.
        sample_num (int)
           Total number of samples to consider in window; defaults to 82.
        bit_volts (float)
            NPX AP has a resolution of 2.3438 µV/bit; defaults to 2.3438.
        saved_file_name (str)
            The name of the saved figure file; defaults to 'waveform_shape'.
        ----------

        Returns
        ----------
        waveform_shape (fig)
            A plot of waveform shapes on different channels.
        ----------
        """

        unit_id = kwargs['unit_id'] if 'unit_id' in kwargs.keys() and type(kwargs['unit_id']) == int else 0
        peak_ch = kwargs['peak_ch'] if 'peak_ch' in kwargs.keys() and type(kwargs['peak_ch']) == int else 2
        n_channels = kwargs['n_channels'] if 'n_channels' in kwargs.keys() and type(kwargs['n_channels']) == int else 2
        spike_num = kwargs['spike_num'] if 'spike_num' in kwargs.keys() and type(kwargs['spike_num']) == int else 1000
        sample_num = kwargs['sample_num'] if 'sample_num' in kwargs.keys() and type(kwargs['sample_num']) == int else 82
        bit_volts = kwargs['bit_volts'] if 'bit_volts' in kwargs.keys() and type(kwargs['bit_volts']) == float else 2.3438
        saved_file_name = kwargs['saved_file_name'] if 'saved_file_name' in kwargs.keys() and type(kwargs['saved_file_name']) == str else 'waveform_shape'

        if os.path.exists(self.kilosort_output_dir):
            # find .bin file
            for file in os.listdir(self.kilosort_output_dir):
                if '.bin' in file:
                    # load .npy files
                    spike_times = np.load('{}{}spike_times.npy'.format(self.kilosort_output_dir, os.sep))
                    spike_clusters = np.load('{}{}spike_clusters.npy'.format(self.kilosort_output_dir, os.sep))

                    # load raw .bin file
                    npx_recording = np.memmap('{}{}{}'.format(self.kilosort_output_dir, os.sep, file), mode='r', dtype=np.int16, order='C')
                    npx_samples = npx_recording.shape[0] // 385
                    raw_data = npx_recording.reshape((385, npx_samples), order='F')
                    del npx_recording

                    # get waveforms for particular unit
                    cluster_indices = np.where(spike_clusters == unit_id)[0]
                    spikes = np.take(spike_times, cluster_indices)

                    waveforms = np.empty((spike_num, raw_data.shape[0], sample_num))
                    waveforms[:] = np.nan

                    spikes_selected = spikes[np.logical_and(spikes >= 20, spikes <= (raw_data.shape[1] - (sample_num - 20)))]

                    np.random.shuffle(spikes_selected)

                    for wv_idx, peak_time in enumerate(spikes_selected[:spike_num]):
                        start = int(peak_time - 20)
                        end = start + sample_num
                        waveforms[wv_idx, :, :] = raw_data[:, start:end] * bit_volts

                    # get mean waveforms for peak channel and N channels around it
                    mean_waveforms = np.zeros(((n_channels * 2) + 1, sample_num))
                    for idx, ch in enumerate(range(-n_channels, n_channels + 1)):
                        mean_waveforms[idx, :] = waveforms[:, peak_ch + ch, :].mean(axis=0)

                    # plot (nb: currently only works for 2 channels around peak!)
                    acceptable_subplots = [(0, 0), (1, 1), (2, 0), (3, 1), (4, 0)]
                    fig, ax = plt.subplots(5, 2, figsize=(5, 6), dpi=300)
                    for i in range(5):
                        for j in range(2):
                            ax[i, j].axis('off')
                            if (i, j) in acceptable_subplots:
                                idx = acceptable_subplots.index((i, j))
                                ax[i, j].plot(mean_waveforms[idx, :], c='#000000')
                    ax[2, 0].hlines(y=-200, xmin=20, xmax=36, linewidth=2, color='#000000')
                    if self.save_fig:
                        if os.path.exists(self.save_dir):
                            fig.savefig(f'{self.save_dir}{os.sep}{saved_file_name}.{self.fig_format}')
                        else:
                            print("The specified save directory doesn't exist. Try again.")
                            sys.exit()
                    plt.show()

            else:
                print("No .bin file in the kilosort directory. Try again.")
                sys.exit()
        else:
            print("The specified kilosort directory doesn't exist. Try again.")
            sys.exit()

    def plot_spiking_cch(self, **kwargs):
        """
        Description
        ----------
        Plot spiking cross-correlogram.
        ----------

        Parameters
        ----------
        **kwargs (dictionary)
        add_prob (bool)
            To add probabilities on plot; defaults to False.
        prob_cmap (str)
            Colormap for probabilities; defaults to 'seismic_r'.
        fig_size_of_choice (tuple)
            Size choice for the CCH figure; defaults to (8, 6).
        cch_color_alpha (dict)
            The color and transparency of CCH; defaults to {'c': '#080808', 'alpha': .4}.
        mid_vline (bool)
            To include the mid CCH vertical line or not; defaults to True.
        mid_vline_params (dict)
            The parameters of the mid vertical line; defaults to {'ls': ':', 'lw': 1, 'c': '#000000', 'alpha': .4}.
        to_normalize (bool)
            To normalize the CCH by the number of spikes of the lower spiking unit; defaults to False.
        plot_pair_position (bool)
            Plot the anatomical positions of the cell pair in questions; defaults to False.
        plot_cch_convolved (bool)
            Plot the convolved CCH; defaults to False.
        ----------

        Returns
        ----------
        cch (fig)
            A plot of CCH.
        ----------
        """

        add_prob = kwargs['add_prob'] if 'add_prob' in kwargs.keys() and type(kwargs['add_prob']) == bool else False
        prob_cmap = kwargs['prob_cmap'] if 'prob_cmap' in kwargs.keys() and type(kwargs['prob_cmap']) == str else 'seismic_r'
        fig_size_of_choice = kwargs['fig_size_of_choice'] if 'fig_size_of_choice' in kwargs.keys() and type(kwargs['fig_size_of_choice']) == tuple else (8, 6)
        cch_color_alpha = kwargs['cch_color_alpha'] if 'cch_color_alpha' in kwargs.keys() \
                                                       and type(kwargs['cch_color_alpha']) == dict else {'c': '#080808', 'alpha': .4}
        mid_vline = kwargs['mid_vline'] if 'mid_vline' in kwargs.keys() and type(kwargs['mid_vline']) == bool else True
        mid_vline_params = kwargs['mid_vline_params'] if 'mid_vline_params' in kwargs.keys() \
                                                         and type(kwargs['mid_vline_params']) == dict else {'ls': ':', 'lw': 1, 'c': '#000000', 'alpha': .4}
        to_normalize = kwargs['to_normalize'] if 'to_normalize' in kwargs.keys() and type(kwargs['to_normalize']) == bool else False
        plot_pair_position = kwargs['plot_pair_position'] if 'plot_pair_position' in kwargs.keys() and type(kwargs['plot_pair_position']) == bool else False
        plot_cch_convolved = kwargs['plot_cch_convolved'] if 'plot_cch_convolved' in kwargs.keys() and type(kwargs['plot_cch_convolved']) == bool else False

        # find relevant information for plotting
        with open(f'{self.cch_data_dir}{os.sep}{self.area_file}', 'r') as json_pairs_file:
            data = json.load(json_pairs_file)
            for animal_session in data.keys():
                if self.cell_pair_id in data[animal_session].keys():
                    as_id = animal_session
                    rat_id = [rat for rat in self.animal_ids.keys() if rat in as_id][0]
                    break
            else:
                print('The requested cell pair cannot be found in this file, try again!')
                sys.exit()

        n_spikes_1 = np.asarray(data[as_id][self.cell_pair_id]['data'], dtype=np.float32)[0, 3]
        n_spikes_2 = np.asarray(data[as_id][self.cell_pair_id]['data'], dtype=np.float32)[0, 4]
        print(f"Cell 1 had {n_spikes_1} spikes and cell 2 had {n_spikes_2} spikes")

        if 'distal' in as_id:
            if rat_id != 'johnjohn':
                as_id_other_bank = as_id.replace('distal', 'intermediate')
            else:
                as_id_other_bank = 'johnjohn_230520_intermediate'
        else:
            if rat_id != 'johnjohn':
                as_id_other_bank = as_id.replace('intermediate', 'distal')
            else:
                as_id_other_bank = 'johnjohn_210520_distal'

        if to_normalize:
            cch = np.asarray(data[as_id][self.cell_pair_id]['data'], dtype=np.float32)[:, 0] \
                  / np.min([np.asarray(data[as_id][self.cell_pair_id]['data'], dtype=np.float32)[0, 3],
                            np.asarray(data[as_id][self.cell_pair_id]['data'], dtype=np.float32)[0, 4]])
            cch_convolved = np.asarray(data[as_id][self.cell_pair_id]['data'], dtype=np.float32)[:, 1] \
                            / np.min([np.asarray(data[as_id][self.cell_pair_id]['data'], dtype=np.float32)[0, 3],
                                      np.asarray(data[as_id][self.cell_pair_id]['data'], dtype=np.float32)[0, 4]])
        else:
            cch = np.asarray(data[as_id][self.cell_pair_id]['data'], dtype=np.float32)[:, 0]
            cch_convolved = np.asarray(data[as_id][self.cell_pair_id]['data'], dtype=np.float32)[:, 1]

        if add_prob:
            normalize = plt.Normalize(vmin=-8, vmax=0)
            cmap = plt.cm.get_cmap(prob_cmap)
            if data[as_id][self.cell_pair_id]['excitatory']:
                probabilities = np.asarray(data[as_id][self.cell_pair_id]['data'], dtype=np.float32)[:, 2]
            else:
                probabilities = 1 - np.asarray(data[as_id][self.cell_pair_id]['data'], dtype=np.float32)[:, 2]
            log_probabilities = np.log10(probabilities)

        fig = plt.figure(figsize=fig_size_of_choice)
        ax = fig.add_subplot(1, 1, 1)
        if plot_pair_position:
            spc = pd.read_csv(self.sp_profiles_csv)
            filtered_spc = spc[(spc['session_id'] == as_id) | (spc['session_id'] == as_id_other_bank)
                               & ~(spc['cluster_id'] == self.cell_pair_id.split('-')[0])
                               & ~(spc['cluster_id'] == self.cell_pair_id.split('-')[1])].iloc[:, -3:]
            corr_cls = spc[(spc['cluster_id'] == self.cell_pair_id.split('-')[0])
                           | (spc['cluster_id'] == self.cell_pair_id.split('-')[1])]

            unit1_position = corr_cls[corr_cls['cluster_id'] == self.cell_pair_id.split('-')[0]].iloc[:, -3:]
            unit2_position = corr_cls[corr_cls['cluster_id'] == self.cell_pair_id.split('-')[1]].iloc[:, -3:]

            color_1 = self.profile_colors[spc.loc[spc[(spc['session_id'] == as_id) & (spc['cluster_id'] == self.cell_pair_id.split('-')[0])].index, 'profile'].to_string()[-2:]]
            color_2 = self.profile_colors[spc.loc[spc[(spc['session_id'] == as_id) & (spc['cluster_id'] == self.cell_pair_id.split('-')[1])].index, 'profile'].to_string()[-2:]]

            inset_axes = fig.add_axes(rect=[.475, .475, .5, .5], projection='3d')
            inset_axes.scatter(filtered_spc.iloc[:, 0], filtered_spc.iloc[:, 1], filtered_spc.iloc[:, 2], color='#808080', alpha=.1)
            inset_axes.scatter(unit1_position.iloc[:, 0], unit1_position.iloc[:, 1], unit1_position.iloc[:, 2], color=color_1, alpha=1, s=30)
            inset_axes.scatter(unit2_position.iloc[:, 0], unit2_position.iloc[:, 1], unit2_position.iloc[:, 2], color=color_2, alpha=1, s=30)
            inset_axes.plot3D([unit1_position.iloc[0, 0], unit2_position.iloc[0, 0]],
                              [unit1_position.iloc[0, 1], unit2_position.iloc[0, 1]],
                              [unit1_position.iloc[0, 2], unit2_position.iloc[0, 2]], ls='-', c='#000000')
            if rat_id in ['kavorka', 'frank', 'johnjohn']:
                inset_axes.invert_xaxis()
            inset_axes.set_xlabel('AP (mm)')
            inset_axes.set_ylabel('ML (mm)')
            inset_axes.set_zlabel('DV (mm)')
            inset_axes.view_init(elev=self.elev_azimuth_3d[rat_id][0], azim=self.elev_azimuth_3d[rat_id][1])
        ax.plot(list(range(cch.shape[0])), cch, drawstyle='steps-mid', color=cch_color_alpha['c'], alpha=cch_color_alpha['alpha'])
        if plot_cch_convolved:
            ax.plot(list(range(cch.shape[0])), cch_convolved, color=cch_color_alpha['c'], alpha=cch_color_alpha['alpha'])
        if add_prob:
            for ii in range(cch.shape[0]):
                ax.fill_between(x=[ii], y1=[0], y2=[cch[ii]], step='mid', color=cmap(normalize(log_probabilities[ii])))
        if mid_vline:
            ax.axvline(x=50, ls=mid_vline_params['ls'], lw=mid_vline_params['lw'], color=mid_vline_params['c'], alpha=mid_vline_params['alpha'])
        ax.set_title(f'{as_id}_{self.cell_pair_id}')
        ax.set_xticks(np.arange(0, 101, 25))
        ax.set_xticklabels(np.arange(-20, 21, 10))
        ax.set_xlabel('Time offset (ms)')
        if to_normalize:
            ax.set_ylabel('Normalized cross-correlation (A.U.)')
        else:
            ax.set_ylabel('Number of spike coincidences')
        if plot_pair_position:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
        if add_prob:
            cax, _ = cbar.make_axes(ax)
            color_bar = cbar.ColorbarBase(cax, cmap=cmap, norm=normalize)
            color_bar.set_label('log$_{10}$(p-value)')
            color_bar.ax.tick_params(size=0)
        if self.save_fig:
            if os.path.exists(self.save_dir):
                fig.savefig(f'{self.save_dir}{os.sep}{as_id}_{self.cell_pair_id}.{self.fig_format}', dpi=300)
        plt.show()

    def plot_cch_pairs_anatomically(self, **kwargs):
        """
        Description
        ----------
        Plot anatomical distribution of all significant CCH pairs in one animal.
        ----------

        Parameters
        ----------
        **kwargs (dictionary)
        chosen_rat (str)
            The animal to plot the cell pairs for; defaults to 'kavorka'.
        special_pairs (list / bool)
            Special pairs to be plotted; defaults to False.
        ----------

        Returns
        ----------
        anatomical_cch_distribution (fig)
            A plot of of the CCH pairs anatomical positions.
        ----------
        """

        chosen_rat = kwargs['chosen_rat'] if 'chosen_rat' in kwargs.keys() and kwargs['chosen_rat'] in self.animal_ids.keys() else 'kavorka'
        special_pairs = kwargs['special_pairs'] if 'special_pairs' in kwargs.keys() and type(kwargs['special_pairs']) == list else False

        if type(special_pairs) == list:
            special_pairs_split = []
            for pair in special_pairs:
                for one_cl in pair.split('-'):
                    special_pairs_split.append(one_cl)

        # load the data
        with open(self.cch_summary_file, 'r') as summary_file:
            cch_summary_dict = json.load(summary_file)

        point_3d_dict = {'other': {area: {'cl': [], 'X': [], 'Y': [], 'Z': []} for area in self.areas_by_rat[chosen_rat]},
                         'special': {area: {'cl': [], 'X': [], 'Y': [], 'Z': []} for area in self.areas_by_rat[chosen_rat]}}
        line_3d_dict = {}

        for area_area in cch_summary_dict.keys():
            cl_areas = [area_area[:len(area_area)//2], area_area[len(area_area)//2:]]
            if chosen_rat in self.area_rats[area_area]:
                for a_s in cch_summary_dict[area_area][chosen_rat].keys():
                    a_s_split = a_s.split('_')
                    if cl_areas[0] != cl_areas[1]:
                        correct_session_name = ''
                        for num in range(1, 5):
                            if os.path.exists(f'{self.data_file_dir}{os.sep}clean_data_{a_s_split[0]}_{a_s_split[1]}_s{num}_{a_s_split[2]}_dark_reheaded_XYZeuler_notricks.pkl'):
                                correct_session_name = f'{self.data_file_dir}{os.sep}clean_data_{a_s_split[0]}_{a_s_split[1]}_s{num}_{a_s_split[2]}_dark_reheaded_XYZeuler_notricks.pkl'
                                break

                        cat1_cl = select_clusters.ClusterFinder(session=correct_session_name,
                                                                cluster_groups_dir=self.cluster_groups_dir,
                                                                sp_profiles_csv=self.sp_profiles_csv).get_desired_clusters(filter_by_area=[cl_areas[0]],
                                                                                                                           filter_by_cluster_type='good')
                    for cl in cch_summary_dict[area_area][chosen_rat][a_s]['clusters'].keys():
                        if type(special_pairs) == list and cl in special_pairs_split:
                            dict_of_choice = 'special'
                        else:
                            dict_of_choice = 'other'
                        if cl_areas[0] == cl_areas[1]:
                            cl_area = cl_areas[0]
                        else:
                            if cl in cat1_cl:
                                cl_area = cl_areas[0]
                            else:
                                cl_area = cl_areas[1]
                        if cl not in point_3d_dict[dict_of_choice][cl_area]['cl']:
                            point_3d_dict[dict_of_choice][cl_area]['cl'].append(cl)
                            point_3d_dict[dict_of_choice][cl_area]['X'].append(cch_summary_dict[area_area][chosen_rat][a_s]['clusters'][cl]['XYZ'][0])
                            point_3d_dict[dict_of_choice][cl_area]['Y'].append(cch_summary_dict[area_area][chosen_rat][a_s]['clusters'][cl]['XYZ'][1])
                            point_3d_dict[dict_of_choice][cl_area]['Z'].append(cch_summary_dict[area_area][chosen_rat][a_s]['clusters'][cl]['XYZ'][2])

        for area_area in cch_summary_dict.keys():
            if chosen_rat in self.area_rats[area_area]:
                for a_s in cch_summary_dict[area_area][chosen_rat].keys():
                    for pair_idx, pair in enumerate(cch_summary_dict[area_area][chosen_rat][a_s]['pairs']):
                        cl1, cl2 = pair.split('-')
                        direction = cch_summary_dict[area_area][chosen_rat][a_s]['directionality'][pair_idx]
                        if direction == -1:
                            direction_cls = [cl1, cl2]
                        else:
                            direction_cls = [cl2, cl1]
                        line_3d_dict[pair] = [cch_summary_dict[area_area][chosen_rat][a_s]['clusters'][direction_cls[0]]['XYZ'],
                                              cch_summary_dict[area_area][chosen_rat][a_s]['clusters'][direction_cls[1]]['XYZ'],
                                              cch_summary_dict[area_area][chosen_rat][a_s]['strength'][pair_idx],
                                              cch_summary_dict[area_area][chosen_rat][a_s]['type'][pair_idx]]

        line_types = {'excitatory': '-', 'inhibitory': '-.'}
        fig = plt.figure(figsize=(6, 8), dpi=500)
        ax = fig.add_subplot(projection='3d')
        for pair in line_3d_dict.keys():
            pair_data = line_3d_dict[pair]
            ax.plot([pair_data[0][0], pair_data[1][0]], [pair_data[0][1], pair_data[1][1]], [pair_data[0][2], pair_data[1][2]],
                    ls=line_types[pair_data[3]], lw=pair_data[2]*3, color='#000000')
        for area in point_3d_dict['other'].keys():
            ax.scatter(point_3d_dict['other'][area]['X'], point_3d_dict['other'][area]['Y'], point_3d_dict['other'][area]['Z'],
                       color=self.area_colors[area], alpha=.8)
        for area in point_3d_dict['special'].keys():
            if len(point_3d_dict['special'][area]['cl']) > 0:
                ax.scatter(point_3d_dict['special'][area]['X'], point_3d_dict['special'][area]['Y'], point_3d_dict['special'][area]['Z'],
                           color=self.area_colors[area], ec='#000000', alpha=1)
        ax.view_init(elev=self.elev_azimuth_3d[chosen_rat][0], azim=self.elev_azimuth_3d[chosen_rat][1])
        ax.set_title(f'#{self.animal_ids[chosen_rat]}', pad=0)
        ax.invert_xaxis()
        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        ax.set_xlabel('AP (mm)')
        ax.set_ylabel('ML (mm)')
        ax.set_zlabel('DV (mm)')
        ax.grid(False)
        plt.show()
        if self.save_fig:
            if os.path.exists(self.save_dir):
                fig.savefig(f'{self.save_dir}{os.sep}cch_anatomical_{chosen_rat}.{self.fig_format}', dpi=500)
            else:
                print("Specified save directory doesn't exist. Try again.")
                sys.exit()

