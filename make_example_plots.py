# -*- coding: utf-8 -*-

"""

@author: bartulem

Make example plots.

"""

from neural_activity import Spikes
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from scipy.stats import sem


class PlotExamples:
    def __init__(self, session=0, cluster_name=0, kilosort_output_dir=0,
                 save_fig=False, fig_format='png', save_dir='/home/bartulm/Downloads'):
        self.session = session
        self.cluster_name = cluster_name
        self.kilosort_output_dir = kilosort_output_dir
        self.save_fig = save_fig
        self.fig_format = fig_format
        self.save_dir = save_dir

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
        session_name, peth, raster, peth_beh = Spikes(input_file=self.session).get_peths(get_clusters=[self.cluster_name], smooth=True, smooth_sd=3, raster=True, beh_raster='speeds')

        # get means/SEMs for spikes/behavior
        peth_mean = peth[self.cluster_name]['peth'].mean(axis=0)
        peth_sem = sem(a=peth[self.cluster_name]['peth'], axis=0, nan_policy='omit')
        beh_mean = np.nanmean(peth_beh, axis=0)
        beh_sem = sem(a=peth_beh, axis=0, nan_policy='omit')

        # plot figure
        fig = plt.figure(figsize=(5, 5), dpi=300, tight_layout=True)
        ax1 = fig.add_subplot(111, label='1')
        ax2 = fig.add_subplot(111, label='2', frame_on=False)
        ax3 = fig.add_subplot(111, label='3', frame_on=False)
        ax1.eventplot(raster[self.cluster_name], colors=raster_color, lineoffsets=1, linelengths=1, linewidths=.1)
        ax2.plot(range(peth[self.cluster_name]['peth'].shape[1]), peth_mean, color=peth_color)
        ax2.fill_between(range(peth[self.cluster_name]['peth'].shape[1]), peth_mean+peth_sem, peth_mean-peth_sem, color=peth_color, alpha=.3)
        ax2.axvspan(200, 300, alpha=0.2, color=raster_color)
        ax3.plot(range(peth_beh.shape[1]), beh_mean, color=beh_color)
        ax3.fill_between(range(peth_beh.shape[1]), beh_mean+beh_sem, beh_mean-beh_sem, color=beh_color, alpha=.3)
        ax1.set_xticks([])
        ax1.set_yticks([])
        ax2.set_xticks(np.arange(0, 401, 50))
        ax2.set_xticklabels(np.arange(-10, 10.1, 2.5))
        ax2.set_xlabel('Time relative to sound onset (s)')
        ax2.yaxis.tick_right()
        ax2.yaxis.set_label_position('right')
        ax2.set_ylabel('Firing rate (spikes/s)')
        ax3.set_xticks([])
        ax3.yaxis.tick_left()
        ax3.yaxis.set_label_position('left')
        ax3.tick_params(axis='y', colors=beh_color)
        ax3.set_ylabel('Speed (cm/s)', color=beh_color)
        if self.save_fig:
            if os.path.exists(self.save_dir):
                fig.savefig(f'{self.save_dir}{os.sep}{self.cluster_name}_wh_noise_peth.{self.fig_format}')
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
                    mean_waveforms = np.zeros(((n_channels*2)+1, sample_num))
                    for idx, ch in enumerate(range(-n_channels, n_channels+1)):
                        mean_waveforms[idx, :] = waveforms[:, peak_ch+ch, :].mean(axis=0)

                    # plot (nb: currently only works for 2 channels around peak!)
                    acceptable_subplots = [(0, 0), (1, 1), (2, 0), (3, 1), (4, 0)]
                    fig, ax = plt.subplots(5, 2, figsize= (5, 6), dpi=300)
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

