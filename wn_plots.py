# -*- coding: utf-8 -*-

"""

@author: bartulem

Plot white noise characteristics.

"""

import os
import sys
import scipy
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


class WhiteNoise:

    def __init__(self, white_noise_file=0):
        self.white_noise_file = white_noise_file

    def data_plotter(self, **kwargs):
        """
        Description
        ----------
        This method plots the spectogram of the white noise recorded by the US microphone.
        ----------

        Parameters
        ----------
        **kwargs (dictionary)
        save_fig (bool)
            Save the figure or not; defaults to False.
        fig_format (str)
            The format of the figure; defaults to 'png'.
        save_dir (bool)
            Directory to save the figure in; defaults to '/home/bartulm/Downloads'.
        spec_cmap (str)
            Color of the PETH; defaults to 'cividis'.
        raw_data_color (str)
            Color of the other plots; defaults to '#000000'.
        axins_color (str)
            Color of the ticks and labels on the inset plots; defaults to '#FFFFFF'.
        ----------

        Returns
        ----------
        white_noise_quality (fig)
            White noise spectogram with raw data.
        ----------
        """

        save_fig = kwargs['save_fig'] if 'save_fig' in kwargs.keys() and type(kwargs['save_fig']) == bool else False
        fig_format = kwargs['fig_format'] if 'fig_format' in kwargs.keys() and type(kwargs['fig_format']) == str else 'png'
        save_dir = kwargs['save_dir'] if 'save_dir' in kwargs.keys() and type(kwargs['save_dir']) == str else '/home/bartulm/Downloads'
        spec_cmap = kwargs['spec_cmap'] if 'spec_cmap' in kwargs.keys() and type(kwargs['spec_cmap']) == str else 'cividis'
        raw_data_color = kwargs['raw_data_color'] if 'raw_data_color' in kwargs.keys() and type(kwargs['raw_data_color']) == str else '#000000'
        axins_color = kwargs['axins_color'] if 'axins_color' in kwargs.keys() and type(kwargs['axins_color']) == str else '#FFFFFF'

        if os.path.exists(self.white_noise_file) and self.white_noise_file[-4:] == '.wav':
            # load the .wav file data
            sampling_rate, data = wavfile.read(self.white_noise_file)

            # compute fft on the white noise data
            n = data.shape[0]
            period = 1/sampling_rate
            yf = scipy.fft.fft(data)
            xf_plot = np.linspace(0.0, 1.0/(2.0*period), n//2)
            yf_plot = 2.0/n * np.abs(yf[:n//2])

            # make figure
            fig, ax = plt.subplots(1, 1, figsize=(12, 6), dpi=300)
            ax1 = plt.subplot(111)
            pxx, freq, t, cax = ax1.specgram(data, Fs=sampling_rate, vmin=-10, cmap=spec_cmap)
            ax1.axhline(y=27e3, xmin=.106, xmax=.106+(5/(ax1.get_xlim()[1])), c=axins_color)
            ax1.text(x=6.21, y=23e3, s='Noise stim', c=axins_color)
            ax1.set_xlabel('Time (s)')
            ax1.set_ylabel('Frequency (kHz)')
            ax1.set_ylim(0, 1e5)
            ax1.set_yticks(ax1.get_yticks())
            ax1.set_yticklabels([f'{int(y_label / 1e3)}' for y_label in ax1.get_yticks()])
            cbar = fig.colorbar(cax, shrink=.5)
            cbar.set_label('10$log_{10}$ Spectral Power [dB]')
            cbar.ax.tick_params(size=0)

            # plot raw data inset
            axins1 = inset_axes(ax1, width='35%', height='35%', loc=2)
            axins1.plot(data[3009837-(sampling_rate // 50):3009837+(sampling_rate // 50)] / np.max(data), c=raw_data_color)
            axins1.set_xlabel('Time relative to noise stim (ms)', c=axins_color)
            axins1.tick_params(axis='both', colors=axins_color)
            axins1_xticks = np.arange(0, 25e3, 5e3)
            axins1.set_xticks(axins1_xticks)
            axins1.set_xticklabels(['-20', '-10', '0', '10', '20'])
            axins1.set_yticks([])
            axins1.text(x=1, y=.8, s='Raw signal (a.u.)')

            # plot magnitude inset
            axins2 = inset_axes(ax1, width='35%', height='35%', loc=1)
            axins2.plot(xf_plot[:6000000], yf_plot[:6000000], c=raw_data_color)
            axins2.set_xlabel('Frequency (kHz)', c=axins_color)
            axins2.text(x=5e4, y=58, s='Magnitude (a.u.)')
            axins2.tick_params(axis='both', colors=axins_color)
            axins2_xticks = np.arange(0, 12e4, 2e4)
            axins2.set_xticks(axins2_xticks)
            axins2.set_xticklabels([f'{int(x2_label / 1e3)}' for x2_label in axins2_xticks])
            axins2.set_yticks([])
            if save_fig:
                if os.path.exists(save_dir):
                    fig.savefig(f'{save_dir}{os.sep}white_noise_characteristics.{fig_format}', )
                else:
                    print("Specified save directory doesn't exist. Try again.")
                    sys.exit()
            plt.show()

        else:
            print(f"Invalid location or no .wav file provided. Please try again.")
            sys.exit()
