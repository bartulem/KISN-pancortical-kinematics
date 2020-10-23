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
    def __init__(self, session=0, cluster_name=0):
        self.session = session
        self.cluster_name = cluster_name

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
        save_fig (bool)
            Save the figure or not; defaults to False.
        fig_format (str)
            The format of the figure; defaults to 'png'.
        save_dir (bool)
            Directory to save the figure in; defaults to '/home/bartulm/Downloads'.
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

        save_fig = kwargs['save_fig'] if 'save_fig' in kwargs.keys() and type(kwargs['save_fig']) == bool else False
        fig_format = kwargs['fig_format'] if 'fig_format' in kwargs.keys() and type(kwargs['fig_format']) == str else 'png'
        save_dir = kwargs['save_dir'] if 'save_dir' in kwargs.keys() and type(kwargs['save_dir']) == str else '/home/bartulm/Downloads'
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
        if save_fig:
            if os.path.exists(save_dir):
                fig.savefig(f'{save_dir}{os.sep}{self.cluster_name}_wh_noise_peth.{fig_format}')
            else:
                print("Specified save directory doesn't exist. Try again.")
                sys.exit()
        plt.show()
