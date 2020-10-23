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
from scipy.stats import sem


class PlotGroupResults:
    def __init__(self, session_list=[], cluster_groups_dir=0):
        self.session_list = session_list
        self.cluster_groups_dir = cluster_groups_dir

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
            The bin size of the PETH; defaults to 500 (ms).
        window_size (int/float)
            The unilateral window size; defaults to 0.5 (seconds).
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
        bin_size_ms = kwargs['bin_size_ms'] if 'bin_size_ms' in kwargs.keys() and type(kwargs['bin_size_ms']) == int else 500
        window_size = kwargs['window_size'] if 'window_size' in kwargs.keys() and type(kwargs['window_size']) == int else .5
        save_fig = kwargs['save_fig'] if 'save_fig' in kwargs.keys() and type(kwargs['save_fig']) == bool else False
        fig_format = kwargs['fig_format'] if 'fig_format' in kwargs.keys() and type(kwargs['fig_format']) == str else 'png'
        save_dir = kwargs['save_dir'] if 'save_dir' in kwargs.keys() and type(kwargs['save_dir']) == str else '/home/bartulm/Downloads'

        if not os.path.exists(self.cluster_groups_dir):
            print(f"Invalid location for directory {self.cluster_groups_dir}. Please try again.")
            sys.exit()

        sound_stim_data = {}
        if len(self.session_list) == 0:
            for one_session in self.session_list:
                if os.path.exists(one_session):
                    relevant_session_clusters = ClusterFinder(session=one_session,
                                                              cluster_groups_dir=self.cluster_groups_dir).get_desired_clusters(filter_by_area=relevant_areas,
                                                                                                                               filter_by_cluster_type=relevant_cluster_types)
                    session_name, peth = Spikes(input_file=one_session).get_peths(get_clusters=relevant_session_clusters,
                                                                                  bin_size_ms=bin_size_ms,
                                                                                  window_size=window_size)
                    sound_stim_data[session_name] = peth
                else:
                    print(f"Invalid location for file {one_session}. Please try again.")
                    sys.exit()
        else:
            print("No session provided.")
            sys.exit()

        return sound_stim_data
