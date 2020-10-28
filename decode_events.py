# -*- coding: utf-8 -*-

"""

@author: bartulem

Decode events like sound stimulation or luminance.

"""

import numpy as np
import neural_activity
from select_clusters import ClusterFinder
from sessions2load import Session


class Decoder:

    def __init__(self, input_file=''):
        self.input_file = input_file

    def decode_sound_stim(self, **kwargs):
        """
        Description
        ----------
        This method uses a simple nearest neighbor decoder to classify sound stimulation
        events.
        ----------

        Parameters
        ----------
        **kwargs (dictionary)
        cluster_groups_dir (str)
            The directory with the cluster_groups .json file; defaults to '/home/bartulm/Insync/mimica.bartul@gmail.com/OneDrive/Work/data/posture_2020/cluster_groups_info'.
        sp_profiles_csv (str)
            The directory with the sp_profiles .csv file; defaults to '/home/bartulm/Insync/mimica.bartul@gmail.com/OneDrive/Work/data/posture_2020/spiking_profiles/spiking_profiles.csv'.
        cluster_areas (list)
            Cluster area(s) of choice; defaults to ['A'].
        cluster_type (str)
            Cluster type of choice; defaults to True.
        number_of_decoding_per_run (int)
            The number of times to decode event per run; defaults to 10.
        decoding_cell_number_array (np.ndarray):
            An array of numbers of cells to decode with; defaults to np.array([5, 10, 20, 50, 100]).
        shuffle_num (int)
            Number of shuffles; defaults to 1000.
        to_smooth (bool)
            Smooth PETHs; defaults to False.
        smooth_sd (int)
            The SD of the smoothing window; defaults to 1 (bin).
        smooth_axis (int)
            The smoothing axis in a 2D array; defaults to 1 (smooths within rows).
        ----------

        Returns
        ----------
        file_info (str)
            The shortened version of the file name.
        ----------
        """

        cluster_groups_dir = kwargs['cluster_groups_dir'] if 'cluster_groups_dir' in kwargs.keys() and type(kwargs['cluster_groups_dir']) == str \
            else '/home/bartulm/Insync/mimica.bartul@gmail.com/OneDrive/Work/data/posture_2020/cluster_groups_info'
        sp_profiles_csv = kwargs['sp_profiles_csv'] if 'sp_profiles_csv' in kwargs.keys() and type(kwargs['sp_profiles_csv']) == str \
            else '/home/bartulm/Insync/mimica.bartul@gmail.com/OneDrive/Work/data/posture_2020/spiking_profiles/spiking_profiles.csv'
        cluster_areas = kwargs['cluster_areas'] if 'cluster_areas' in kwargs.keys() and type(kwargs['cluster_areas']) == list else ['A']
        cluster_type = kwargs['cluster_type'] if 'cluster_type' in kwargs.keys() and type(kwargs['cluster_type']) == str else True
        number_of_decoding_per_run = kwargs['number_of_decoding_per_run'] if 'number_of_decoding_per_run' in kwargs.keys() \
                                                                             and type(kwargs['number_of_decoding_per_run']) == int else 10
        decoding_cell_number_array = kwargs['decoding_cell_number_array'] if 'decoding_cell_number_array' in kwargs.keys() \
                                                                             and type(kwargs['decoding_cell_number_array']) == np.array else np.array([5, 10, 20, 50, 100])
        shuffle_num = kwargs['shuffle_num'] if 'shuffle_num' in kwargs.keys() and type(kwargs['shuffle_num']) == int else 1000
        to_smooth = kwargs['to_smooth'] if 'to_smooth' in kwargs.keys() and type(kwargs['to_smooth']) == bool else False
        smooth_sd = kwargs['smooth_sd'] if 'smooth_sd' in kwargs.keys() and type(kwargs['smooth_sd']) == int else 1
        smooth_axis = kwargs['smooth_axis'] if 'smooth_axis' in kwargs.keys() and type(kwargs['smooth_axis']) == int else 1

        # choose clusters you'd like to decode with
        chosen_clusters = ClusterFinder(session=self.input_file,
                                        cluster_groups_dir=cluster_groups_dir,
                                        sp_profiles_csv=sp_profiles_csv).get_desired_clusters(filter_by_area=cluster_areas,
                                                                                              filter_by_cluster_type=cluster_type)
        # get framerate and total frame count
        file_name, extracted_frame_info = Session(session=self.input_file).data_loader(extract_variables=['framerate', 'total_frame_num', 'imu_sound'])

        # get activity dictionary
        file_id, activity_dictionary = neural_activity.Spikes(input_file=self.input_file).convert_activity_to_frames_with_shuffles(get_clusters=chosen_clusters,
                                                                                                                                   to_shuffle=True)
        # get all cluster IDs
        all_cluster_names = list(activity_dictionary.keys())

        # conduct decoding
        decoding_accuracy = np.zeros((decoding_cell_number_array.shape[0], number_of_decoding_per_run))
        shuffled_decoding_accuracy = np.zeros((shuffle_num, decoding_cell_number_array.shape[0], number_of_decoding_per_run))
        for decode_num in number_of_decoding_per_run:
            for ca_idx, cell_amount in enumerate(decoding_cell_number_array):
                cells_array = np.zeros((cell_amount, extracted_frame_info['total_frame_num']))
                shuffled_cells_array = np.zeros((shuffle_num, cell_amount, extracted_frame_info['total_frame_num']))

                # shuffle cluster names
                np.random.shuffle(all_cluster_names)

                # select clusters
                selected_clusters = all_cluster_names[:cell_amount]

                # get all cell / shuffled data in their respective arrays
                for idx, selected_cluster in enumerate(selected_clusters):
                    cells_array[idx, :] = activity_dictionary[selected_cluster]['activity']
                    for shuffle_idx in range(shuffle_num):
                        shuffled_cells_array[shuffle_idx, idx, :] = activity_dictionary[selected_cluster]['shuffled'][shuffle_idx, :]

                # smooth spike trains if desired
                if to_smooth:
                    cells_array = neural_activity.gaussian_smoothing(array=cells_array, sigma=smooth_sd, axis=smooth_axis)
                    shuffled_cells_array = neural_activity.gaussian_smoothing(array=shuffled_cells_array, sigma=smooth_sd, axis=smooth_axis+1)
